"""
Training pipeline for GNN models.

Provides train/val/test loops with early stopping, metric computation,
and model checkpointing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    auc,
)
from typing import Optional


class Trainer:
    """
    Trainer for GNN node classification.

    Handles full-batch transductive training with early stopping
    on a configurable validation metric (default: F1).

    Args:
        model: GNN model (nn.Module).
        loss_fn: Loss function.
        optimizer: torch optimizer.
        device: Target device.
        patience: Early stopping patience.
        checkpoint_dir: Directory to save best model.
        tracker: Optional ExperimentTracker for per-epoch logging.
        disable_checkpointing: If True, skip all checkpoint writes.
        early_stop_metric: Validation metric used for early stopping and
            best-checkpoint selection. One of 'f1', 'mcc', 'pr_auc'.
            Default 'f1' (more stable than MCC on imbalanced data).
    """

    _EARLY_STOP_INIT = {"f1": -1.0, "mcc": -2.0, "pr_auc": -1.0}

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "cpu",
        patience: int = 20,
        checkpoint_dir: str = "./results/models",
        tracker=None,
        disable_checkpointing: bool = False,
        early_stop_metric: str = "f1",
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.patience = patience
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.epoch_ckpt_base = Path("./checkpoints")  # per-epoch recovery checkpoints
        self.tracker = tracker  # ExperimentTracker for per-epoch logging
        self.disable_checkpointing = disable_checkpointing

        if early_stop_metric not in self._EARLY_STOP_INIT:
            raise ValueError(
                f"early_stop_metric must be one of {list(self._EARLY_STOP_INIT)}, "
                f"got {early_stop_metric!r}"
            )
        self.early_stop_metric = early_stop_metric

        # Tracking
        self.best_val_score = self._EARLY_STOP_INIT[early_stop_metric]
        self.best_val_mcc = -2.0  # kept for backwards-compat in recovery ckpt
        self.best_epoch = 0
        self.patience_counter = 0
        self.history = {
            "train_loss": [], "val_loss": [],
            "val_f1": [], "val_mcc": [], "val_pr_auc": [],
        }

    def train_epoch(self, data) -> float:
        """Run one training epoch. Returns training loss."""
        self.model.train()
        self.optimizer.zero_grad()

        out = self.model(data.x.to(self.device), data.edge_index.to(self.device))
        mask = data.train_mask.to(self.device)
        loss = self.loss_fn(out[mask], data.y.to(self.device)[mask])

        loss.backward()
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def evaluate(
        self,
        data,
        mask_name: str = "val_mask",
        threshold: Optional[float] = None,
    ) -> dict:
        """
        Evaluate model on a given mask.

        Args:
            data: PyG Data object.
            mask_name: Attribute name of the boolean mask on data.
            threshold: Optional decision threshold on P(class=1). If None,
                uses argmax (default behavior, threshold=0.5 for binary).
                If set (e.g. from calibrate_threshold on val), this is applied
                to the rare-class probability: pred = 1 iff P(y=1) >= threshold.

        Returns dict with loss, f1, mcc, pr_auc. PR-AUC is threshold-independent.
        """
        self.model.eval()
        mask = getattr(data, mask_name).to(self.device)
        out = self.model(data.x.to(self.device), data.edge_index.to(self.device))

        y_true = data.y.to(self.device)[mask]
        loss = self.loss_fn(out[mask], y_true).item()

        labels = y_true.cpu().numpy()
        probs = F.softmax(out[mask], dim=-1)[:, 1].cpu().numpy()

        if threshold is None:
            preds = out[mask].argmax(dim=-1).cpu().numpy()
        else:
            preds = (probs >= threshold).astype(int)

        f1 = f1_score(labels, preds, pos_label=1, zero_division=0)
        mcc = matthews_corrcoef(labels, preds)

        precision, recall, _ = precision_recall_curve(labels, probs, pos_label=1)
        pr_auc = auc(recall, precision)

        return {"loss": loss, "f1": f1, "mcc": mcc, "pr_auc": pr_auc}

    @torch.no_grad()
    def calibrate_threshold(
        self,
        data,
        mask_name: str = "val_mask",
        n_thresholds: int = 101,
        match_ratio_mask: Optional[str] = None,
        seed: int = 42,
    ) -> dict:
        """
        Find the decision threshold that maximizes F1 on the given mask.

        Sweeps `n_thresholds` values in (0, 1) exclusive, picks the one with
        highest F1. Necessary when train/val/test have different class ratios
        (e.g. Elliptic: train 1:1 post-scenario, test 1:136 natural) — plain
        argmax assumes the test ratio matches training.

        Args:
            data: PyG Data object.
            mask_name: Split to calibrate on (default val_mask).
            n_thresholds: Number of thresholds to sweep in (0, 1).
            match_ratio_mask: If set, downsample the mask to match the class
                ratio of this other mask BEFORE calibrating. Example: calibrate
                on val_mask but with the same prevalence as test_mask, so the
                chosen threshold transfers to test. Essential when val/test
                have very different class ratios (common in temporal splits).
            seed: RNG seed for the resampling, for reproducibility.

        Returns dict: threshold, f1 (at that threshold on the effective mask),
        sweep, and info on the resampling if applied.
        """
        import numpy as _np

        self.model.eval()
        mask = getattr(data, mask_name).to(self.device)
        out = self.model(data.x.to(self.device), data.edge_index.to(self.device))
        y_all = data.y.to(self.device)
        labels = y_all[mask].cpu().numpy()
        probs = F.softmax(out[mask], dim=-1)[:, 1].cpu().numpy()
        resample_info = None

        if match_ratio_mask is not None:
            target_mask = getattr(data, match_ratio_mask)
            target_labels = data.y[target_mask].cpu().numpy()
            t_illicit = int((target_labels == 1).sum())
            t_licit = int((target_labels == 0).sum())
            if t_licit == 0 or t_illicit == 0:
                resample_info = {"applied": False, "reason": "target mask has only one class"}
            else:
                target_ratio = t_illicit / t_licit  # positives per negative
                v_illicit = int((labels == 1).sum())
                v_licit = int((labels == 0).sum())
                v_current_ratio = v_illicit / v_licit
                # To reach target_ratio we either:
                #   (A) downsample illicit:  keep all v_licit, reduce illicit to v_licit*target_ratio
                #   (B) downsample licit:    keep all v_illicit, reduce licit to v_illicit/target_ratio
                # Both are feasible only if the desired count is <= the available count.
                # Pick the feasible option with more TOTAL samples (statistical power).
                # If target_ratio > v_current_ratio → need more illicit per licit → option B
                # If target_ratio < v_current_ratio → need fewer illicit per licit → option A
                rng = _np.random.RandomState(seed)
                if target_ratio <= v_current_ratio:
                    # Downsample illicit (option A) — always feasible because
                    # we're reducing illicit count.
                    new_illicit = max(int(round(v_licit * target_ratio)), 1)
                    new_illicit = min(new_illicit, v_illicit)
                    idx_illicit = _np.where(labels == 1)[0]
                    idx_licit = _np.where(labels == 0)[0]
                    sel_illicit = rng.choice(idx_illicit, size=new_illicit, replace=False)
                    keep_idx = _np.concatenate([sel_illicit, idx_licit])
                else:
                    # Downsample licit (option B).
                    new_licit = max(int(round(v_illicit / target_ratio)), 1)
                    new_licit = min(new_licit, v_licit)
                    idx_illicit = _np.where(labels == 1)[0]
                    idx_licit = _np.where(labels == 0)[0]
                    sel_licit = rng.choice(idx_licit, size=new_licit, replace=False)
                    keep_idx = _np.concatenate([idx_illicit, sel_licit])
                labels = labels[keep_idx]
                probs = probs[keep_idx]
                resample_info = {
                    "applied": True,
                    "target_mask": match_ratio_mask,
                    "target_ratio_illicit_per_licit": target_ratio,
                    "before": {"illicit": v_illicit, "licit": v_licit},
                    "after": {"illicit": int((labels == 1).sum()),
                              "licit": int((labels == 0).sum())},
                }

        thresholds = np.linspace(1.0 / (n_thresholds + 1),
                                 n_thresholds / (n_thresholds + 1),
                                 n_thresholds)
        best_t = 0.5
        best_f1 = -1.0
        sweep = []
        for t in thresholds:
            preds = (probs >= t).astype(int)
            f1 = f1_score(labels, preds, pos_label=1, zero_division=0)
            sweep.append((float(t), float(f1)))
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)

        return {
            "threshold": best_t,
            "f1": float(best_f1),
            "sweep": sweep,
            "resample": resample_info,
        }

    def train(
        self,
        data,
        epochs: int = 300,
        verbose: bool = True,
        run_name: str = "model",
    ) -> dict:
        """
        Full training loop with early stopping.

        Args:
            data: PyG Data object with train/val masks.
            epochs: Maximum epochs.
            verbose: Print progress.
            run_name: Name for checkpoint file.

        Returns:
            Dict with best metrics and training history.
        """
        # ── Recovery checkpoint setup ──────────────────────────────────────────
        safe_run_name = run_name.replace(":", "-")
        epoch_ckpt_path = self.epoch_ckpt_base / safe_run_name / "checkpoint_last.pt"
        start_epoch = 1

        if not self.disable_checkpointing and epoch_ckpt_path.exists():
            try:
                ckpt = torch.load(epoch_ckpt_path, weights_only=False,
                                  map_location=self.device)
                self.model.load_state_dict(ckpt["model_state"])
                self.optimizer.load_state_dict(ckpt["optimizer_state"])
                start_epoch = ckpt["epoch"] + 1
                # Prefer generic best_val_score; fall back to legacy best_val_mcc
                self.best_val_score = ckpt.get(
                    "best_val_score", ckpt.get("best_val_mcc", self.best_val_score)
                )
                self.best_val_mcc = ckpt.get("best_val_mcc", self.best_val_mcc)
                self.patience_counter = ckpt.get("patience_counter", 0)
                torch.set_rng_state(ckpt["torch_rng_state"])
                np.random.set_state(ckpt["numpy_rng_state"])
                if torch.cuda.is_available() and ckpt.get("cuda_rng_state") is not None:
                    torch.cuda.set_rng_state(ckpt["cuda_rng_state"])
                print(f"  Resumed from epoch checkpoint: {epoch_ckpt_path}")
                print(f"  Continuing from epoch {start_epoch} "
                      f"(best val {self.early_stop_metric} so far: {self.best_val_score:.4f})")
            except Exception as exc:
                print(f"  WARNING: Could not load epoch checkpoint ({exc}), starting fresh")
                start_epoch = 1

        for epoch in range(start_epoch, epochs + 1):
            train_loss = self.train_epoch(data)
            val_metrics = self.evaluate(data, "val_mask")

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_f1"].append(val_metrics["f1"])
            self.history["val_mcc"].append(val_metrics["mcc"])
            self.history["val_pr_auc"].append(val_metrics["pr_auc"])

            # Log per-epoch metrics to MLflow (visible as live curves)
            if self.tracker:
                self.tracker.log_epoch(
                    epoch,
                    train_loss=train_loss,
                    val_loss=val_metrics["loss"],
                    val_f1=val_metrics["f1"],
                    val_mcc=val_metrics["mcc"],
                    val_pr_auc=val_metrics["pr_auc"],
                )

            # Early stopping on configured metric (default F1)
            current_score = val_metrics[self.early_stop_metric]
            if current_score > self.best_val_score:
                self.best_val_score = current_score
                self.best_val_mcc = val_metrics["mcc"]  # keep MCC snapshot for reports
                self.best_epoch = epoch
                self.patience_counter = 0
                # Save best model checkpoint
                if not self.disable_checkpointing:
                    safe_run_name = run_name.replace(":", "-")
                    ckpt_path = self.checkpoint_dir / f"{safe_run_name}_best.pt"
                    torch.save(self.model.state_dict(), ckpt_path)
            else:
                self.patience_counter += 1

            if verbose and epoch % 10 == 0:
                print(
                    f"  Epoch {epoch:3d} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f} | "
                    f"Val F1: {val_metrics['f1']:.4f} | "
                    f"Val MCC: {val_metrics['mcc']:.4f} | "
                    f"Val PR-AUC: {val_metrics['pr_auc']:.4f}"
                )

            # Save recovery checkpoint at epoch 1 and every 10 epochs
            # (epoch 1 ensures a checkpoint exists even with few-epoch quick runs)
            if not self.disable_checkpointing and (epoch == 1 or epoch % 10 == 0):
                epoch_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_score": self.best_val_score,
                    "best_val_mcc": self.best_val_mcc,
                    "early_stop_metric": self.early_stop_metric,
                    "patience_counter": self.patience_counter,
                    "torch_rng_state": torch.get_rng_state(),
                    "numpy_rng_state": np.random.get_state(),
                    "cuda_rng_state": (torch.cuda.get_rng_state()
                                       if torch.cuda.is_available() else None),
                }, epoch_ckpt_path)

            if self.patience_counter >= self.patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch} (best: {self.best_epoch})")
                break

        # Delete recovery checkpoint (training completed cleanly)
        if epoch_ckpt_path.exists():
            epoch_ckpt_path.unlink()
            try:
                epoch_ckpt_path.parent.rmdir()
            except OSError:
                pass

        # Load best model
        if not self.disable_checkpointing:
            ckpt_path = self.checkpoint_dir / f"{safe_run_name}_best.pt"
            if ckpt_path.exists():
                self.model.load_state_dict(torch.load(ckpt_path, weights_only=True))

        # Final test evaluation
        test_metrics = self.evaluate(data, "test_mask")

        return {
            "best_epoch": self.best_epoch,
            "best_val_mcc": self.best_val_mcc,
            "best_val_score": self.best_val_score,
            "early_stop_metric": self.early_stop_metric,
            "test_metrics": test_metrics,
            "history": self.history,
        }


def build_model(
    arch_name: str,
    in_channels: int,
    hidden_channels: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3,
    **kwargs,
) -> nn.Module:
    """
    Factory function to build a GNN model by name.

    Args:
        arch_name: One of "GCN", "GraphSAGE", "GAT", "TAGCN".
        in_channels: Number of input features.
        hidden_channels: Hidden dimension.
        num_layers: Number of GNN layers.
        dropout: Dropout rate.

    Returns:
        GNN model instance.
    """
    from src.models.gcn import GCN
    from src.models.sage import GraphSAGE
    from src.models.gat import GAT
    from src.models.tagcn import TAGCN

    models = {
        "GCN": GCN,
        "GraphSAGE": GraphSAGE,
        "GAT": GAT,
        "TAGCN": TAGCN,
    }

    if arch_name not in models:
        raise ValueError(f"Unknown architecture: {arch_name}. Choose from {list(models.keys())}")

    model_cls = models[arch_name]

    # Handle architecture-specific kwargs
    model_kwargs = {
        "in_channels": in_channels,
        "hidden_channels": hidden_channels,
        "num_layers": num_layers,
        "dropout": dropout,
    }

    if arch_name == "GAT" and "heads" in kwargs:
        model_kwargs["heads"] = kwargs["heads"]
    if arch_name == "TAGCN" and "K" in kwargs:
        model_kwargs["K"] = kwargs["K"]

    return model_cls(**model_kwargs)


def tune_hyperparams(
    arch_name: str,
    data,
    loss_fn,
    hp_config: dict,
    device: str,
    patience: int = 10,
) -> dict:
    """
    Run Optuna hyperparameter search for a given architecture and dataset.

    Uses a fraction of the full training epochs (tune_epochs) to speed up
    the search. The best hyperparameters are then used for the full training.

    Args:
        arch_name: Architecture name (e.g. "GCN").
        data: PyG Data object with train/val/test masks.
        loss_fn: Loss function.
        hp_config: Dict from config["models"]["hyperparameter_search"].
        device: Target device.
        patience: Early stopping patience for each trial.

    Returns:
        Dict with keys: hidden_channels, num_layers, dropout, lr.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    n_trials = hp_config.get("optuna_trials", 20)
    # Use ~1/3 of total epochs for each trial to keep search fast
    tune_epochs = max(30, hp_config.get("tune_epochs", 100))

    print(f"  Optuna search: {n_trials} trials × {tune_epochs} epochs "
          f"({arch_name}, device={device})")

    def objective(trial):
        hidden_channels = trial.suggest_categorical(
            "hidden_channels", hp_config["hidden_dim"])
        num_layers = trial.suggest_categorical(
            "num_layers", hp_config["num_layers"])
        dropout = trial.suggest_categorical(
            "dropout", hp_config["dropout"])
        lr = trial.suggest_categorical(
            "lr", hp_config["learning_rate"])

        model = build_model(arch_name, data.num_node_features,
                            hidden_channels, num_layers, dropout)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        trainer = Trainer(
            model, loss_fn, optimizer, device,
            patience=patience,
            disable_checkpointing=True,
        )
        results = trainer.train(
            data, epochs=tune_epochs, verbose=False,
            run_name=f"_optuna_trial_{trial.number}",
        )
        return results["best_val_mcc"]

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    print(f"  Best trial #{study.best_trial.number}: "
          f"hidden={best['hidden_channels']} layers={best['num_layers']} "
          f"dropout={best['dropout']} lr={best['lr']} "
          f"(val MCC={study.best_value:.4f})")

    return {
        "hidden_channels": best["hidden_channels"],
        "num_layers": best["num_layers"],
        "dropout": best["dropout"],
        "lr": best["lr"],
    }
