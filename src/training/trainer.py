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
    based on validation loss or MCC.

    Args:
        model: GNN model (nn.Module).
        loss_fn: Loss function.
        optimizer: torch optimizer.
        device: Target device.
        patience: Early stopping patience.
        checkpoint_dir: Directory to save best model.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "cpu",
        patience: int = 20,
        checkpoint_dir: str = "./results/models",
        tracker=None,
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.patience = patience
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.tracker = tracker  # ExperimentTracker for per-epoch logging

        # Tracking
        self.best_val_mcc = -2.0
        self.best_epoch = 0
        self.patience_counter = 0
        self.history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_mcc": []}

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
    def evaluate(self, data, mask_name: str = "val_mask") -> dict:
        """
        Evaluate model on a given mask.

        Returns dict with loss, f1, mcc, pr_auc.
        """
        self.model.eval()
        mask = getattr(data, mask_name).to(self.device)
        out = self.model(data.x.to(self.device), data.edge_index.to(self.device))

        # True labels on device
        y_true = data.y.to(self.device)[mask]

        # Loss
        loss = self.loss_fn(out[mask], y_true).item()

        # Predictions
        preds = out[mask].argmax(dim=-1).cpu().numpy()
        labels = y_true.cpu().numpy()
        probs = F.softmax(out[mask], dim=-1)[:, 1].cpu().numpy()

        # Metrics
        f1 = f1_score(labels, preds, pos_label=1, zero_division=0)
        mcc = matthews_corrcoef(labels, preds)

        # PR-AUC
        precision, recall, _ = precision_recall_curve(labels, probs, pos_label=1)
        pr_auc = auc(recall, precision)

        return {"loss": loss, "f1": f1, "mcc": mcc, "pr_auc": pr_auc}

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
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(data)
            val_metrics = self.evaluate(data, "val_mask")

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_f1"].append(val_metrics["f1"])
            self.history["val_mcc"].append(val_metrics["mcc"])

            # Log per-epoch metrics to MLflow (visible as live curves)
            if self.tracker:
                self.tracker.log_epoch(
                    epoch,
                    train_loss=train_loss,
                    val_loss=val_metrics["loss"],
                    val_f1=val_metrics["f1"],
                    val_mcc=val_metrics["mcc"],
                )

            # Early stopping on MCC
            if val_metrics["mcc"] > self.best_val_mcc:
                self.best_val_mcc = val_metrics["mcc"]
                self.best_epoch = epoch
                self.patience_counter = 0
                # Save checkpoint
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
                    f"Val MCC: {val_metrics['mcc']:.4f}"
                )

            if self.patience_counter >= self.patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch} (best: {self.best_epoch})")
                break

        # Load best model
        safe_run_name = run_name.replace(":", "-")
        ckpt_path = self.checkpoint_dir / f"{safe_run_name}_best.pt"
        if ckpt_path.exists():
            self.model.load_state_dict(torch.load(ckpt_path, weights_only=True))

        # Final test evaluation
        test_metrics = self.evaluate(data, "test_mask")

        return {
            "best_epoch": self.best_epoch,
            "best_val_mcc": self.best_val_mcc,
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
