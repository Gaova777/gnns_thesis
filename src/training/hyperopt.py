"""
Hyperparameter optimization using Optuna.

Searches over hidden dimensions, layers, dropout, learning rate, and
weight decay for each GNN architecture. Default metric is PR-AUC (more
stable than MCC on imbalanced data); MCC and F1 are also supported.

Warm-start priors (literature-informed default hyperparameters) are enqueued
as trial 0 so Optuna always evaluates a sensible starting config before random
exploration.
"""

import gc
import optuna
from optuna.trial import Trial
import torch
import torch.nn as nn
from typing import Optional

from src.training.trainer import Trainer, build_model
from src.balancing.losses import get_loss_function


# Warm-start hyperparameters: literature-informed default starting points, enqueued
# as Optuna's first trial (study.enqueue_trial). NOTE: these are reasonable defaults,
# NOT taken from a specific published Optuna run — no specific F1 is attributed to them.
_WARM_START_PRIORS = {
    "GCN": {
        "hidden_dim": 211,
        "num_layers": 2,
        "dropout": 0.2361,
        "lr": 8.475e-4,
        "weight_decay": 5e-4,
    },
    "GraphSAGE": {
        "hidden_dim": 140,
        "num_layers": 2,
        "dropout": 0.1135,
        "lr": 5.302e-4,
        "weight_decay": 5e-4,
    },
    "GAT": {
        "hidden_dim": 148,
        "num_layers": 2,
        "dropout": 0.2522,
        "lr": 6.999e-4,
        "weight_decay": 5e-4,
        # heads=4 (el default de GAT), no 8: con hidden=148 y 8 cabezas el tensor de
        # mensajes por arista (234k aristas x 1184) pide ~9,5 GiB y no cabe en la RTX 4060
        # de 8 GB; heads=4 con hidden=148 pica en 4,5 GiB. Ver runs_v4/DECISIONES.md (29-sep).
        "heads": 4,
    },
    # TAGCN has no published Elliptic baseline — reuse GCN prior + typical K.
    "TAGCN": {
        "hidden_dim": 211,
        "num_layers": 2,
        "dropout": 0.3,
        "lr": 8.475e-4,
        "weight_decay": 5e-4,
        "K": 3,
    },
}


def get_warm_start_priors(arch_name: str) -> dict:
    """
    Return literature-informed default hyperparameters for the given architecture.

    These are reasonable starting points (not a specific published result),
    enqueued as Optuna's first trial (study.enqueue_trial) so search is warm-started.
    """
    if arch_name not in _WARM_START_PRIORS:
        raise ValueError(
            f"No warm-start prior for arch {arch_name!r}. "
            f"Available: {list(_WARM_START_PRIORS)}"
        )
    return dict(_WARM_START_PRIORS[arch_name])


def objective(
    trial: Trial,
    data,
    arch_name: str,
    balancing: str = "none",
    device: str = "cpu",
    epochs: int = 200,
    patience: int = 20,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.75,
    metric: str = "pr_auc",
    early_stop_metric: Optional[str] = None,
    hidden_dim_choices: Optional[list] = None,
    search_space: Optional[dict] = None,
    epoch_callback=None,
) -> float:
    """
    Optuna objective function for hyperparameter search.

    Args:
        trial: Optuna trial object.
        data: PyG Data object with masks.
        arch_name: GNN architecture name.
        balancing: Balancing technique.
        device: Target device.
        epochs: Max training epochs.
        patience: Early stopping patience.
        focal_gamma, focal_alpha: Focal Loss hyperparameters (alpha weights rare class).
        metric: Which metric to return ('pr_auc', 'f1', 'mcc').
        early_stop_metric: Metric used for early stopping inside a trial.
                           Defaults to `metric` (consistent with Optuna objective).
        search_space: Optional ranges from the config ``hyperparameter_search`` block
                      (num_layers, dropout, learning_rate, weight_decay as [lo, hi]).
                      Missing keys fall back to the historical defaults.
        epoch_callback: Passed to the Trainer (time budget / heartbeat).

    Returns:
        Validation score (to maximize) — either val_pr_auc, val_f1 or val_mcc.
    """
    # Early stopping metric must match the Optuna objective so best_val_score
    # tracks the requested metric; otherwise we'd optimize X but return best of Y.
    early_stop_metric = metric

    # Suggest hyperparameters — wider ranges incorporating literature priors
    # Default choices include all literature priors: 140 (GraphSAGE), 148 (GAT), 211 (GCN/TAGCN).
    # Override via hidden_dim_choices when the config wants a VRAM/speed-constrained cap.
    _choices = hidden_dim_choices if hidden_dim_choices else [64, 128, 140, 148, 211, 256]
    hidden_dim = trial.suggest_categorical("hidden_dim", _choices)
    sp = search_space or {}
    nl = sp.get("num_layers") or [2, 3]
    do = sp.get("dropout") or [0.1, 0.5]
    lr_r = sp.get("learning_rate") or [1e-4, 1e-2]
    wd_r = sp.get("weight_decay") or [1e-5, 1e-3]
    # GAT se capa a 2 capas: con 3 capas + heads=4 + hidden=148 sobre el grafo completo
    # el pico pasa de ~7 GiB y hace OOM en la RTX 4060. 2 capas = campo receptivo de 2
    # saltos (el default estándar en Elliptic) y pica en ~4,5 GiB. Ver runs_v4/DECISIONES.md.
    nl_max = 2 if arch_name == "GAT" else int(max(nl))
    num_layers = trial.suggest_int("num_layers", int(min(nl)), nl_max)
    dropout = trial.suggest_float("dropout", float(do[0]), float(do[1]))
    lr = trial.suggest_float("lr", float(lr_r[0]), float(lr_r[1]), log=True)
    weight_decay = trial.suggest_float("weight_decay", float(wd_r[0]), float(wd_r[1]), log=True)

    # Architecture-specific params
    kwargs = {}
    if arch_name == "GAT":
        # {2,4} en vez de {4,8}: 8 cabezas con hidden=148 sobre el grafo completo no cabe
        # en 8 GB de VRAM (OOM). El sobre {2,4} x {64,128,148} entra siempre. DECISIONES.md.
        kwargs["heads"] = trial.suggest_categorical("heads", [2, 4])
    if arch_name == "TAGCN":
        kwargs["K"] = trial.suggest_int("K", 2, 4)

    # Build model
    model = build_model(
        arch_name,
        in_channels=data.num_node_features,
        hidden_channels=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        **kwargs,
    )

    # Loss function (alpha semantics: weight for rare class — use 0.75+ for imbalance)
    loss_fn = get_loss_function(
        balancing,
        labels=data.y,
        mask=data.train_mask,
        gamma=focal_gamma,
        alpha=focal_alpha,
        device=device,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Train (no checkpointing inside trials — saves disk + IO)
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        patience=patience,
        disable_checkpointing=True,
        early_stop_metric=early_stop_metric,
        epoch_callback=epoch_callback,
    )

    try:
        results = trainer.train(
            data, epochs=epochs, verbose=False, run_name=f"trial_{trial.number}"
        )
        # best_val_score is captured at the best epoch (not last), using the same
        # metric Optuna is maximizing. That's what we return.
        return results["best_val_score"]
    finally:
        # Liberar la VRAM del trial: sin esto el proceso acumula memoria trial a trial
        # y el reentrenamiento final de GAT (grafo completo) hace OOM. No cambia el
        # cómputo, solo devuelve la memoria al asignador. Ver runs_v4/DECISIONES.md.
        try:
            model.to("cpu")
        except Exception:
            pass
        del model, optimizer, trainer
        gc.collect()
        if str(device) != "cpu":
            torch.cuda.empty_cache()


def run_hyperopt(
    data,
    arch_name: str,
    balancing: str = "none",
    n_trials: int = 50,
    device: str = "cpu",
    epochs: int = 200,
    patience: int = 20,
    study_name: Optional[str] = None,
    metric: str = "pr_auc",
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.75,
    warm_start: bool = True,
    hidden_dim_choices: Optional[list] = None,
    search_space: Optional[dict] = None,
    epoch_callback=None,
    timeout_s: Optional[float] = None,
    show_progress_bar: bool = True,
) -> dict:
    """
    Run hyperparameter optimization for a given architecture + balancing combo.

    Args:
        data: PyG Data object.
        arch_name: GNN architecture name.
        balancing: Balancing technique.
        n_trials: Number of Optuna trials (total, including warm-start).
        device: Target device.
        epochs: Max epochs per trial.
        patience: Early stopping patience.
        study_name: Optional study name for Optuna.
        metric: Optuna objective metric ('pr_auc', 'f1', 'mcc').
        focal_gamma, focal_alpha: Focal Loss hyperparameters.
        warm_start: If True, enqueue literature-optimal priors as trial 0.
        search_space: Config ranges (see ``objective``).
        epoch_callback: Called after every epoch of every trial; an exception it raises
            (e.g. ConfigTimeoutError) aborts the whole study and propagates.
        timeout_s: Overall Optuna budget in seconds (default 8 min × n_trials).

    Returns:
        Dict with best_params, best_score, metric, and the Optuna study.
        Also returns 'best_mcc' as legacy alias (equals best_score iff metric='mcc',
        otherwise the test-set MCC achieved by the winning trial's hyperparameters).
    """
    if study_name is None:
        study_name = f"{arch_name}_{balancing}_{metric}"

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        # Aggressive pruning for slow archs (GAT/TAGCN): kick in after 2 warmup
        # trials and prune after 15 steps so bad trials die quickly.
        pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=15),
    )

    # Warm start: enqueue a trial with literature-optimal hyperparameters
    if warm_start:
        try:
            prior = get_warm_start_priors(arch_name)
            # If config restricts hidden_dim, snap prior to nearest available
            if hidden_dim_choices and prior["hidden_dim"] not in hidden_dim_choices:
                original = prior["hidden_dim"]
                prior["hidden_dim"] = min(hidden_dim_choices,
                                          key=lambda x: abs(x - original))
                print(f"  Warm-start hidden_dim snapped {original}→{prior['hidden_dim']} "
                      f"(config limited to {hidden_dim_choices})")
            study.enqueue_trial(prior)
            print(f"  Warm-start trial enqueued for {arch_name}: {prior}")
        except ValueError as exc:
            print(f"  WARNING: {exc}. Proceeding without warm start.")

    # Per-trial timeout protects against runaway training (observed 25 min on
    # GAT with unlucky HPs). A good HP should complete hyp_epochs=60 easily.
    # Default: 8 minutes per trial.
    study.optimize(
        lambda trial: objective(
            trial, data, arch_name, balancing, device, epochs, patience,
            focal_gamma=focal_gamma, focal_alpha=focal_alpha, metric=metric,
            hidden_dim_choices=hidden_dim_choices,
            search_space=search_space, epoch_callback=epoch_callback,
        ),
        n_trials=n_trials,
        timeout=timeout_s if timeout_s is not None else n_trials * 480,
        show_progress_bar=show_progress_bar,
    )

    print(f"\n  Best trial (val {metric}={study.best_trial.value:.4f}):")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}")

    return {
        "best_params": study.best_trial.params,
        "best_score": study.best_trial.value,
        # Legacy alias: older callers read 'best_mcc'. Keeps them working even
        # when the actual metric is pr_auc/f1 — value is the best val score.
        "best_mcc": study.best_trial.value,
        "metric": metric,
        "study": study,
    }
