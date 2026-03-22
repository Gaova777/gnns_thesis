"""
Hyperparameter optimization using Optuna.

Searches over hidden dimensions, layers, dropout, and learning rate
for each GNN architecture, optimizing MCC on the validation set.
"""

import optuna
from optuna.trial import Trial
import torch
import torch.nn as nn
from typing import Optional

from src.training.trainer import Trainer, build_model
from src.balancing.losses import get_loss_function


def objective(
    trial: Trial,
    data,
    arch_name: str,
    balancing: str = "none",
    device: str = "cpu",
    epochs: int = 200,
    patience: int = 20,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.25,
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

    Returns:
        Validation MCC (to maximize).
    """
    # Suggest hyperparameters
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    num_layers = trial.suggest_int("num_layers", 2, 3)
    dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    # Architecture-specific params
    kwargs = {}
    if arch_name == "GAT":
        kwargs["heads"] = trial.suggest_categorical("heads", [4, 8])
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

    # Loss function
    loss_fn = get_loss_function(
        balancing,
        labels=data.y,
        mask=data.train_mask,
        gamma=focal_gamma,
        alpha=focal_alpha,
        device=device,
    )

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        patience=patience,
    )

    results = trainer.train(
        data, epochs=epochs, verbose=False, run_name=f"trial_{trial.number}"
    )

    return results["best_val_mcc"]


def run_hyperopt(
    data,
    arch_name: str,
    balancing: str = "none",
    n_trials: int = 50,
    device: str = "cpu",
    epochs: int = 200,
    patience: int = 20,
    study_name: Optional[str] = None,
) -> dict:
    """
    Run hyperparameter optimization for a given architecture + balancing combo.

    Args:
        data: PyG Data object.
        arch_name: GNN architecture name.
        balancing: Balancing technique.
        n_trials: Number of Optuna trials.
        device: Target device.
        epochs: Max epochs per trial.
        patience: Early stopping patience.
        study_name: Optional study name for Optuna.

    Returns:
        Dict with best_params and best_mcc.
    """
    if study_name is None:
        study_name = f"{arch_name}_{balancing}"

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    study.optimize(
        lambda trial: objective(
            trial, data, arch_name, balancing, device, epochs, patience
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    print(f"\n  Best trial (MCC={study.best_trial.value:.4f}):")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}")

    return {
        "best_params": study.best_trial.params,
        "best_mcc": study.best_trial.value,
        "study": study,
    }
