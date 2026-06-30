"""
Explainer runner using PyG's native torch_geometric.explain API.

Wraps GNNExplainer and PGExplainer for standardized execution
across different models and configurations.
"""

import torch
import torch.nn as nn
from torch_geometric.explain import Explainer, GNNExplainer, PGExplainer
from torch_geometric.explain import Explanation
from torch_geometric.data import Data
from typing import Optional


def create_explainer(
    model: nn.Module,
    method: str = "GNNExplainer",
    epochs: int = 200,
    lr: float = 0.01,
) -> Explainer:
    """
    Create a PyG Explainer with the specified algorithm.

    Args:
        model: Trained GNN model.
        method: "GNNExplainer" or "PGExplainer".
        epochs: Training epochs for the explainer.
        lr: Learning rate for the explainer.

    Returns:
        Configured Explainer instance.
    """
    if method == "GNNExplainer":
        algorithm = GNNExplainer(epochs=epochs, lr=lr)
        explainer = Explainer(
            model=model,
            algorithm=algorithm,
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type="object",
            model_config=dict(
                mode="multiclass_classification",
                task_level="node",
                return_type="raw",
            ),
        )
    elif method == "PGExplainer":
        # CRITICAL FIXES for PyG 2.7 PGExplainer defaults (methodological findings):
        # 1. edge_size=0.05 (default) causes mode collapse (mask=0 everywhere).
        #    Verified on Cora (balanced) and Elliptic. Fix: edge_size=0.005.
        # 2. temp=[5.0, 2.0] (default) causes NaN explosions on large graphs with
        #    extreme logits (Elliptic 234k edges + class_weighting magnitude).
        #    Fix: temp=[1.0, 1.0] for numerical stability.
        # See scripts/debug_pgexplainer_hyperparams.py for empirical evidence.
        algorithm = PGExplainer(
            epochs=epochs,
            lr=min(lr, 0.003),
            edge_size=0.005,    # default 0.05 → mode collapse
            edge_ent=1.0,       # keep default entropy regularizer
            temp=[1.0, 1.0],    # default [5.0, 2.0] → NaN overflow on large graphs
        )
        explainer = Explainer(
            model=model,
            algorithm=algorithm,
            explanation_type="phenomenon",
            edge_mask_type="object",
            model_config=dict(
                mode="multiclass_classification",
                task_level="node",
                return_type="raw",
            ),
        )
    else:
        raise ValueError(f"Unknown explainer method: {method}")

    return explainer


def explain_nodes(
    explainer: Explainer,
    data: Data,
    node_indices: list,
    device: str = "cpu",
) -> list:
    """
    Generate explanations for a list of nodes.

    Args:
        explainer: Configured PyG Explainer.
        data: PyG Data object.
        node_indices: List of node indices to explain.
        device: Target device.

    Returns:
        List of Explanation objects.
    """
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    explanations = []

    # PGExplainer with 'phenomenon' explanation_type needs a target
    # PyG stores explanation_type as an enum, so compare via str()
    needs_target = "phenomenon" in str(explainer.explanation_type).lower()

    for idx in node_indices:
        kwargs = {"index": idx}
        if needs_target:
            kwargs["target"] = data.y[idx].to(device)
        explanation = explainer(x, edge_index, **kwargs)
        explanations.append(explanation)

    return explanations


def train_pgexplainer(
    explainer: Explainer,
    data: Data,
    device: str = "cpu",
) -> bool:
    """
    Train PGExplainer's parametric model on training nodes.

    Uses a rollback strategy: saves weights before each step and restores
    them if the step produces NaN loss, preventing weight corruption cascades.

    Returns True if training produced usable weights, False otherwise.
    """
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    train_indices = torch.where(data.train_mask)[0]
    explainer.algorithm.to(device)

    # CRITICAL FIX — gradient clipping via monkey-patching the optimizer.
    # PyG PGExplainer.train() has no hook for gradient clipping between .backward()
    # and .step(). On large graphs (Elliptic 234k edges) with class_weighted logits,
    # gradients explode → NaN loss. Wrap optimizer.step() to clip norm first.
    _pg_optimizer = explainer.algorithm.optimizer
    _orig_step = _pg_optimizer.step

    def _step_with_clip(*args, **kwargs):
        torch.nn.utils.clip_grad_norm_(
            explainer.algorithm.parameters(), max_norm=1.0
        )
        return _orig_step(*args, **kwargs)

    _pg_optimizer.step = _step_with_clip

    # PGExplainer loss = cross_entropy(y_hat, y).
    # target must be class indices (long scalars), NOT raw logits.
    target = data.y.to(device)

    loss = 0.0
    nan_epochs = 0
    valid_steps = 0

    for epoch in range(explainer.algorithm.epochs):
        # Save state once per epoch — rollback entire epoch on NaN (50 nodes × 100 epochs
        # = only 100 snapshots max instead of 10,000 per-step snapshots).
        epoch_state = None
        try:
            epoch_state = {k: v.clone() for k, v in explainer.algorithm.state_dict().items()}
        except Exception:
            pass  # LazyModule not yet initialized on first epoch

        epoch_loss = 0.0
        epoch_valid = 0
        had_nan = False

        for idx in train_indices[:50]:  # 100→50 nodes: faster, still representative
            step_loss = explainer.algorithm.train(
                epoch, explainer.model, x, edge_index,
                target=target,
                index=idx.item(),
            )
            if torch.is_tensor(step_loss):
                step_loss = step_loss.item()

            if not (step_loss == step_loss):  # NaN
                had_nan = True
                break  # abort this epoch

            epoch_loss += step_loss
            epoch_valid += 1

        if had_nan and epoch_state is not None:
            explainer.algorithm.load_state_dict(epoch_state)  # rollback entire epoch
            nan_epochs += 1
        elif epoch_valid > 0:
            loss += epoch_loss / epoch_valid
            valid_steps += 1

    # Final health check
    has_nan_weights = any(
        torch.isnan(p).any()
        for p in explainer.algorithm.parameters()
        if p is not None
    )

    if has_nan_weights:
        print(f"  PGExplainer: weights contain NaN after training — unusable")
        return False

    total_epochs = nan_epochs + valid_steps
    nan_pct = 100 * nan_epochs / max(total_epochs, 1)
    avg_loss = loss / max(valid_steps, 1)

    if nan_epochs > 0:
        print(f"  PGExplainer: {nan_pct:.0f}% epochs rolled back due to NaN, "
              f"loss={avg_loss:.4f} over {valid_steps} clean epochs — proceeding")
    else:
        print(f"  PGExplainer training complete (loss={avg_loss:.4f})")
    return True


def select_explanation_nodes(
    data: Data,
    n_per_class: int = 100,
    mask_name: str = "test_mask",
    seed: int = 42,
) -> dict:
    """
    Select nodes to explain from the test set.

    Selects n_per_class illicit and n_per_class licit nodes.

    Args:
        data: PyG Data object.
        n_per_class: Number of nodes per class.
        mask_name: Which mask to select from.
        seed: Random seed.

    Returns:
        Dict with "illicit" and "licit" node index lists.
    """
    import numpy as np
    rng = np.random.RandomState(seed)
    mask = getattr(data, mask_name)

    illicit = torch.where(mask & (data.y == 1))[0].numpy()
    licit = torch.where(mask & (data.y == 0))[0].numpy()

    n_illicit = min(n_per_class, len(illicit))
    n_licit = min(n_per_class, len(licit))

    selected_illicit = rng.choice(illicit, size=n_illicit, replace=False).tolist()
    selected_licit = rng.choice(licit, size=n_licit, replace=False).tolist()

    print(f"  Selected {n_illicit} illicit and {n_licit} licit nodes for explanation")

    return {
        "illicit": selected_illicit,
        "licit": selected_licit,
        "all": selected_illicit + selected_licit,
    }
