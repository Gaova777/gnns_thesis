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
        algorithm = PGExplainer(epochs=epochs, lr=min(lr, 0.003))
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
) -> None:
    """
    Train PGExplainer's parametric model on training nodes.

    Must be called before generating explanations with PGExplainer.

    Args:
        explainer: Explainer with PGExplainer algorithm.
        data: PyG Data object.
        device: Target device.
    """
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)

    # Get training node indices
    train_indices = torch.where(data.train_mask)[0]

    # Move PGExplainer's internal MLP to the same device as the model
    explainer.algorithm.to(device)

    # Get model predictions once — PGExplainer needs them as training target
    with torch.no_grad():
        out = explainer.model(x, edge_index)

    # PGExplainer needs to be trained on examples first
    loss = float("nan")
    nan_epochs = 0
    for epoch in range(explainer.algorithm.epochs):
        loss = 0.0
        for idx in train_indices[:100]:  # Subsample for efficiency
            step_loss = explainer.algorithm.train(
                epoch, explainer.model, x, edge_index,
                target=out,
                index=idx.item(),
            )
            if torch.is_tensor(step_loss):
                step_loss = step_loss.item()
            if not (step_loss == step_loss):  # NaN check
                nan_epochs += 1
                break
            loss += step_loss
        # Clip gradients to prevent divergence
        if hasattr(explainer.algorithm, "_mlp"):
            torch.nn.utils.clip_grad_norm_(
                explainer.algorithm._mlp.parameters(), max_norm=1.0
            )
    if nan_epochs > 0:
        print(f"  WARNING: PGExplainer loss was NaN in {nan_epochs} epochs "
              f"— explanations may be unreliable (final loss: {loss})")
    else:
        print(f"  PGExplainer training complete (final loss: {loss:.4f})")


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
