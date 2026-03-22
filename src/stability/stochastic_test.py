"""
Stochastic stability tests for XAI methods.

Re-executes explainers with different random seeds to measure
consistency of explanations.
"""

import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data
from typing import Optional

from src.explainability.explainer_runner import create_explainer, explain_nodes
from src.explainability.shap_runner import explain_node_shap
from src.explainability.extraction import extract_subgraph, extract_feature_ranking


def run_stochastic_replicas(
    model: nn.Module,
    data: Data,
    node_idx: int,
    method: str = "GNNExplainer",
    num_replicas: int = 30,
    top_k_edges: int = 20,
    device: str = "cpu",
    explainer_epochs: int = 200,
    explainer_lr: float = 0.01,
    shap_samples: int = 100,
) -> dict:
    """
    Run multiple explainer replicas with different seeds on the same node.

    Args:
        model: Trained GNN model.
        data: PyG Data object.
        node_idx: Node to explain repeatedly.
        method: Explainer method name.
        num_replicas: Number of replicas.
        top_k_edges: Top-k edges for subgraph extraction.
        device: Target device.
        explainer_epochs: Epochs for GNNExplainer/PGExplainer.
        explainer_lr: Learning rate for explainer.
        shap_samples: Number of SHAP samples per replica.

    Returns:
        Dict with:
          - subgraphs: List of edge sets (one per replica)
          - feature_rankings: List of ranking arrays
          - shap_values: List of SHAP value arrays (if applicable)
    """
    subgraphs = []
    feature_rankings = []
    shap_values_list = []

    for replica in range(num_replicas):
        seed = 42 + replica * 17  # Deterministic but varied seeds

        # Set global seeds for PyTorch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        if method in ("GNNExplainer", "PGExplainer"):
            explainer = create_explainer(
                model, method, epochs=explainer_epochs, lr=explainer_lr
            )
            explanations = explain_nodes(explainer, data, [node_idx], device)
            exp = explanations[0]

            subgraph = extract_subgraph(exp, top_k=top_k_edges)
            ranking = extract_feature_ranking(explanation=exp)

            subgraphs.append(subgraph)
            feature_rankings.append(ranking)

        elif method == "GNNShap":
            result = explain_node_shap(
                model, data, node_idx, num_samples=shap_samples,
                device=device, seed=seed
            )
            ranking = result["feature_ranking"]
            shap_vals = result["shap_values"]

            feature_rankings.append(ranking)
            shap_values_list.append(shap_vals)

    return {
        "node_idx": node_idx,
        "method": method,
        "num_replicas": num_replicas,
        "subgraphs": subgraphs,
        "feature_rankings": feature_rankings,
        "shap_values": shap_values_list if shap_values_list else None,
    }


def run_stochastic_test_batch(
    model: nn.Module,
    data: Data,
    node_indices: list,
    method: str = "GNNExplainer",
    num_replicas: int = 30,
    top_k_edges: int = 20,
    device: str = "cpu",
    **kwargs,
) -> list:
    """
    Run stochastic stability tests for multiple nodes.

    Args:
        model: Trained GNN model.
        data: PyG Data object.
        node_indices: List of node indices.
        method: Explainer method.
        num_replicas: Replicas per node.
        top_k_edges: Top-k edges for subgraph.
        device: Device.

    Returns:
        List of result dicts (one per node).
    """
    results = []
    for i, idx in enumerate(node_indices):
        if (i + 1) % 5 == 0:
            print(f"    Stochastic test: node {i+1}/{len(node_indices)}")
        result = run_stochastic_replicas(
            model, data, idx, method, num_replicas,
            top_k_edges, device, **kwargs
        )
        results.append(result)
    return results
