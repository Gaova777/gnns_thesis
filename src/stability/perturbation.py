"""
Perturbation-based stability tests.

Injects controlled Gaussian noise into node features and measures
how explanations change in response to minimal input perturbations.
"""

import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data
from copy import deepcopy
from typing import Optional

from src.explainability.explainer_runner import create_explainer, explain_nodes
from src.explainability.shap_runner import explain_node_shap
from src.explainability.extraction import extract_subgraph, extract_feature_ranking


def perturb_node_features(
    data: Data,
    node_idx: int,
    noise_level: float = 0.01,
    seed: int = 42,
) -> Data:
    """
    Add Gaussian noise to a specific node's features.

    Args:
        data: PyG Data object.
        node_idx: Index of node to perturb.
        noise_level: Standard deviation of Gaussian noise (σ).
        seed: Random seed.

    Returns:
        New Data object with perturbed features.
    """
    rng = np.random.RandomState(seed)
    data_perturbed = deepcopy(data)

    noise = torch.tensor(
        rng.normal(0, noise_level, size=data.x[node_idx].shape),
        dtype=torch.float32,
    )
    data_perturbed.x[node_idx] += noise

    return data_perturbed


def run_perturbation_test(
    model: nn.Module,
    data: Data,
    node_idx: int,
    method: str = "GNNExplainer",
    noise_levels: list = None,
    num_perturbations: int = 10,
    top_k_edges: int = 20,
    device: str = "cpu",
    explainer_epochs: int = 200,
    explainer_lr: float = 0.01,
    shap_samples: int = 100,
) -> dict:
    """
    Test explanation stability under feature perturbations.

    For each noise level, generates multiple perturbed versions of the
    input and compares the resulting explanations against the original.

    Args:
        model: Trained GNN model.
        data: Original PyG Data object.
        node_idx: Node to explain.
        method: Explainer method.
        noise_levels: List of noise σ values.
        num_perturbations: Perturbations per noise level.
        top_k_edges: Top-k edges for subgraph.
        device: Device.

    Returns:
        Dict with original explanation and perturbed explanations
        organized by noise level.
    """
    if noise_levels is None:
        noise_levels = [0.01, 0.05, 0.10]

    # Get original (unperturbed) explanation
    torch.manual_seed(42)
    if method in ("GNNExplainer", "PGExplainer"):
        explainer = create_explainer(model, method, explainer_epochs, explainer_lr)
        orig_explanations = explain_nodes(explainer, data, [node_idx], device)
        orig_subgraph = extract_subgraph(orig_explanations[0], top_k_edges)
        orig_ranking = extract_feature_ranking(explanation=orig_explanations[0])
    else:
        result = explain_node_shap(model, data, node_idx, shap_samples, device, seed=42)
        orig_subgraph = set()
        orig_ranking = result["feature_ranking"]

    # Perturbed explanations by noise level
    perturbed_results = {}
    for sigma in noise_levels:
        level_subgraphs = []
        level_rankings = []

        for p in range(num_perturbations):
            seed = 42 + p * 13
            data_p = perturb_node_features(data, node_idx, sigma, seed)

            torch.manual_seed(seed)
            if method in ("GNNExplainer", "PGExplainer"):
                explainer = create_explainer(model, method, explainer_epochs, explainer_lr)
                exps = explain_nodes(explainer, data_p, [node_idx], device)
                subgraph = extract_subgraph(exps[0], top_k_edges)
                ranking = extract_feature_ranking(explanation=exps[0])
            else:
                r = explain_node_shap(model, data_p, node_idx, shap_samples, device, seed=seed)
                subgraph = set()
                ranking = r["feature_ranking"]

            level_subgraphs.append(subgraph)
            level_rankings.append(ranking)

        perturbed_results[sigma] = {
            "subgraphs": level_subgraphs,
            "feature_rankings": level_rankings,
        }

    return {
        "node_idx": node_idx,
        "method": method,
        "original_subgraph": orig_subgraph,
        "original_ranking": orig_ranking,
        "perturbed": perturbed_results,
    }
