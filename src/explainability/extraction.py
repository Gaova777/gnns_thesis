"""
Extraction utilities for explanation outputs.

Converts raw PyG Explanation objects and SHAP values into
standardized formats for stability comparison.
"""

import torch
import numpy as np
import json
from pathlib import Path
from torch_geometric.explain import Explanation
from typing import Optional


def extract_subgraph(
    explanation: Explanation,
    top_k: int = 20,
) -> set:
    """
    Extract the top-k most important edges from an explanation.

    Args:
        explanation: PyG Explanation object with edge_mask.
        top_k: Number of top edges to include in the subgraph.

    Returns:
        Set of (src, dst) edge tuples.
    """
    if explanation.edge_mask is None:
        return set()

    edge_mask = explanation.edge_mask.cpu().detach()
    edge_index = explanation.edge_index.cpu()

    # Get top-k edge indices by importance
    k = min(top_k, len(edge_mask))
    _, top_indices = torch.topk(edge_mask, k)

    subgraph = set()
    for idx in top_indices:
        src = edge_index[0, idx].item()
        dst = edge_index[1, idx].item()
        subgraph.add((src, dst))

    return subgraph


def extract_feature_ranking(
    explanation: Explanation = None,
    shap_values: np.ndarray = None,
) -> np.ndarray:
    """
    Extract feature importance ranking from an explanation.

    Uses either PyG node_mask or SHAP values.

    Args:
        explanation: PyG Explanation with node_mask (optional).
        shap_values: SHAP values array (optional).

    Returns:
        Array of feature indices sorted by importance (descending).
    """
    if shap_values is not None:
        return np.argsort(np.abs(shap_values))[::-1].copy()

    # Use getattr to avoid AttributeError on PyG Explanation objects that
    # don't have node_mask (e.g. PGExplainer with explanation_type="phenomenon")
    node_mask_val = getattr(explanation, "node_mask", None) if explanation is not None else None
    if node_mask_val is not None:
        node_mask = node_mask_val.cpu().detach().numpy()
        # Average over nodes if multi-node mask
        if node_mask.ndim > 1:
            importance = np.abs(node_mask).mean(axis=0)
        else:
            importance = np.abs(node_mask)
        return np.argsort(importance)[::-1].copy()

    return np.array([])


def serialize_explanation(
    node_idx: int,
    subgraph: set,
    feature_ranking: np.ndarray,
    shap_values: np.ndarray = None,
    metadata: dict = None,
) -> dict:
    """
    Serialize an explanation to a JSON-compatible dict.

    Args:
        node_idx: Explained node index.
        subgraph: Set of (src, dst) edge tuples.
        feature_ranking: Feature importance ranking array.
        shap_values: Optional SHAP values.
        metadata: Optional metadata dict.

    Returns:
        Serializable dict.
    """
    result = {
        "node_idx": int(node_idx),
        "subgraph": [list(edge) for edge in subgraph],
        "feature_ranking": feature_ranking.tolist(),
    }
    if shap_values is not None:
        result["shap_values"] = shap_values.tolist()
    if metadata:
        result["metadata"] = metadata
    return result


def save_explanations(
    explanations: list,
    filepath: str,
) -> None:
    """Save serialized explanations to JSON."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(explanations, f, indent=2)
    print(f"  Saved {len(explanations)} explanations to {filepath}")


def load_explanations(filepath: str) -> list:
    """Load explanations from JSON."""
    with open(filepath, "r") as f:
        return json.load(f)
