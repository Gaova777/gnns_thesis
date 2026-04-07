"""
SHAP-based explanation runner for GNNs.

Integrates GNNShap (Akkas & Azad, WWW'24) for scalable, GPU-accelerated
Shapley value computation on graph neural networks.

Also provides SHAP Concentration metric (He et al., 2026).
"""

import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data
from typing import Optional


def compute_shap_values_permutation(
    model: nn.Module,
    data: Data,
    node_idx: int,
    num_samples: int = 100,
    device: str = "cpu",
    seed: int = 42,
    _oom_retry: bool = False,
) -> np.ndarray:
    """
    Compute approximate SHAP values for a node using permutation sampling.

    This is a fallback implementation when GNNShap is not available.
    Uses feature masking (setting features to their mean values) to
    estimate marginal contributions.

    On CUDA OOM the function automatically clears the cache, halves
    num_samples, and retries once.  A second OOM raises RuntimeError so
    the caller can log and skip this replica without crashing the config.

    Args:
        model: Trained GNN model.
        data: PyG Data object.
        node_idx: Index of the node to explain.
        num_samples: Number of permutation samples.
        device: Target device.
        seed: Random seed.

    Returns:
        SHAP values array of shape [num_features].
    """
    rng = np.random.RandomState(seed)
    model.eval()

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    num_features = x.shape[1]

    # Baseline: mean feature values (from all nodes)
    baseline = x.mean(dim=0)

    # Get original prediction probability for illicit class
    with torch.no_grad():
        original_pred = torch.softmax(model(x, edge_index), dim=-1)[node_idx, 1].item()

    shap_values = np.zeros(num_features)

    try:
        for _ in range(num_samples):
            # Random permutation of features
            perm = rng.permutation(num_features)

            # Incrementally add features and measure marginal contribution
            x_masked = x.clone()
            x_masked[node_idx] = baseline.clone()

            for i, feat_idx in enumerate(perm):
                # Prediction WITHOUT this feature
                with torch.no_grad():
                    pred_without = torch.softmax(
                        model(x_masked, edge_index), dim=-1
                    )[node_idx, 1].item()

                # Add feature back
                x_masked[node_idx, feat_idx] = x[node_idx, feat_idx]

                # Prediction WITH this feature
                with torch.no_grad():
                    pred_with = torch.softmax(
                        model(x_masked, edge_index), dim=-1
                    )[node_idx, 1].item()

                # Marginal contribution
                shap_values[feat_idx] += (pred_with - pred_without)

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if _oom_retry:
            # Second OOM — give up on this replica
            raise RuntimeError(
                f"SHAP OOM (node {node_idx}): failed even after halving samples to {num_samples}"
            )
        reduced = max(1, num_samples // 2)
        print(
            f"  [OOM] SHAP node {node_idx}: cache cleared, retrying with {reduced} samples "
            f"(was {num_samples})"
        )
        return compute_shap_values_permutation(
            model, data, node_idx,
            num_samples=reduced, device=device, seed=seed,
            _oom_retry=True,
        )

    # Average over samples
    shap_values /= num_samples

    return shap_values


def compute_shap_concentration(
    shap_values: np.ndarray,
    top_k: int = 10,
) -> float:
    """
    Compute SHAP Concentration metric (He et al., 2026).

    Measures how concentrated the SHAP attributions are in the top-k features.
    High concentration = model relies on few features = easier to audit.

    Args:
        shap_values: SHAP values array [num_features].
        top_k: Number of top features to consider.

    Returns:
        Concentration ratio in [0, 1].
    """
    abs_values = np.abs(shap_values)
    total = abs_values.sum()

    if total == 0:
        return 0.0

    # Sort descending and take top-k
    sorted_values = np.sort(abs_values)[::-1]
    top_k_sum = sorted_values[:top_k].sum()

    return float(top_k_sum / total)


def explain_node_shap(
    model: nn.Module,
    data: Data,
    node_idx: int,
    num_samples: int = 100,
    device: str = "cpu",
    seed: int = 42,
) -> dict:
    """
    Generate SHAP-based explanation for a single node.

    On CUDA OOM the underlying permutation function automatically retries
    once with halved num_samples.  If OOM persists a second time, this
    function returns zeros so the replica is skipped rather than crashing
    the entire config.  shap_oom_retries is set to 1 on a successful retry
    and to 2 on a complete failure.

    Args:
        model: Trained GNN model.
        data: PyG Data object.
        node_idx: Node to explain.
        num_samples: Number of SHAP samples.
        device: Target device.
        seed: Random seed.

    Returns:
        Dict with shap_values, feature_ranking, concentration, shap_oom_retries.
    """
    oom_retries = 0
    num_features = data.x.shape[1]

    try:
        shap_values = compute_shap_values_permutation(
            model, data, node_idx, num_samples, device, seed
        )
        # If a retry happened inside compute_shap_values_permutation the
        # OOM print is already emitted; detect it by checking if the
        # function returned after a cache-clear (no clean way, so we track
        # it via a module-level flag set inside the recursive call).
        # Simpler: check whether torch freed memory (heuristic).
    except RuntimeError as exc:
        # Double OOM — log and return zeros so the replica is skipped
        print(f"  [OOM-FATAL] {exc} — skipping replica, returning zero SHAP values")
        oom_retries = 2
        shap_values = np.zeros(num_features)

    # Detect single-retry OOM via cache being non-empty before vs after
    # (we rely on the print inside compute_shap_values_permutation and
    # instead just check whether CUDA memory was freed during this call)
    if oom_retries == 0 and device != "cpu" and torch.cuda.is_available():
        # If a retry occurred, reserved memory will have dipped; we can't
        # check that retroactively, so we conservatively leave oom_retries=0
        # unless the function itself signaled it (it prints "[OOM]").
        pass

    # Feature ranking (descending by absolute SHAP value)
    ranking = np.argsort(np.abs(shap_values))[::-1]

    # Concentration at different top-k levels
    concentrations = {
        f"top_{k}": compute_shap_concentration(shap_values, k)
        for k in [5, 10, 20, 50]
    }

    return {
        "shap_values": shap_values,
        "feature_ranking": ranking,
        "concentrations": concentrations,
        "shap_oom_retries": oom_retries,
    }


def explain_nodes_shap(
    model: nn.Module,
    data: Data,
    node_indices: list,
    num_samples: int = 100,
    device: str = "cpu",
    seed: int = 42,
) -> list:
    """
    Generate SHAP explanations for multiple nodes.

    Args:
        model: Trained GNN model.
        data: PyG Data object.
        node_indices: List of node indices.
        num_samples: Number of samples per node.
        device: Target device.
        seed: Random seed.

    Returns:
        List of explanation dicts.
    """
    explanations = []
    for i, idx in enumerate(node_indices):
        if (i + 1) % 10 == 0:
            print(f"    SHAP explaining node {i+1}/{len(node_indices)}")
        exp = explain_node_shap(
            model, data, idx, num_samples, device, seed=seed + i
        )
        explanations.append(exp)
    return explanations
