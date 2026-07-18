"""
Stability metrics for evaluating XAI explanation consistency.

Implements:
  - Jaccard Index (subgraph overlap)
  - Spearman Rank Correlation (feature ranking agreement)
  - SHAP Concentration (attribution focus)
  - Fidelity+ / Fidelity- (via PyG built-in)
"""

import numpy as np
from scipy import stats
from itertools import combinations
from typing import Optional


def jaccard_index(set_a: set, set_b: set) -> float:
    """
    Compute Jaccard Index between two sets (of edges or nodes).

    J(A, B) = |A ∩ B| / |A ∪ B|

    Args:
        set_a: First set of elements.
        set_b: Second set of elements.

    Returns:
        Jaccard index in [0, 1]. Returns 0 if both sets are empty.
    """
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def pairwise_jaccard(subgraphs: list) -> dict:
    """
    Compute pairwise Jaccard indices across multiple subgraph replicas.

    Args:
        subgraphs: List of edge sets from multiple replicas.

    Returns:
        Dict with mean, std, min, max, and all pairwise values.
    """
    if len(subgraphs) < 2:
        # AUDIT FIX: <2 usable replicas => stability is NOT measurable. Return NaN,
        # never a spurious perfect 1.0. Downstream must treat NaN as "no medible".
        nan = float("nan")
        return {"mean": nan, "std": nan, "min": nan, "max": nan,
                "values": [], "n_replicas": len(subgraphs),
                "reason": "insufficient_replicas"}

    jaccard_values = []
    for a, b in combinations(range(len(subgraphs)), 2):
        j = jaccard_index(subgraphs[a], subgraphs[b])
        jaccard_values.append(j)

    return {
        "mean": float(np.mean(jaccard_values)),
        "std": float(np.std(jaccard_values)),
        "min": float(np.min(jaccard_values)),
        "max": float(np.max(jaccard_values)),
        "values": jaccard_values,
    }


def spearman_rank_agreement(
    ranking_a: np.ndarray,
    ranking_b: np.ndarray,
    top_k: Optional[int] = None,
) -> float:
    """
    Compute Spearman rank correlation between two feature rankings.

    Args:
        ranking_a: Feature indices sorted by importance (descending).
        ranking_b: Feature indices sorted by importance (descending).
        top_k: If specified, only consider the top-k features.

    Returns:
        Spearman correlation coefficient in [-1, 1].
    """
    if top_k:
        ranking_a = ranking_a[:top_k]
        ranking_b = ranking_b[:top_k]

    # Convert rankings to rank positions
    n = max(len(ranking_a), len(ranking_b))
    ranks_a = np.zeros(n)
    ranks_b = np.zeros(n)

    for pos, feat in enumerate(ranking_a):
        if feat < n:
            ranks_a[feat] = pos
    for pos, feat in enumerate(ranking_b):
        if feat < n:
            ranks_b[feat] = pos

    corr, _ = stats.spearmanr(ranks_a, ranks_b)
    return float(corr) if not np.isnan(corr) else 0.0


def pairwise_spearman(
    rankings: list,
    top_k: Optional[int] = None,
) -> dict:
    """
    Compute pairwise Spearman correlations across multiple ranking replicas.

    Args:
        rankings: List of feature ranking arrays.
        top_k: Top-k features to consider.

    Returns:
        Dict with mean, std, min, max.
    """
    if len(rankings) < 2:
        # AUDIT FIX: <2 usable replicas => not measurable. NaN, not a fake 1.0.
        nan = float("nan")
        return {"mean": nan, "std": nan, "min": nan, "max": nan,
                "n_replicas": len(rankings), "reason": "insufficient_replicas"}

    spearman_values = []
    for a, b in combinations(range(len(rankings)), 2):
        rho = spearman_rank_agreement(rankings[a], rankings[b], top_k)
        spearman_values.append(rho)

    return {
        "mean": float(np.mean(spearman_values)),
        "std": float(np.std(spearman_values)),
        "min": float(np.min(spearman_values)),
        "max": float(np.max(spearman_values)),
    }


def shap_concentration(
    shap_values: np.ndarray,
    top_k: int = 10,
) -> float:
    """
    SHAP Concentration metric.

    Measures what fraction of total attribution is concentrated
    in the top-k most important features.

    Args:
        shap_values: SHAP values array.
        top_k: Number of top features.

    Returns:
        Concentration in [0, 1].
    """
    abs_vals = np.abs(shap_values)
    total = abs_vals.sum()
    if total == 0:
        return 0.0
    sorted_vals = np.sort(abs_vals)[::-1]
    return float(sorted_vals[:top_k].sum() / total)


def compute_stability_metrics(
    stochastic_result: dict,
    top_k_features: int = 20,
) -> dict:
    """
    Compute all stability metrics from a stochastic test result.

    Args:
        stochastic_result: Output from run_stochastic_replicas().
        top_k_features: Top-k for Spearman and concentration.

    Returns:
        Dict with jaccard, spearman, and concentration metrics.
    """
    metrics = {
        "node_idx": stochastic_result["node_idx"],
        "method": stochastic_result["method"],
        "num_replicas": stochastic_result["num_replicas"],
    }

    # Jaccard (subgraph stability)
    if stochastic_result["subgraphs"]:
        metrics["jaccard"] = pairwise_jaccard(stochastic_result["subgraphs"])

    # Spearman (feature ranking stability)
    if stochastic_result["feature_rankings"]:
        rankings = [np.array(r) for r in stochastic_result["feature_rankings"]]
        metrics["spearman"] = pairwise_spearman(rankings, top_k=top_k_features)

    # SHAP Concentration
    if stochastic_result.get("shap_values"):
        concentrations = [
            shap_concentration(np.array(sv), top_k=top_k_features)
            for sv in stochastic_result["shap_values"]
        ]
        metrics["shap_concentration"] = {
            "mean": float(np.mean(concentrations)),
            "std": float(np.std(concentrations)),
            "values": concentrations,
        }

    return metrics


def compute_perturbation_stability(
    perturbation_result: dict,
    top_k_features: int = 20,
) -> dict:
    """
    Compute stability metrics between original and perturbed explanations.

    Args:
        perturbation_result: Output from run_perturbation_test().
        top_k_features: Top-k for ranking comparison.

    Returns:
        Dict with metrics organized by noise level.
    """
    metrics = {
        "node_idx": perturbation_result["node_idx"],
        "method": perturbation_result["method"],
        "noise_levels": {},
    }

    orig_subgraph = perturbation_result["original_subgraph"]
    orig_ranking = np.array(perturbation_result["original_ranking"])

    for sigma, perturbed in perturbation_result["perturbed"].items():
        level_metrics = {}

        # Jaccard vs original
        if orig_subgraph and perturbed["subgraphs"]:
            j_values = [
                jaccard_index(orig_subgraph, sg) for sg in perturbed["subgraphs"]
            ]
            level_metrics["jaccard_vs_original"] = {
                "mean": float(np.mean(j_values)),
                "std": float(np.std(j_values)),
            }

        # Spearman vs original
        if len(orig_ranking) > 0 and perturbed["feature_rankings"]:
            s_values = [
                spearman_rank_agreement(orig_ranking, np.array(r), top_k_features)
                for r in perturbed["feature_rankings"]
            ]
            level_metrics["spearman_vs_original"] = {
                "mean": float(np.mean(s_values)),
                "std": float(np.std(s_values)),
            }

        metrics["noise_levels"][str(sigma)] = level_metrics

    return metrics
