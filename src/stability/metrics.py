"""
Stability metrics for evaluating XAI explanation consistency.

Implements:
  - v4 (score-based, NaN + reason when not measurable): spearman_full, spearman_topk,
    jaccard_edges_topk, spearman_edges, node_stability_v4, aggregate_nodes_v4
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
    ranking_a = np.asarray(ranking_a)
    ranking_b = np.asarray(ranking_b)
    if ranking_a.size == 0 or ranking_b.size == 0:
        # AUDIT FIX (v4): an empty ranking means the explainer produced NO feature
        # attribution (e.g. PGExplainer has no node_mask). That is "not applicable",
        # not "zero agreement": returning 0.0 here was read as PGExplainer "mode
        # collapse". NaN propagates to the mean so it can never pass as a value.
        return float("nan")

    # AUDIT FIX (R1): size the rank vectors by the NUMBER OF FEATURES (max index + 1),
    # not by top_k. The rankings hold feature INDICES (argsort output, e.g. 0..165), so
    # the rank vectors must be indexed by feature id to be comparable. The previous code
    # used np.zeros(top_k) + `if feat < top_k`, which silently DROPPED every feature whose
    # index exceeded top_k -> with top_k=20 over 166 features only ~2-3 of the top-20
    # survived and the rest stayed tied at 0, corrupting the Spearman. Now correct for any
    # top_k; identical to the old behaviour when top_k=None (used by the phase1 pipeline).
    n_features = int(max(ranking_a.max(), ranking_b.max())) + 1

    # Features outside the (optionally truncated) ranking are tied at the worst rank.
    ranks_a = np.full(n_features, n_features, dtype=float)
    ranks_b = np.full(n_features, n_features, dtype=float)

    top_a = ranking_a[:top_k] if top_k else ranking_a
    top_b = ranking_b[:top_k] if top_k else ranking_b
    for pos, feat in enumerate(top_a):
        ranks_a[int(feat)] = pos
    for pos, feat in enumerate(top_b):
        ranks_b[int(feat)] = pos

    corr, _ = stats.spearmanr(ranks_a, ranks_b)
    # Undefined correlation (a constant rank vector) is NOT agreement 0: NaN (v4 fix).
    return float(corr) if not np.isnan(corr) else float("nan")


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


# ════════════════════════════════════════════════════════════════════════════
# v4 metrics — computed on SCORE vectors (not argsort rankings)
# ════════════════════════════════════════════════════════════════════════════
#
# Conventions (v4):
#   * Inputs are importance SCORES per replica, aligned across replicas: a matrix
#     [n_replicas, n_items] (items = features, or the edges of the node's explanation
#     subgraph). Ranks are taken with scipy's average-rank rule, so ties get the mean
#     rank instead of an arbitrary argsort order.
#   * A metric that cannot be measured is NaN, never 0.0, and carries a reason:
#       REASON_NA            the explainer does not produce this output (PGExplainer has
#                            no feature mask; ShapleyFeatures has no edge mask)
#       REASON_FEW_EDGES     the explanation subgraph has < 2 edges (Jaccard/Spearman of
#                            edges is trivially 1 or undefined there)
#       REASON_FEW_REPLICAS  < 2 usable replicas
#       REASON_CONSTANT      a replica's score vector is constant (rank correlation
#                            undefined; e.g. an all-zero mask)
#   * spearman_full: Spearman over ALL items (the primary measure of ORDER stability).
#   * spearman_topk: the v3 measure (top-k by position, the rest tied at the worst rank).
#     With k=20 of 165 features, 145 features are tied, so it mostly measures SET overlap
#     of the top-k; kept only as a sensitivity analysis.

REASON_NA = "no_aplica"
REASON_FEW_EDGES = "menos_de_2_aristas"
REASON_FEW_REPLICAS = "menos_de_2_replicas"
REASON_CONSTANT = "vector_constante"

_NAN = float("nan")


def _n_items(scores) -> Optional[int]:
    """Number of items of a score matrix, or None if the explainer gave no output."""
    if scores is None:
        return None
    m = np.asarray(scores, dtype=float)
    return int(m.shape[1]) if m.ndim == 2 else None


def _as_score_matrix(scores) -> Optional[np.ndarray]:
    """[n_rep, n_items] float array, dropping replicas that are missing/all-NaN."""
    if scores is None:
        return None
    m = np.asarray(scores, dtype=float)
    if m.ndim != 2 or m.shape[1] == 0:
        return None
    keep = ~np.all(np.isnan(m), axis=1)
    return m[keep]


def _pairwise(values_fn, m: np.ndarray) -> tuple[float, float, list, Optional[str]]:
    vals = []
    reasons = []
    for a, b in combinations(range(m.shape[0]), 2):
        v, r = values_fn(m[a], m[b])
        if r is not None:
            reasons.append(r)
        else:
            vals.append(v)
    if not vals:
        return _NAN, _NAN, [], (reasons[0] if reasons else REASON_FEW_REPLICAS)
    return float(np.mean(vals)), float(np.std(vals)), vals, None


def _spearman_scores(a: np.ndarray, b: np.ndarray) -> tuple[float, Optional[str]]:
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return _NAN, REASON_CONSTANT
    rho = stats.spearmanr(a, b).statistic  # average ranks for ties
    if np.isnan(rho):
        return _NAN, REASON_CONSTANT
    return float(rho), None


def spearman_full(scores) -> dict:
    """Mean pairwise Spearman between replicas over the FULL score vector.

    Args:
        scores: [n_replicas, n_items] importance scores (absolute values are the
            caller's choice; they are ranked as given). ``None``/empty → no_aplica.
    Returns:
        {"mean", "std", "n_pairs", "reason"}; reason is None when measurable.
    """
    m = _as_score_matrix(scores)
    if m is None:
        return {"mean": _NAN, "std": _NAN, "n_pairs": 0, "reason": REASON_NA}
    if m.shape[0] < 2:
        return {"mean": _NAN, "std": _NAN, "n_pairs": 0, "reason": REASON_FEW_REPLICAS}
    mean, std, vals, reason = _pairwise(_spearman_scores, m)
    return {"mean": mean, "std": std, "n_pairs": len(vals), "reason": reason}


def _topk_ranking(v: np.ndarray) -> np.ndarray:
    """Descending argsort with a deterministic tie-break (stable on -v)."""
    return np.argsort(-v, kind="stable")


def spearman_topk(scores, k: int = 20) -> dict:
    """v3-style Spearman on the top-k positions (sensitivity analysis only).

    Items outside each replica's top-k are tied at the worst rank; see the module
    note: with k << n_items this measures top-k SET agreement more than order.
    """
    m = _as_score_matrix(scores)
    if m is None:
        return {"mean": _NAN, "std": _NAN, "n_pairs": 0, "reason": REASON_NA}
    if m.shape[0] < 2:
        return {"mean": _NAN, "std": _NAN, "n_pairs": 0, "reason": REASON_FEW_REPLICAS}

    def fn(a, b):
        if np.ptp(a) == 0 or np.ptp(b) == 0:
            return _NAN, REASON_CONSTANT
        rho = spearman_rank_agreement(_topk_ranking(a), _topk_ranking(b), top_k=k)
        return (rho, None) if rho == rho else (_NAN, REASON_CONSTANT)

    mean, std, vals, reason = _pairwise(fn, m)
    return {"mean": mean, "std": std, "n_pairs": len(vals), "reason": reason}


def effective_topk_edges(n_edges: int, k: int) -> int:
    """Top-k used for edge Jaccard: min(k, n_edges - 1).

    If k >= n_edges every replica selects ALL edges and Jaccard is 1 by construction
    (the artefact v3 had on 0–2-edge subgraphs). Capping at n_edges - 1 keeps the
    selection a proper subset so the index can actually disagree.
    """
    return max(1, min(int(k), int(n_edges) - 1))


def jaccard_edges_topk(edge_scores, k: int = 20) -> dict:
    """Mean pairwise Jaccard of the top-k edges between replicas.

    Only defined when the explanation subgraph has >= 2 edges (else NaN,
    ``menos_de_2_aristas``); ``None`` scores → ``no_aplica``. The effective k is
    ``effective_topk_edges(n_edges, k)``. Ties at the cut are broken by edge
    position (stable sort), identically for every replica.
    """
    n_edges = _n_items(edge_scores)
    if n_edges is None:
        return {"mean": _NAN, "std": _NAN, "n_pairs": 0, "k_eff": 0, "reason": REASON_NA}
    if n_edges < 2:
        return {"mean": _NAN, "std": _NAN, "n_pairs": 0, "k_eff": 0,
                "reason": REASON_FEW_EDGES}
    m = _as_score_matrix(edge_scores)
    if m.shape[0] < 2:
        return {"mean": _NAN, "std": _NAN, "n_pairs": 0, "k_eff": 0,
                "reason": REASON_FEW_REPLICAS}
    k_eff = effective_topk_edges(n_edges, k)
    sets = [set(_topk_ranking(row)[:k_eff].tolist()) for row in m]
    vals = [len(sets[a] & sets[b]) / len(sets[a] | sets[b])
            for a, b in combinations(range(len(sets)), 2)]
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
            "n_pairs": len(vals), "k_eff": k_eff, "reason": None}


def spearman_edges(edge_scores) -> dict:
    """Mean pairwise Spearman of the EDGE-mask scores between replicas.

    The stability measure for PGExplainer (which only produces an edge mask). NaN
    with ``menos_de_2_aristas`` if the subgraph has < 2 edges.
    """
    n_edges = _n_items(edge_scores)
    if n_edges is None:
        return {"mean": _NAN, "std": _NAN, "n_pairs": 0, "reason": REASON_NA}
    if n_edges < 2:
        return {"mean": _NAN, "std": _NAN, "n_pairs": 0, "reason": REASON_FEW_EDGES}
    return spearman_full(edge_scores)


def node_stability_v4(feat_scores, edge_scores, top_k_features: int = 20,
                      top_k_edges: int = 20) -> dict:
    """All v4 per-node metrics for one (model, explainer, node).

    Args:
        feat_scores: [n_rep, n_features] or None (explainer without feature output).
        edge_scores: [n_rep, n_edges] or None (explainer without edge output).
    Returns flat dict: spearman_full, spearman_topk, jaccard_edges_topk,
    spearman_edges, each with a ``<name>_reason`` (None if measurable).
    """
    out = {}
    for name, res in (("spearman_full", spearman_full(feat_scores)),
                      ("spearman_topk", spearman_topk(feat_scores, top_k_features)),
                      ("jaccard_edges_topk", jaccard_edges_topk(edge_scores, top_k_edges)),
                      ("spearman_edges", spearman_edges(edge_scores))):
        out[name] = res["mean"]
        out[f"{name}_reason"] = res["reason"]
    return out


def aggregate_nodes_v4(per_node: list[dict], metric: str,
                       mask: Optional[list] = None) -> dict:
    """Mean over nodes of a v4 metric, ignoring NaN; reason = most frequent node reason
    when nothing is measurable. ``mask`` (bools) restricts to a subset (e.g. TP nodes)."""
    rows = [r for i, r in enumerate(per_node) if mask is None or mask[i]]
    vals = [r[metric] for r in rows if r.get(metric) == r.get(metric)]
    if vals:
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                "n": len(vals), "reason": None}
    reasons = [r.get(f"{metric}_reason") for r in rows if r.get(f"{metric}_reason")]
    reason = max(set(reasons), key=reasons.count) if reasons else "sin_nodos"
    return {"mean": _NAN, "std": _NAN, "n": 0, "reason": reason}
