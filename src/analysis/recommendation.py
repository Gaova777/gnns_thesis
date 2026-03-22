"""
Recommendation matrix generator.

Filters experiment results to identify configurations meeting
success thresholds and builds the final recommendation matrix
with confidence intervals.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Optional


def build_recommendation_matrix(
    df: pd.DataFrame,
    f1_min: float = 0.80,
    mcc_min: float = 0.70,
    jaccard_min: float = 0.70,
    confidence_level: float = 0.95,
    n_bootstrap: int = 1000,
) -> pd.DataFrame:
    """
    Build the final recommendation matrix filtering by success criteria.

    Identifies configurations where:
      - F1 (illicit class) ≥ f1_min
      - MCC ≥ mcc_min
      - Jaccard stability ≥ jaccard_min

    Args:
        df: Full results DataFrame.
        f1_min: Minimum F1 threshold.
        mcc_min: Minimum MCC threshold.
        jaccard_min: Minimum Jaccard stability threshold.
        confidence_level: Confidence level for intervals.
        n_bootstrap: Number of bootstrap samples.

    Returns:
        Recommendation DataFrame with aggregated metrics and CIs.
    """
    # Group by configuration
    group_cols = ["scenario", "architecture", "balancing", "explainer"]

    grouped = df.groupby(group_cols).agg(
        f1_mean=("pred_f1", "mean"),
        f1_std=("pred_f1", "std"),
        mcc_mean=("pred_mcc", "mean"),
        mcc_std=("pred_mcc", "std"),
        jaccard_mean=("stab_jaccard_mean", "mean"),
        jaccard_std=("stab_jaccard_mean", "std"),
        spearman_mean=("stab_spearman_mean", "mean"),
        n_runs=("pred_f1", "count"),
    ).reset_index()

    # Filter by thresholds
    passing = grouped[
        (grouped["f1_mean"] >= f1_min)
        & (grouped["mcc_mean"] >= mcc_min)
        & (grouped["jaccard_mean"] >= jaccard_min)
    ].copy()

    if len(passing) == 0:
        print("  ⚠ No configurations meet all threshold criteria.")
        print(f"  Closest configurations:")
        # Show top 5 closest
        grouped["score"] = (
            grouped["f1_mean"] / f1_min
            + grouped["mcc_mean"] / mcc_min
            + grouped["jaccard_mean"] / jaccard_min
        )
        return grouped.nlargest(5, "score")

    # Add confidence intervals
    alpha = 1 - confidence_level
    for col in ["f1", "mcc", "jaccard"]:
        n = passing["n_runs"]
        mean = passing[f"{col}_mean"]
        std = passing[f"{col}_std"]
        t_val = stats.t.ppf(1 - alpha / 2, n - 1)
        passing[f"{col}_ci_lower"] = mean - t_val * std / np.sqrt(n)
        passing[f"{col}_ci_upper"] = mean + t_val * std / np.sqrt(n)

    # Sort by combined score
    passing["combined_score"] = (
        passing["f1_mean"] + passing["mcc_mean"] + passing["jaccard_mean"]
    )
    passing = passing.sort_values("combined_score", ascending=False)

    print(f"  ✓ {len(passing)} configurations meet all criteria")
    return passing


def export_latex_table(
    matrix: pd.DataFrame,
    filepath: str = "./results/recommendation_matrix.tex",
    caption: str = "Recommendation Matrix",
) -> None:
    """Export recommendation matrix as LaTeX table."""
    cols_to_show = [
        "scenario", "architecture", "balancing", "explainer",
        "f1_mean", "mcc_mean", "jaccard_mean", "spearman_mean",
    ]
    available = [c for c in cols_to_show if c in matrix.columns]

    latex = matrix[available].to_latex(
        index=False,
        float_format="%.3f",
        caption=caption,
        label="tab:recommendation_matrix",
    )

    with open(filepath, "w") as f:
        f.write(latex)
    print(f"  Saved LaTeX table to {filepath}")
