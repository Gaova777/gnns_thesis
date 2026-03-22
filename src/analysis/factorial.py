"""
Factorial analysis for identifying significant interactions.

Uses multi-factor ANOVA to determine how architecture, balancing
technique, and explainer method interact to affect stability metrics.
"""

import pandas as pd
import numpy as np
from typing import Optional

try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


def run_factorial_anova(
    df: pd.DataFrame,
    metric_col: str = "stab_jaccard_mean",
    factors: list = None,
) -> dict:
    """
    Run factorial ANOVA on stability metrics.

    Tests main effects and interactions of Architecture × Balancing × Explainer.

    Args:
        df: Results DataFrame with columns for factors and metrics.
        metric_col: Name of the metric column to analyze.
        factors: List of factor column names.

    Returns:
        Dict with ANOVA table, significant factors, and effect sizes.
    """
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels is required for factorial analysis")

    if factors is None:
        factors = ["architecture", "balancing", "explainer"]

    # Ensure factors are categorical
    for f in factors:
        df[f] = df[f].astype(str)

    # Build formula with all main effects and interactions
    formula = f"{metric_col} ~ " + " * ".join([f"C({f})" for f in factors])

    model = ols(formula, data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    # Identify significant effects (p < 0.05)
    significant = anova_table[anova_table["PR(>F)"] < 0.05].index.tolist()

    # Effect sizes (eta-squared)
    ss_total = anova_table["sum_sq"].sum()
    eta_sq = (anova_table["sum_sq"] / ss_total).to_dict()

    return {
        "anova_table": anova_table,
        "significant_effects": significant,
        "eta_squared": eta_sq,
        "r_squared": model.rsquared,
        "model_summary": str(model.summary()),
    }


def run_posthoc_tukey(
    df: pd.DataFrame,
    metric_col: str = "stab_jaccard_mean",
    group_col: str = "architecture",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Run Tukey HSD post-hoc test for pairwise comparisons.

    Args:
        df: Results DataFrame.
        metric_col: Metric to compare.
        group_col: Factor to compare groups of.
        alpha: Significance level.

    Returns:
        Tukey HSD results DataFrame.
    """
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels is required")

    result = pairwise_tukeyhsd(df[metric_col], df[group_col], alpha=alpha)
    return pd.DataFrame(
        data=result._results_table.data[1:],
        columns=result._results_table.data[0],
    )


def analyze_degradation_by_imbalance(
    df: pd.DataFrame,
    stability_col: str = "stab_jaccard_mean",
    scenario_col: str = "scenario",
) -> pd.DataFrame:
    """
    Analyze how stability degrades across imbalance levels.

    Computes mean and CI for each scenario, grouped by architecture.

    Args:
        df: Results DataFrame.
        stability_col: Stability metric column.
        scenario_col: Scenario column.

    Returns:
        Summary DataFrame.
    """
    from scipy import stats

    summary = df.groupby([scenario_col, "architecture"]).agg(
        mean=(stability_col, "mean"),
        std=(stability_col, "std"),
        count=(stability_col, "count"),
    ).reset_index()

    # 95% CI
    summary["ci_95"] = summary.apply(
        lambda r: stats.t.ppf(0.975, r["count"] - 1) * r["std"] / np.sqrt(r["count"])
        if r["count"] > 1 else 0,
        axis=1,
    )

    return summary
