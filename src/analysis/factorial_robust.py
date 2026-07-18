"""
PARCHE 3 (auditoría) — análisis estadístico robusto para respuestas acotadas
y medidas repetidas.

Módulo NUEVO: colócalo en src/analysis/factorial_robust.py

Motivación: `factorial.py` usa ANOVA (ols) con dos supuestos violados en este
diseño:
  1. Independencia: los mismos nodos se explican bajo múltiples condiciones
     (arquitectura × balanceo × explicador) → medidas repetidas.
  2. Normalidad/homocedasticidad: Jaccard ∈ [0,1] y Spearman ∈ [-1,1] son
     respuestas acotadas; la ANOVA clásica no es apropiada.

Este módulo ofrece dos caminos:
  A) MixedLM (statsmodels) — cuando hay observaciones DESAGREGADAS (una fila por
     nodo o por par de réplicas) con un identificador de grupo (node_id). El nodo
     entra como efecto aleatorio. Es la opción correcta si guardas la estabilidad
     por nodo, no solo el promedio por config.
  B) ART-ANOVA (Aligned Rank Transform) — cuando solo hay medias por config. Es
     una alternativa no paramétrica que sí admite interacciones, apropiada para
     respuestas acotadas. Implementada aquí sin dependencias externas.

Ambos caminos reportan tamaños de efecto, que son el foco (no solo p-values).
"""

import numpy as np
import pandas as pd
from itertools import combinations

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


# --------------------------------------------------------------------------- #
#  A) Modelo mixto (medidas repetidas) — requiere datos desagregados por nodo #
# --------------------------------------------------------------------------- #
def run_mixed_model(
    df: pd.DataFrame,
    metric_col: str = "jaccard",
    factors: list = None,
    group_col: str = "node_id",
) -> dict:
    """
    Modelo lineal mixto con `group_col` (p.ej. node_id) como efecto aleatorio.

    df debe estar DESAGREGADO: una fila por observación de estabilidad (por nodo,
    o por par de réplicas), con las columnas de factores y `group_col`.

    Returns dict con la tabla de efectos fijos, varianza del efecto aleatorio,
    y un pseudo-R2.
    """
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels es requerido para el modelo mixto")
    if factors is None:
        factors = ["architecture", "balancing", "explainer"]
    for f in factors:
        df[f] = df[f].astype(str)

    fixed = " * ".join([f"C({f})" for f in factors])
    formula = f"{metric_col} ~ {fixed}"
    md = smf.mixedlm(formula, df, groups=df[group_col])
    mfit = md.fit(reml=True, method="lbfgs")

    return {
        "summary": str(mfit.summary()),
        "fixed_effects": mfit.fe_params.to_dict(),
        "pvalues": mfit.pvalues.to_dict(),
        "random_effect_var": float(mfit.cov_re.iloc[0, 0]) if mfit.cov_re.size else np.nan,
        "residual_var": float(mfit.scale),
    }


# --------------------------------------------------------------------------- #
#  B) ART-ANOVA (Aligned Rank Transform) — para medias por config acotadas     #
# --------------------------------------------------------------------------- #
def _aligned_rank_transform(df, response, factors):
    """
    Alinea la respuesta para cada efecto (main e interacciones) restando todos
    los demás efectos estimados por medias de celda, y rankea el residuo alineado.
    Devuelve un dict {efecto: columna_rankeada}.
    """
    grand = df[response].mean()
    # medias marginales por cada subconjunto de factores
    def cell_mean(cols):
        if not cols:
            return pd.Series(grand, index=df.index)
        return df.groupby(cols)[response].transform("mean")

    effects = []
    for r in range(1, len(factors) + 1):
        effects += [list(c) for c in combinations(factors, r)]

    aligned_cols = {}
    for eff in effects:
        # efecto "puro" = media de celda del efecto menos las medias de todos los
        # sub-efectos contenidos (inclusión-exclusión), centrado en la gran media
        mu_eff = cell_mean(eff)
        adj = mu_eff.copy()
        for r in range(1, len(eff)):
            for sub in combinations(eff, r):
                sign = (-1) ** (len(eff) - len(sub))
                adj = adj + sign * cell_mean(list(sub))
        # término de la gran media
        adj = adj + ((-1) ** len(eff)) * grand
        # residuo alineado = respuesta - (todo lo demás) = respuesta - (mu_full - efecto_puro)
        mu_full = cell_mean(factors)
        aligned = df[response] - (mu_full - adj)
        aligned_cols["*".join(eff)] = aligned.rank()
    return aligned_cols


def run_art_anova(
    df: pd.DataFrame,
    metric_col: str = "stab_jaccard_mean",
    factors: list = None,
) -> dict:
    """
    ANOVA sobre transformación de rango alineado (ART). No paramétrico, admite
    interacciones, apropiado para respuestas acotadas (Jaccard, Spearman).

    Para cada efecto se ajusta un ANOVA sobre su columna alineada+rankeada y se
    reporta el término correspondiente. Incluye eta-cuadrado parcial como tamaño
    de efecto.
    """
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels es requerido para ART-ANOVA")
    if factors is None:
        factors = ["architecture", "balancing", "explainer"]
    d = df.copy()
    d = d.dropna(subset=[metric_col])
    for f in factors:
        d[f] = d[f].astype(str)

    aligned = _aligned_rank_transform(d, metric_col, factors)
    results = {}
    full_formula_rhs = " * ".join([f"C({f})" for f in factors])
    for eff, col in aligned.items():
        d["_art_"] = col.values
        m = smf.ols(f"_art_ ~ {full_formula_rhs}", data=d).fit()
        tab = anova_lm(m, typ=2)
        term = " * ".join([f"C({f})" for f in eff.split("*")])
        # localizar la fila del término (statsmodels ordena los C(a):C(b))
        key = None
        for idx in tab.index:
            fs = set(part for part in idx.replace(":", "*").split("*"))
            if fs == set(f"C({f})" for f in eff.split("*")):
                key = idx; break
        if key is not None:
            ss_total = tab["sum_sq"].sum()
            results[eff] = {
                "F": float(tab.loc[key, "F"]),
                "p_value": float(tab.loc[key, "PR(>F)"]),
                "partial_eta_sq": float(tab.loc[key, "sum_sq"] / ss_total),
            }
    return results


def cohens_d(a, b):
    """Tamaño de efecto para una comparación pareada de dos grupos."""
    a, b = np.asarray(a), np.asarray(b)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else np.nan


# --------------------------------------------------------------------------- #
#  Guard de degeneración — evita reportar estadística sobre datos sin varianza #
# --------------------------------------------------------------------------- #
def check_design(df, metric_col, factors, min_var=1e-8, min_per_cell=2):
    """
    Revisa si el diseño soporta un análisis factorial válido. Devuelve un dict
    con banderas; si `ok` es False, NO se debe confiar en F/p de ART-ANOVA
    (típicamente porque la respuesta es casi constante — el caso de modelos que
    no discriminan y producen Jaccard=1.0 en todas partes).
    """
    d = df.dropna(subset=[metric_col])
    var = float(d[metric_col].var())
    n_unique = int(d[metric_col].nunique())
    cell_counts = d.groupby([f for f in factors]).size()
    small_cells = int((cell_counts < min_per_cell).sum())
    levels = {f: int(d[f].nunique()) for f in factors}
    degenerate_factor = [f for f, k in levels.items() if k < 2]
    ok = (var >= min_var) and (n_unique >= 3) and (not degenerate_factor)
    msg = []
    if var < min_var:
        msg.append(f"respuesta casi constante (var={var:.2e}) — los modelos no discriminan; "
                   "no interpretar F/p")
    if n_unique < 3:
        msg.append(f"solo {n_unique} valores distintos de la respuesta")
    if degenerate_factor:
        msg.append(f"factores con un solo nivel: {degenerate_factor}")
    if small_cells:
        msg.append(f"{small_cells} celdas con < {min_per_cell} observaciones")
    return {"ok": ok, "variance": var, "n_unique": n_unique,
            "levels": levels, "small_cells": small_cells, "warnings": msg}
