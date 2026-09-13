# -*- coding: utf-8 -*-
"""Robustez a 3 semillas de los TRES explicadores y de las metricas de rendimiento.

Complementa a escenario_3semillas.py (que solo cubre GNNExplainer por escenario).
Aqui se consolida, sobre results_seedsweep/ y results_v3/reeval_metrics.csv:

  1. Estabilidad por arquitectura de cada familia de explicador, con la metrica que
     le corresponde por diseno y promediada sobre las tres semillas de modelo:
       - GNNExplainer : Spearman de features (stab_spearman_mean)
       - GNNShap      : Spearman de features (stab_spearman_mean)  [no produce aristas]
       - PGExplainer  : Jaccard de aristas   (stab_jaccard_mean)   [no produce features]
     NO se mezclan Spearman de features y Jaccard de aristas en una misma columna.

  2. ROC-AUC / PR-AUC / precision@50 a 3 semillas por split (val/test), desde el
     reeval (columna seed anadida por reeval_rocauc.py).

Genera:
  tesis_latex/tables/elliptic_robustez_3semillas.tex   (Spearman features: GNNExplainer vs GNNShap)
  tesis_latex/chapter_4/images_ch4/robustez_explicadores.png
  results_seedsweep/robustez_3semillas_summary.csv      (todas las familias + metricas, para citar)

La raiz del repo se autodetecta; se puede forzar con GNN_REPO_ROOT. El script es
defensivo ante columnas vacias o familias ausentes (no aborta: reporta n/d).
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(os.environ.get("GNN_REPO_ROOT", Path(__file__).resolve().parents[2]))
CSV = ROOT / "results_seedsweep" / "xai-gnn-stability-seedsweep.csv"
REEVAL = ROOT / "results_v3" / "reeval_metrics.csv"
TAB = ROOT / "tesis_latex" / "tables" / "elliptic_robustez_3semillas.tex"
FIG = ROOT / "tesis_latex" / "chapter_4" / "images_ch4" / "robustez_explicadores.png"
SUMMARY = ROOT / "results_seedsweep" / "robustez_3semillas_summary.csv"

ARQ = ["GraphSAGE", "GAT", "GCN", "TAGCN"]
# (explicador -> (columna de estabilidad, etiqueta de metrica))
SPEC = {
    "GNNExplainer": ("stab_spearman_mean", "Spearman (features)"),
    "GNNShap": ("stab_spearman_mean", "Spearman (features)"),
    "PGExplainer": ("stab_jaccard_mean", "Jaccard (aristas)"),
}


def com(v, d=3):
    return "n/d" if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{v:.{d}f}".replace(".", ",")


def ci95(vals):
    """Media e IC95 (t de Student). Devuelve (media, lo, hi, n)."""
    v = np.asarray([x for x in vals if pd.notna(x)], dtype=float)
    n = len(v)
    if n == 0:
        return (np.nan, np.nan, np.nan, 0)
    m = float(v.mean())
    if n == 1:
        return (m, np.nan, np.nan, 1)
    se = v.std(ddof=1) / np.sqrt(n)
    if se == 0:
        return (m, m, m, n)
    lo, hi = stats.t.interval(0.95, n - 1, loc=m, scale=se)
    return (m, float(lo), float(hi), n)


def estabilidad(d, summary_rows):
    """Estabilidad por arquitectura y explicador (cada uno con su metrica)."""
    spear = {}  # {explicador: {arch: (m, lo, hi, n)}}  solo Spearman, para la tabla
    print("\n=== ESTABILIDAD A 3 SEMILLAS (por arquitectura) ===")
    for expl, (col, etq) in SPEC.items():
        sub = d[d["explainer"] == expl].copy()
        if sub.empty:
            print(f"\n[{expl}] sin filas en el sweep (aun no corrido).")
            continue
        sub[col] = pd.to_numeric(sub.get(col), errors="coerce")
        nseeds = sub["seed"].nunique() if "seed" in sub.columns else "?"
        print(f"\n[{expl}] metrica={etq}  ({nseeds} semillas)")
        per = {}
        for a in ARQ:
            m, lo, hi, n = ci95(sub[sub["architecture"] == a][col].values)
            per[a] = (m, lo, hi, n)
            print(f"   {a:10s} media={com(m)}  IC95=[{com(lo)}; {com(hi)}]  n={n}")
            summary_rows.append(dict(explainer=expl, metric=etq, architecture=a,
                                     mean=m, ci_lo=lo, ci_hi=hi, n=n))
        if col == "stab_spearman_mean":
            spear[expl] = per
    return spear


def rendimiento(summary_rows):
    """ROC-AUC / PR-AUC / precision@50 a 3 semillas por split, desde el reeval."""
    if not REEVAL.exists():
        print(f"\n[reeval] {REEVAL} no existe aun; se omite el bloque de rendimiento.")
        return
    r = pd.read_csv(REEVAL)
    if "seed" not in r.columns:
        print("\n[reeval] el CSV no trae columna seed (reeval antiguo); se omite.")
        return
    # El manuscrito reporta el rendimiento sobre los modelos que superan el filtro
    # de calidad; se replica ese filtro para que las cifras a 3 semillas sean
    # comparables con las de la semilla unica.
    qp = r["quality_passed"].astype(str).str.lower().isin(["true", "1"])
    r = r[qp]
    print("\n=== RENDIMIENTO A 3 SEMILLAS (reeval, modelos que pasan el filtro, por split) ===")
    for split in ["val", "test"]:
        s = r[r["split"] == split]
        nseeds = s["seed"].nunique()
        print(f"\n[{split}]  ({nseeds} semillas)")
        for metric in ["roc_auc", "pr_auc_trap", "prec_at_50"]:
            vals = pd.to_numeric(s[metric], errors="coerce")
            m, lo, hi, n = ci95(vals.values)
            print(f"   {metric:12s} media={com(m)}  IC95=[{com(lo)}; {com(hi)}]  n={n}")
            summary_rows.append(dict(explainer=f"reeval/{split}", metric=metric,
                                     architecture="TODAS", mean=m, ci_lo=lo, ci_hi=hi, n=n))


def escribir_tabla(spear):
    """Tabla LaTeX: Spearman de features a 3 semillas, GNNExplainer vs GNNShap."""
    cols = [e for e in ("GNNExplainer", "GNNShap") if e in spear]
    if not cols:
        print("\n[tabla] sin datos de Spearman aun; no se escribe la tabla.")
        return
    EOL = "\\\\"
    filas = []
    for a in ARQ:
        celdas = []
        for e in cols:
            m, lo, hi, n = spear[e].get(a, (np.nan, np.nan, np.nan, 0))
            celdas.append(f"{com(m)} [{com(lo)}; {com(hi)}]" if n >= 2 else com(m))
        filas.append(f"{a} & " + " & ".join(celdas) + f" {EOL}")
    enc = " & ".join(f"\\textbf{{{e}}}" for e in cols)
    align = "l" + "c" * len(cols)
    tex = rf"""\begin{{table}}[htbp]
\centering
\renewcommand{{\arraystretch}}{{1.3}}
\begin{{tabular}}{{{align}}}
\toprule
\textbf{{Arquitectura}} & {enc} {EOL}
\midrule
{chr(10).join(filas)}
\bottomrule
\end{{tabular}}
\caption{{Estabilidad de las explicaciones medida como correlación de Spearman sobre el ranking de \textit{{features}}, promediada sobre las tres semillas de modelo y con intervalo de confianza al 95\%. Ambos explicadores atribuyen importancia a \textit{{features}} y comparten escala, pero su lectura difiere. GNNExplainer discrimina entre arquitecturas y reproduce la partición en dos grupos, con GAT y GCN por encima de GraphSAGE y TAGCN. GNNShap, en cambio, se satura en valores próximos a la unidad para las cuatro arquitecturas, de modo que su estabilidad es alta pero no separa arquitecturas, y se muestra en la misma escala solo por comparabilidad. PGExplainer opera sobre aristas y se reporta aparte con la métrica de Jaccard.}}
\label{{tab:elliptic-robustez-3s}}
\end{{table}}
"""
    TAB.parent.mkdir(parents=True, exist_ok=True)
    TAB.write_text(tex, encoding="utf-8")
    print("\nescrito:", TAB)

    # figura: barras con IC por arquitectura, una serie por explicador de features
    x = np.arange(len(ARQ))
    w = 0.8 / max(1, len(cols))
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    colores = {"GNNExplainer": "#1C7293", "GNNShap": "#E07A5F"}
    for j, e in enumerate(cols):
        ys = [spear[e].get(a, (np.nan,))[0] for a in ARQ]
        errs = []
        for a in ARQ:
            m, lo, hi, n = spear[e].get(a, (np.nan, np.nan, np.nan, 0))
            errs.append((m - lo) if n >= 2 and pd.notna(lo) else 0.0)
        ax.bar(x + j * w - 0.4 + w / 2, ys, w, yerr=errs, capsize=3,
               label=e, color=colores.get(e, None), zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(ARQ)
    ax.set_ylabel("Spearman de features (3 semillas)")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", ls=":", color="0.85", zorder=0)
    ax.legend()
    ax.set_title("Estabilidad de explicadores de features a 3 semillas", fontsize=10)
    plt.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG, dpi=300)
    plt.close()
    print("escrito:", FIG)


def main():
    if not CSV.exists():
        raise FileNotFoundError(f"No existe {CSV}")
    d = pd.read_csv(CSV)
    summary_rows = []
    spear = estabilidad(d, summary_rows)
    rendimiento(summary_rows)
    escribir_tabla(spear)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(SUMMARY, index=False)
    print("\nescrito:", SUMMARY)


if __name__ == "__main__":
    main()
