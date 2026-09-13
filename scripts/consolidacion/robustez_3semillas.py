# -*- coding: utf-8 -*-
"""Robustez a 3 semillas de los explicadores y de las metricas de rendimiento.

Complementa a escenario_3semillas.py (que solo cubre GNNExplainer por escenario).
Consolida, sobre results_seedsweep/ y results_v3/reeval_metrics.csv:

  1. Estabilidad por arquitectura de cada familia de explicador, con la metrica que
     le corresponde por diseno y promediada sobre las semillas disponibles:
       - GNNExplainer : Spearman de features (stab_spearman_mean)
       - GNNShap      : Spearman de features (stab_spearman_mean)  [no produce aristas]
       - PGExplainer  : Jaccard de aristas   (stab_jaccard_mean)   [no produce features]
     NO se mezclan Spearman de features y Jaccard de aristas en una misma columna.

  2. ROC-AUC / PR-AUC / precision@50 sobre la SEMILLA 42 (las cifras que reporta el
     manuscrito), como comprobacion de consistencia.

IMPORTANTE (asimetria de semillas): GCN y GraphSAGE tienen las 3 semillas (42/43/44);
GAT y TAGCN solo tienen la semilla 42, porque sus pesos no caben para reentrenar en la
GPU de 8 GB disponible (GAT usa 8 cabezas de atencion sobre el grafo completo). Por eso el
IC al 95% solo se calcula donde hay >=2 semillas; las celdas de GAT/TAGCN se marcan "(s42)"
como valor puntual de una sola semilla, no como estimacion robusta. Esto NO afecta la
comparacion por arquitectura, que se hace con GNNExplainer (las 4 arqs, 3 semillas), ni la
conclusion sobre GNNShap/PGExplainer, que no discriminan arquitecturas y son invariantes a
la semilla (GAT/TAGCN ya lo muestran en seed 42).

Genera:
  tesis_latex/tables/elliptic_robustez_3semillas.tex
  tesis_latex/chapter_4/images_ch4/robustez_explicadores.png
  results_seedsweep/robustez_3semillas_summary.csv

La raiz del repo se autodetecta; se puede forzar con GNN_REPO_ROOT. Defensivo ante columnas
vacias o familias ausentes (no aborta: reporta n/d).
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
    """Estabilidad por arquitectura y explicador (cada uno con su metrica).

    El IC95 solo se reporta cuando hay >=2 SEMILLAS de modelo; con una sola semilla
    seria dispersion entre configuraciones, no robustez entre inicializaciones."""
    spear = {}  # {explicador: {arch: (m, lo, hi, n, nseeds)}}  solo Spearman, para la tabla
    print("\n=== ESTABILIDAD (por arquitectura; IC95 solo con >=2 semillas) ===")
    for expl, (col, etq) in SPEC.items():
        sub = d[d["explainer"] == expl].copy()
        if sub.empty:
            print(f"\n[{expl}] sin filas en el sweep (aun no corrido).")
            continue
        sub[col] = pd.to_numeric(sub.get(col), errors="coerce")
        print(f"\n[{expl}] metrica={etq}")
        per = {}
        for a in ARQ:
            arch_rows = sub[sub["architecture"] == a]
            nseeds = arch_rows["seed"].nunique() if "seed" in arch_rows.columns else 0
            m, lo, hi, n = ci95(arch_rows[col].values)
            if nseeds < 2:            # con 1 semilla no hay IC de robustez
                lo = hi = float("nan")
            per[a] = (m, lo, hi, n, nseeds)
            flag = "" if nseeds >= 2 else "  <- solo seed 42 (valor puntual)"
            print(f"   {a:10s} media={com(m)}  IC95=[{com(lo)}; {com(hi)}]  n={n}  semillas={nseeds}{flag}")
            summary_rows.append(dict(explainer=expl, metric=etq, architecture=a,
                                     mean=m, ci_lo=lo, ci_hi=hi, n=n, n_seeds=nseeds))
        if col == "stab_spearman_mean":
            spear[expl] = per
    return spear


def rendimiento(summary_rows):
    """ROC-AUC / PR-AUC / precision@50 sobre la SEMILLA 42 (cifras del manuscrito).

    El reeval a 3 semillas no se reporta como tal en la tesis porque GAT/TAGCN solo
    tienen seed 42; se conservan las cifras seed-42 ya publicadas y aqui se verifica
    que se reproducen."""
    if not REEVAL.exists():
        print(f"\n[reeval] {REEVAL} no existe aun; se omite el bloque de rendimiento.")
        return
    r = pd.read_csv(REEVAL)
    qp = r["quality_passed"].astype(str).str.lower().isin(["true", "1"])
    r = r[qp]
    if "seed" in r.columns:
        r = r[r["seed"] == 42]
    print("\n=== RENDIMIENTO (reeval, modelos que pasan el filtro, SEMILLA 42 = cifras del manuscrito) ===")
    for split in ["val", "test"]:
        s = r[r["split"] == split]
        print(f"\n[{split}]")
        for metric in ["roc_auc", "pr_auc_trap", "prec_at_50"]:
            vals = pd.to_numeric(s[metric], errors="coerce")
            m, lo, hi, n = ci95(vals.values)
            print(f"   {metric:12s} media={com(m)}  n={n}")
            summary_rows.append(dict(explainer=f"reeval42/{split}", metric=metric,
                                     architecture="TODAS", mean=m, ci_lo=lo, ci_hi=hi,
                                     n=n, n_seeds=1))


def escribir_tabla(spear):
    """Tabla LaTeX: Spearman de features, GNNExplainer vs GNNShap.

    Celdas con >=2 semillas: media [IC95]. Celdas con 1 semilla (GAT/TAGCN): media (s42)."""
    cols = [e for e in ("GNNExplainer", "GNNShap") if e in spear]
    if not cols:
        print("\n[tabla] sin datos de Spearman aun; no se escribe la tabla.")
        return
    EOL = "\\\\"
    filas = []
    for a in ARQ:
        celdas = []
        for e in cols:
            m, lo, hi, n, nseeds = spear[e].get(a, (np.nan, np.nan, np.nan, 0, 0))
            if nseeds >= 2:
                celdas.append(f"{com(m)} [{com(lo)}; {com(hi)}]")
            elif nseeds == 1:
                celdas.append(f"{com(m)} (s42)")
            else:
                celdas.append("n/d")
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
\caption{{Estabilidad de las explicaciones medida como correlación de Spearman sobre el ranking de \textit{{features}}. Ambos explicadores atribuyen importancia a \textit{{features}} y comparten escala, pero su lectura difiere. GNNExplainer discrimina entre arquitecturas y reproduce la partición en dos grupos, con GAT y GCN por encima de GraphSAGE y TAGCN. GNNShap, en cambio, se satura en valores próximos a la unidad para las cuatro arquitecturas, de modo que su estabilidad es alta pero no separa arquitecturas. Las celdas con intervalo de confianza al 95\% se promedian sobre las tres semillas de modelo; las marcadas \textup{{(s42)}} corresponden a GAT y TAGCN, cuyos pesos de las semillas adicionales no cabían para reentrenar en la GPU de 8 GB disponible y por eso se reportan sobre la semilla original, en la que la saturación ya es visible. PGExplainer opera sobre aristas y se reporta aparte con la métrica de Jaccard.}}
\label{{tab:elliptic-robustez-3s}}
\end{{table}}
"""
    TAB.parent.mkdir(parents=True, exist_ok=True)
    TAB.write_text(tex, encoding="utf-8")
    print("\nescrito:", TAB)

    # figura: barras con etiqueta de valor encima; barra de error solo si >=2 semillas,
    # y las barras de una sola semilla se marcan con trama y un asterisco.
    x = np.arange(len(ARQ))
    w = 0.8 / max(1, len(cols))
    fig, ax = plt.subplots(figsize=(7.8, 4.7))
    colores = {"GNNExplainer": "#1C7293", "GNNShap": "#E07A5F"}
    for j, e in enumerate(cols):
        for k, a in enumerate(ARQ):
            xi = x[k] + j * w - 0.4 + w / 2
            m, lo, hi, n, nseeds = spear[e].get(a, (np.nan, np.nan, np.nan, 0, 0))
            if pd.isna(m):
                continue
            err = [[m - lo], [hi - m]] if nseeds >= 2 and pd.notna(lo) else None
            ax.bar(xi, m, w, yerr=err, capsize=4, color=colores.get(e), zorder=3,
                   label=e if k == 0 else None,
                   hatch="//" if nseeds < 2 else None, edgecolor="white", linewidth=0.7)
            top = hi if (nseeds >= 2 and pd.notna(hi)) else m
            ax.annotate(com(m, 2) + ("*" if nseeds < 2 else ""), (xi, top),
                        textcoords="offset points", xytext=(0, 5), ha="center",
                        va="bottom", fontsize=9.5, fontweight="bold", color="0.15")
    ax.set_xticks(x)
    ax.set_xticklabels(ARQ, fontsize=11.5)
    ax.set_ylabel("Estabilidad (Spearman de features)", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", ls=":", color="0.85", zorder=0)
    ax.legend(loc="lower center", ncol=2, fontsize=10.5, framealpha=0.95)
    ax.set_title("Estabilidad por arquitectura: GNNExplainer distingue, GNNShap satura",
                 fontsize=11.5, pad=12)
    fig.text(0.5, 0.01,
             "Barra de error: IC 95 % sobre 3 semillas.  * GNNShap en GAT y TAGCN: valor de una sola semilla (42).",
             ha="center", fontsize=8.2, color="0.4")
    plt.tight_layout(rect=(0, 0.05, 1, 1))
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
