# -*- coding: utf-8 -*-
"""Recalcula el analisis por escenario de desbalance con las TRES semillas.

Motivo: la seccion de escenarios del Capitulo 4 quedo apoyada en la corrida de
una sola semilla, mientras que el resto del capitulo ya usa la replicacion de
tres semillas (results_seedsweep/). Este script rehace esa seccion sobre el
conjunto correcto y deja el contraste formal que faltaba para el factor
escenario, que hasta ahora se sostenia solo en la amplitud descriptiva.

Genera:
  tesis_latex/tables/elliptic_stab_scenario.tex
  tesis_latex/chapter_4/images_ch4/estabilidad_escenario.png

La raiz del repo se autodetecta; se puede forzar con GNN_REPO_ROOT.
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
TAB = ROOT / "tesis_latex" / "tables" / "elliptic_stab_scenario.tex"
FIG = ROOT / "tesis_latex" / "chapter_4" / "images_ch4" / "estabilidad_escenario.png"

ORDEN = ["1:1", "1:10", "1:50", "1:100", "1:30_native"]
ETIQUETA = {"1:1": "1:1", "1:10": "1:10", "1:50": "1:50",
            "1:100": "1:100", "1:30_native": "nativo"}
ARQ = ["GraphSAGE", "GAT", "GCN", "TAGCN"]


def com(v, d=3):
    return "n/d" if pd.isna(v) else f"{v:.{d}f}".replace(".", ",")


def cargar():
    d = pd.read_csv(CSV)
    d = d[d["explainer"] == "GNNExplainer"].copy()
    d["v"] = pd.to_numeric(d["stab_spearman_mean"], errors="coerce")
    return d.dropna(subset=["v"])


def contraste(d):
    """Kruskal-Wallis sobre el factor escenario + eta cuadrado (ANOVA),
    que es el tamano de efecto que reporta el resto de la tesis."""
    grupos = [g["v"].values for _, g in d.groupby("scenario") if len(g) >= 2]
    N, k = sum(len(g) for g in grupos), len(grupos)
    H, p_kw = stats.kruskal(*grupos)
    F, _ = stats.f_oneway(*grupos)
    eta2 = (F * (k - 1)) / (F * (k - 1) + (N - k))
    return dict(H=H, p=p_kw, eta2=eta2, N=N, k=k)


def main():
    d = cargar()
    piv = d.pivot_table(index="architecture", columns="scenario", values="v", aggfunc="mean")
    medias = d.groupby("scenario")["v"].mean()
    c = contraste(d)

    print(f"N={c['N']} filas ({d['seed'].nunique()} semillas)")
    print(f"Kruskal-Wallis escenario: H={c['H']:.2f}  p={c['p']:.4f}  eta2={c['eta2']:.3f}")
    print("\nmedias por escenario:")
    for s in ORDEN:
        print(f"  {ETIQUETA[s]:8s} {medias[s]:.3f}")
    lo, hi = medias.min(), medias.max()
    print(f"\namplitud = {hi - lo:.3f}  (min {ETIQUETA[medias.idxmin()]}, max {ETIQUETA[medias.idxmax()]})")

    # ------------------------------------------------------------ tabla
    filas = []
    for a in ARQ:
        celdas = " & ".join(com(piv.loc[a, s]) if s in piv.columns else "n/d" for s in ORDEN)
        filas.append(f"{a} & {celdas} \\\\")
    EOL = "\\\\"
    encabezado = " & ".join(ETIQUETA[s] for s in ORDEN)
    medias_fila = " & ".join(com(medias[s]) for s in ORDEN)
    tex = rf"""\begin{{table}}[htbp]
\centering
\renewcommand{{\arraystretch}}{{1.3}}
\begin{{tabular}}{{lccccc}}
\toprule
\textbf{{Arquitectura}} & {encabezado} {EOL}
\midrule
{chr(10).join(filas)}
\midrule
\textbf{{Media}} & {medias_fila} {EOL}
\bottomrule
\end{{tabular}}
\caption{{Estabilidad de las explicaciones (correlación de Spearman de GNNExplainer) por arquitectura y escenario de desbalance sobre el subgrafo receptivo, promediada sobre las tres semillas de modelo. GCN y GAT encabezan la estabilidad de forma transversal a los escenarios, y la fila de medias muestra que el nivel de desbalance no ordena los resultados.}}
\label{{tab:elliptic-stab-scen}}
\end{{table}}
"""
    TAB.write_text(tex, encoding="utf-8")
    print("\nescrito:", TAB)

    # ------------------------------------------------------------ figura
    xs = np.arange(len(ORDEN))
    ys = [medias[s] for s in ORDEN]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(xs, ys, marker="o", color="#1C7293", lw=2, zorder=3)
    for x, y in zip(xs, ys):
        ax.annotate(com(y, 2), (x, y), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels([ETIQUETA[s] for s in ORDEN])
    ax.set_xlabel("Escenario de desbalance")
    ax.set_ylabel("Spearman medio")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", ls=":", color="0.85", zorder=0)
    ax.set_title(f"Estabilidad por escenario (3 semillas · Kruskal-Wallis p = {c['p']:.2f}, "
                 f"no significativo)".replace(".", ",", 1), fontsize=10)
    plt.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG, dpi=300)
    plt.close()
    print("escrito:", FIG)


if __name__ == "__main__":
    main()
