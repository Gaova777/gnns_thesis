"""
gen_tables.py — Regenera tablas LaTeX del manuscrito desde los CSV de resultados (sin GPU).

Fuente única de verdad: phase1/results_robust_agg.csv (definición UNIFORME de todas las métricas,
la misma que usa el texto del Capítulo 5). Solo dependencias de la librería estándar (csv, statistics)
para que corra en cualquier máquina sin torch/pandas.

Tablas soportadas:
  - synth_factorial (tab:synth-factorial) → tesis_latex/tables/synth_factorial.tex
    Cierra R3: la versión previa mostraba Fidelity+ SOLO para GNNExplainer con la definición vieja
    (0,95/1,00/0,96/0,93, PyG built-in single-seed), contradiciendo el texto y synth_robust_full.tex.
    Ahora las 4 columnas salen de results_robust_agg.csv (media sobre las 3 model_seeds de g0).

Uso:
  python phase1/gen_tables.py                         # regenera synth_factorial.tex
  python phase1/gen_tables.py --dry-run               # imprime la tabla sin escribir
  python phase1/gen_tables.py --agg otra.csv --out otra.tex
"""
import csv
import argparse
import statistics as st

ARCHS = ["GraphSAGE", "GAT", "GCN", "TAGCN"]
EXPLAINERS = ["GNNExplainer", "PGExplainer", "GNNShap"]
# columnas de la tabla → (encabezado LaTeX, columna del CSV)
COLUMNS = [
    ("Estab.",          "spearman_mean"),
    ("Plaus. aristas",  "plaus_edge_mean"),
    ("Plaus. feat.",    "plaus_feat_mean"),
    ("Fidelity+",       "fid_plus_mean"),
]


def _num(x):
    try:
        v = float(x)
        return v if v == v else None   # descarta NaN
    except (TypeError, ValueError):
        return None


def load_g0(path):
    """Filas del grafo principal (graph_seed=0), que es la matriz factorial completa."""
    rows = list(csv.DictReader(open(path)))
    return [r for r in rows if r.get("graph_seed") == "0"]


def cell_mean(rows, arch, explainer, col):
    """Media de `col` sobre model_seeds para (arch, explainer). None si no hay dato (métrica n/d)."""
    vals = [_num(r[col]) for r in rows
            if r["architecture"] == arch and r["explainer"] == explainer]
    vals = [v for v in vals if v is not None]
    return round(st.mean(vals), 2) if vals else None


def fmt(v):
    """Formato español: coma decimal, 2 decimales; n/d si no aplica."""
    return "n/d" if v is None else f"{v:.2f}".replace(".", ",")


def build_synth_factorial(rows):
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\renewcommand{\arraystretch}{1.3}",
        r"\small",
        r"\begin{tabular}{ll cccc}",
        r"\toprule",
        r"\textbf{Arquitectura} & \textbf{Explicador} & \textbf{Estab.} & "
        r"\textbf{Plaus. aristas} & \textbf{Plaus. feat.} & \textbf{Fidelity+} \\",
        r"\midrule",
    ]
    for i, arch in enumerate(ARCHS):
        for explainer in EXPLAINERS:
            cells = [fmt(cell_mean(rows, arch, explainer, col)) for _, col in COLUMNS]
            lines.append(f"{arch} & {explainer} & " + " & ".join(cells) + r" \\")
        if i < len(ARCHS) - 1:
            lines.append(r"\midrule")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Resultados de la matriz factorial sintetica por arquitectura y explicador "
        r"(media sobre 3 semillas de modelo, grafo g0). Los valores n/d corresponden a metricas que "
        r"un explicador no produce por diseno. Fidelity+ se calcula con la definicion manual uniforme "
        r"(caida de probabilidad al retirar el top-k) para los tres explicadores. PGExplainer domina "
        r"la plausibilidad de aristas, GNNExplainer la de features y la fidelidad, y GNNShap la estabilidad.}",
        r"\label{tab:synth-factorial}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", default="phase1/results_robust_agg.csv")
    ap.add_argument("--out", default="tesis_latex/tables/synth_factorial.tex")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    rows = load_g0(a.agg)
    tex = build_synth_factorial(rows)

    print(f"Fuente: {a.agg}  ({len(rows)} filas g0)")
    print("Fidelity+ por (arquitectura, explicador) — definicion uniforme:")
    for arch in ARCHS:
        parts = [f"{e[:4]}={fmt(cell_mean(rows, arch, e, 'fid_plus_mean'))}" for e in EXPLAINERS]
        print(f"  {arch:10} " + "  ".join(parts))
    print()
    print(tex)

    if a.dry_run:
        print("[dry-run] no se escribio nada.")
        return
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write(tex)
    print(f"-> escrito {a.out}")


if __name__ == "__main__":
    main()
