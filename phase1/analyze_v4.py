"""Análisis del eje sintético v4: cada métrica contra SU línea base aleatoria.

Lee results_phase1_v4/results_v4.csv (run_phase1_v4.py) y, por arquitectura × explicador,
reporta media real, media azar y ganancia (real − azar) de plaus_edge, plaus_feat, fid+ y fid−,
con Wilcoxon pareado real vs azar sobre las celdas (arch, escenario, balanceo, semilla).
Lectura de fid−: p0 − p(solo top-k); más cerca de 0 (o negativa) = más suficiente. Por eso la
ganancia «buena» en fid− es NEGATIVA (real < azar), y en fid+ y plausibilidad es POSITIVA.
Filtra por alineamiento (--aligned-only) usando la columna aligned (alignment_check.py).
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd
from scipy import stats

PAIRS = [("plaus_edge_mean", "plaus_edge_rand", +1), ("plaus_feat_mean", "plaus_feat_rand", +1),
         ("fid_plus_mean", "fid_plus_rand", +1), ("fid_minus_mean", "fid_minus_rand", -1)]


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (arch, ex), g in df.groupby(["architecture", "explainer"]):
        r = {"architecture": arch, "explainer": ex, "n_cells": len(g),
             "val_pr_auc": round(g["val_pr_auc"].astype(float).mean(), 3),
             "spearman": round(pd.to_numeric(g["spearman_mean"], errors="coerce").mean(), 3)}
        for real, rand, sign in PAIRS:
            x = pd.to_numeric(g[real], errors="coerce"); y = pd.to_numeric(g[rand], errors="coerce")
            ok = x.notna() & y.notna()
            name = real.replace("_mean", "")
            if ok.sum() == 0:
                continue
            r[name] = round(x[ok].mean(), 3); r[name + "_rand"] = round(y[ok].mean(), 3)
            r[name + "_gain"] = round((x[ok] - y[ok]).mean(), 3)
            p = float("nan")
            if ok.sum() >= 6 and (x[ok] - y[ok]).abs().sum() > 0:
                alt = "greater" if sign > 0 else "less"
                p = stats.wilcoxon(x[ok], y[ok], alternative=alt).pvalue
            r[name + "_p"] = p if math.isnan(p) else float(f"{p:.2g}")
        rows.append(r)
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="results_phase1_v4/results_v4.csv")
    ap.add_argument("--out", default="results_phase1_v4/summary_v4.csv")
    ap.add_argument("--aligned-only", action="store_true")
    a = ap.parse_args(argv)
    df = pd.read_csv(a.csv)
    df = df[df["status"] == "ok"]
    if a.aligned_only and "aligned" in df:
        df = df[df["aligned"].astype(str).str.lower() == "true"]
    s = summarize(df)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)
    print(s.to_string(index=False))
    s.to_csv(a.out, index=False)
    print(f"escrito {a.out}")


if __name__ == "__main__":
    main()
