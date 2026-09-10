# -*- coding: utf-8 -*-
"""Grafica las curvas PR y ROC (validacion vs test) para la lamina de rendimiento.

Consume dos CSV producidos por scripts/consolidacion/reeval_rocauc.py:
  - results_v3/reeval_curves.csv   (puntos de curva; requiere --dump-curves)
  - results_v3/reeval_metrics.csv  (AUC medios; para las etiquetas)

Por defecto promedia las curvas sobre los modelos con quality_passed=True (los
23 que aprendieron), que son los que producen los AUC reportados en el
manuscrito (val: ROC 0,884 / PR 0,367; test: ROC 0,653 / PR 0,017).

Salida: presentacion_latex/fig/curvas_pr_roc.png

Uso (tras correr reeval con --dump-curves y traer los CSV):
    python scripts/make_fig_curves.py
    python scripts/make_fig_curves.py --all          # incluye los 60 modelos
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.environ.get("GNN_REPO_ROOT", r"E:\pryectos\thesisgnn\gnns_thesis")
CURVES = os.path.join(REPO, "results_v3", "reeval_curves.csv")
METRICS = os.path.join(REPO, "results_v3", "reeval_metrics.csv")
OUT = os.path.join(REPO, "presentacion_latex", "fig", "curvas_pr_roc.png")

COL_VAL = "#1C7293"
COL_TEST = "#C0392B"


def com(v):
    return f"{v:.3f}".replace(".", ",")


def mean_curve(df, split, ycol):
    """Curva media punto a punto sobre configuraciones, para un split."""
    s = df[df["split"] == split]
    g = s.groupby("grid_x")[ycol].mean().reset_index().sort_values("grid_x")
    return g["grid_x"].to_numpy(float), g[ycol].to_numpy(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="Usar los 60 modelos (por defecto solo quality_passed=True).")
    ap.add_argument("--curves", default=CURVES)
    ap.add_argument("--metrics", default=METRICS)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    curves = pd.read_csv(args.curves)
    metrics = pd.read_csv(args.metrics)
    if not args.all:
        curves = curves[curves["quality_passed"] == True]  # noqa: E712
        metrics = metrics[metrics["quality_passed"] == True]  # noqa: E712

    # AUC medios (autoritativos, del CSV de metricas) para las etiquetas.
    def auc_label(split, col):
        return metrics.loc[metrics["split"] == split, col].mean()

    roc_val = auc_label("val", "roc_auc"); roc_test = auc_label("test", "roc_auc")
    pr_val = auc_label("val", "pr_auc_ap"); pr_test = auc_label("test", "pr_auc_ap")

    fig, (axr, axp) = plt.subplots(1, 2, figsize=(9.6, 4.3))

    # ------------------------- ROC (izquierda) -------------------------
    for split, col, ls, auc_v in (("val", COL_VAL, "-", roc_val),
                                  ("test", COL_TEST, "--", roc_test)):
        x, y = mean_curve(curves, split, "roc_tpr")
        lab = ("Validación" if split == "val" else "Test") + f" (AUC={com(auc_v)})"
        axr.plot(x, y, color=col, ls=ls, lw=2.2, label=lab, zorder=3)
    axr.plot([0, 1], [0, 1], color="0.6", ls=":", lw=1.2, label="azar", zorder=2)
    axr.set_xlabel("Tasa de falsos positivos"); axr.set_ylabel("Tasa de verdaderos positivos")
    axr.set_title("Curva ROC", fontsize=11)
    axr.set_xlim(0, 1); axr.set_ylim(0, 1.02)
    axr.grid(ls=":", color="0.85", zorder=0); axr.legend(fontsize=8.6, loc="lower right")

    # ------------------------- PR (derecha) -------------------------
    for split, col, ls, auc_v in (("val", COL_VAL, "-", pr_val),
                                  ("test", COL_TEST, "--", pr_test)):
        x, y = mean_curve(curves, split, "pr_precision")
        lab = ("Validación" if split == "val" else "Test") + f" (AUC={com(auc_v)})"
        axp.plot(x, y, color=col, ls=ls, lw=2.2, label=lab, zorder=3)
    axp.set_xlabel("Recall (cobertura)"); axp.set_ylabel("Precisión")
    axp.set_title("Curva Precisión–Recall", fontsize=11)
    axp.set_xlim(0, 1); axp.set_ylim(0, 1.02)
    axp.grid(ls=":", color="0.85", zorder=0); axp.legend(fontsize=8.6, loc="upper right")

    fig.suptitle("El ROC-AUC engaña bajo desbalance: la curva PR revela la dificultad real",
                 fontsize=11, y=1.00)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    print("guardado:", args.out)


if __name__ == "__main__":
    main()
