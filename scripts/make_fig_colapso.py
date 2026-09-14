# -*- coding: utf-8 -*-
"""Figura de colapso validacion->test para la lamina 18.
Barras agrupadas: cada metrica con su valor en validacion y en test.
Muestra que PR-AUC y precision@50 se desploman mientras ROC-AUC cae mucho menos
(=> ROC-AUC enganoso bajo desbalance). Valores = manuscrito (tabla val/test)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
# La raiz del repo se autodetecta; se puede forzar con GNN_REPO_ROOT.
REPO = Path(os.environ.get("GNN_REPO_ROOT", Path(__file__).resolve().parents[1]))
OUT = str(REPO / "presentacion_latex" / "fig")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "0.3"})

# orden: primarias primero, ROC-AUC al final (enganosa)
labels = ["PR-AUC", "Precision@50", "ROC-AUC"]
val = [0.367, 0.657, 0.884]
test = [0.017, 0.020, 0.653]

def com(v): return f"{v:.3f}".replace(".", ",")

fig, ax = plt.subplots(figsize=(6.6, 4.0))
xs = np.arange(len(labels)); w = 0.38
b1 = ax.bar(xs - w/2, val, w, color="#1C7293", edgecolor="black", linewidth=0.5,
            label="Validación", zorder=3)
b2 = ax.bar(xs + w/2, test, w, color="#C0392B", edgecolor="black", linewidth=0.5,
            label="Test (periodo posterior)", zorder=3)

for i in range(len(labels)):
    ax.text(xs[i] - w/2, val[i] + 0.02, com(val[i]), ha="center", fontsize=8.8)
    ax.text(xs[i] + w/2, test[i] + 0.02, com(test[i]), ha="center", fontsize=8.8,
            fontweight="bold", color="#C0392B")

# sombra + anotacion sobre ROC-AUC (la enganosa), en el hueco entre grupos
ax.axvspan(xs[2] - 0.5, xs[2] + 0.5, color="#F4B41A", alpha=0.10, zorder=0)
ax.annotate("ROC-AUC cae mucho menos\n(engañoso bajo desbalance)",
            xy=(xs[2] + w/2, test[2] + 0.01), xytext=(1.13, 0.80),
            ha="center", va="center", fontsize=8.2, color="0.20",
            arrowprops=dict(arrowstyle="->", color="0.45", lw=1))

# llave del colapso en las primarias
ax.annotate("", xy=(xs[0] + w/2, test[0]), xytext=(xs[0] - w/2, val[0]),
            arrowprops=dict(arrowstyle="->", color="0.5", lw=1.1, ls="--"))
ax.text(xs[0] + 0.02, 0.20, "colapso", rotation=90, va="center",
        ha="left", fontsize=8.3, color="0.4")

ax.set_xticks(xs); ax.set_xticklabels(labels)
ax.set_ylabel("Valor de la métrica")
ax.set_ylim(0, 1.05)
ax.grid(axis="y", ls=":", color="0.85", zorder=0)
ax.legend(fontsize=8.4, loc="upper left", framealpha=0.95, borderaxespad=0.6)
ax.set_title("Rendimiento validación vs. test (mismos modelos)", fontsize=10, pad=10)
plt.tight_layout()
p = os.path.join(OUT, "colapso_val_test.png")
plt.savefig(p, dpi=300); plt.close()
print("guardado:", p)
