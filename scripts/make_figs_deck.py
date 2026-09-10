# -*- coding: utf-8 -*-
"""Regenera 2 figuras del deck con barras de error (IC 95%) y linea de azar.

Barras = valores publicados del manuscrito (consistencia figura<->texto);
IC = calculados desde los CSV versionados. Salida: presentacion_latex/fig/.

Nota sobre la figura de disociacion: solo se grafican GNNExplainer y
PGExplainer, que son los dos explicadores que producen mascara de aristas y
por tanto los unicos comparables en plausibilidad de aristas contra la misma
linea base de azar (0,403). GNNShap no produce mascara de aristas por diseno;
su plausibilidad es de features, vive en otra escala y tiene su propia linea
base (0,075), asi que mezclarla en la misma figura la haria ilegible. Sus
valores van en la lamina de respaldo, etiquetados con su metrica.

La raiz del repo se autodetecta; se puede forzar con GNN_REPO_ROOT.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(os.environ.get("GNN_REPO_ROOT", Path(__file__).resolve().parents[1]))
OUT = REPO / "presentacion_latex" / "fig"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "0.3"})

AZAR_ARISTAS = 0.403
AZAR_FEATURES = 0.075


def boot_ci(x, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(pd.to_numeric(pd.Series(x), errors="coerce").dropna(), float)
    if len(x) == 0: return float("nan"), float("nan"), float("nan"), 0
    if len(x) < 2:  return float(x.mean()), float(x.mean()), float(x.mean()), len(x)
    means = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(n)])
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), len(x)


def com(v): return f"{v:.3f}".replace(".", ",")
def com2(v): return f"{v:.2f}".replace(".", ",")

# ============================== FIGURA: RANKING ==============================
# medias e IC publicados (manuscrito, tab:ic) - verificados por bootstrap sobre el seedsweep
order = ["GAT", "GCN", "GraphSAGE", "TAGCN"]
grupo = {"GAT": "alto", "GCN": "alto", "GraphSAGE": "bajo", "TAGCN": "bajo"}
mean_pub = {"GAT": 0.782, "GCN": 0.758, "GraphSAGE": 0.735, "TAGCN": 0.672}
ci_pub = {"GAT": (0.758, 0.805), "GCN": (0.724, 0.787), "GraphSAGE": (0.710, 0.756), "TAGCN": (0.632, 0.717)}
df = pd.read_csv(REPO / "results_seedsweep" / "xai-gnn-stability-seedsweep.csv")
df = df[df["explainer"] == "GNNExplainer"]
print("=== RANKING: verificacion (datos vs publicado) ===")
for a in order:
    m, lo, hi, nn = boot_ci(df.loc[df["architecture"] == a, "stab_spearman_mean"])
    print(f"  {a:10s} datos: media={com(m)} IC=[{com(lo)};{com(hi)}] (n={nn}) | publicado: {com(mean_pub[a])} [{com(ci_pub[a][0])};{com(ci_pub[a][1])}]")

fig, ax = plt.subplots(figsize=(5.6, 3.7))
xs = np.arange(len(order))
means = [mean_pub[a] for a in order]
elo = [mean_pub[a] - ci_pub[a][0] for a in order]
ehi = [ci_pub[a][1] - mean_pub[a] for a in order]
cols = ["#1C7293" if grupo[a] == "alto" else "#9AA7B4" for a in order]
ax.bar(xs, means, yerr=[elo, ehi], capsize=5, color=cols, edgecolor="black", linewidth=0.6, zorder=3)
for i, a in enumerate(order):
    ax.text(i, ci_pub[a][1] + 0.015, com(means[i]), ha="center", fontsize=9)
ax.set_xticks(xs); ax.set_xticklabels(order)
ax.set_ylabel("Estabilidad (Spearman de features)")
ax.set_ylim(0, 1.0)
ax.axvline(1.5, color="0.75", ls=":", lw=1)
ax.text(0.5, 0.04, "grupo alto", ha="center", color="#1C7293", fontsize=8.5, transform=ax.get_xaxis_transform())
ax.text(2.5, 0.04, "grupo bajo", ha="center", color="#66707a", fontsize=8.5, transform=ax.get_xaxis_transform())
ax.grid(axis="y", ls=":", color="0.85", zorder=0)
ax.set_title("Estabilidad por arquitectura (3 semillas · IC 95%)", fontsize=10)
plt.tight_layout()
p1 = OUT / "ranking_khop_eb.png"
plt.savefig(p1, dpi=300); plt.close()
print("  -> guardado:", p1)

# ============================== FIGURA: DISOCIACION ==============================
r = pd.read_csv(REPO / "phase1" / "results_robust_agg.csv")
expl = ["GNNExplainer", "PGExplainer"]          # unicos con mascara de aristas
pub_plaus = {"GNNExplainer": 0.50, "PGExplainer": 0.80}
pub_fid = {"GNNExplainer": 0.56, "PGExplainer": 0.11}
print("\n=== DISOCIACION: verificacion (datos robustos vs publicado) ===")
plaus_e, fid_e = {}, {}
for e in expl:
    pm, plo, phi, pn = boot_ci(r.loc[r["explainer"] == e, "plaus_edge_mean"])
    fm, flo, fhi, fn = boot_ci(r.loc[r["explainer"] == e, "fid_plus_mean"])
    plaus_e[e] = (phi - plo) / 2 if pn >= 2 else 0.0
    fid_e[e] = (fhi - flo) / 2 if fn >= 2 else 0.0
    print(f"  {e:12s} plaus aristas datos={com(pm)} (n={pn}) pub={com2(pub_plaus[e])} | fid datos={com(fm)} (n={fn}) pub={com2(pub_fid[e])}")
# contexto de GNNShap, que va en la lamina de respaldo y no en esta figura
sm, slo, shi, sn = boot_ci(r.loc[r["explainer"] == "GNNShap", "plaus_feat_mean"])
sf, _, _, _ = boot_ci(r.loc[r["explainer"] == "GNNShap", "fid_plus_mean"])
print(f"  {'GNNShap':12s} plaus FEATURES datos={com(sm)} (n={sn}, azar {com(AZAR_FEATURES)}) | fid datos={com(sf)}")
print("               (fuera de la figura: metrica distinta, escala distinta, azar distinto)")

fig, ax = plt.subplots(figsize=(6.2, 3.9))
xs = np.arange(len(expl)); w = 0.32
gp = [pub_plaus[e] for e in expl]; gpe = [plaus_e[e] for e in expl]
fp = [pub_fid[e] for e in expl]; fpe = [fid_e[e] for e in expl]
ax.bar(xs - w/2, gp, w, yerr=gpe, capsize=4, color="#5CB85C", edgecolor="black", linewidth=0.5,
       label="Plausibilidad de aristas (vs ground-truth)", zorder=3)
ax.bar(xs + w/2, fp, w, yerr=fpe, capsize=4, color="#7A4FA3", edgecolor="black", linewidth=0.5,
       label="Fidelity+ (vs modelo)", zorder=3)
for i, e in enumerate(expl):
    ax.text(i - w/2, gp[i] + gpe[i] + 0.02, com2(gp[i]), ha="center", fontsize=9)
    ax.text(i + w/2, fp[i] + fpe[i] + 0.02, com2(fp[i]), ha="center", fontsize=9)
ax.axhline(AZAR_ARISTAS, color="0.30", ls="--", lw=1.2, zorder=2)
ax.text(len(expl) - 0.34, AZAR_ARISTAS + 0.025, "azar (aristas) = 0,40",
        ha="right", va="bottom", color="0.25", fontsize=8.5)
ax.set_xticks(xs); ax.set_xticklabels(expl)
ax.set_xlim(-0.7, len(expl) - 0.3)
ax.set_ylabel("Valor de la métrica")
ax.set_ylim(0, 1.15)
ax.grid(axis="y", ls=":", color="0.85", zorder=0)
ax.legend(fontsize=8.4, loc="upper left", ncol=1, framealpha=0.95)
ax.set_title("El más plausible no es el más fiel: el orden se invierte", fontsize=10, pad=8)
plt.tight_layout()
p2 = OUT / "disociacion_eb.png"
plt.savefig(p2, dpi=300); plt.close()
print("  -> guardado:", p2)
print("\nLISTO.")
