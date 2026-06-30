"""Add native scenario figures to the thesis_figures package.

The main notebook filters scenarios to [1:1, 1:10, 1:50, 1:100] so native results
are not included. This script generates 4 additional figures specifically for the
1:30_native scenario validation.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csv
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

OUT_DIR = Path("/tmp/thesis-web-learning/thesis_figures")
RESULTS_DIR = OUT_DIR / "results"
ANALYSIS_DIR = OUT_DIR / "analysis"
DISCUSSION_DIR = OUT_DIR / "discussion"

ARCH_COLORS = {"GCN": "#6366f1", "GraphSAGE": "#f59e0b", "GAT": "#10b981", "TAGCN": "#8b5cf6"}
EX_COLORS = {"GNNExplainer": "#10b981", "PGExplainer": "#ef4444", "GNNShap": "#f59e0b"}

plt.style.use("seaborn-v0_8-whitegrid")

# Load data
metas = {}
for p in Path("results_models_v3").glob("*_meta.json"):
    d = json.load(open(p))
    metas[d["run_id"]] = d

rows = []
for p in Path("results_v3").glob("*.csv"):
    try:
        rows.extend(list(csv.DictReader(open(p))))
    except: pass
real = [r for r in rows if r.get("explainer") not in ("SKIPPED_QUALITY_GATE", "SKIPPED")]
# Dedupe
seen = set(); deduped = []
for r in real:
    k = (r["scenario"], r["architecture"], r["balancing"], r["explainer"])
    if k not in seen:
        seen.add(k); deduped.append(r)

# Native-only data
native = [r for r in deduped if r["scenario"] == "1:30_native"]
print(f"Native explainer rows: {len(native)}")

# =========================
# R7 — Native scenario pass rate by arch
# =========================
native_metas = {k: v for k, v in metas.items() if v["scenario"] == "1:30_native"}
arch_counts = defaultdict(lambda: {"pass": 0, "total": 0})
for m in native_metas.values():
    arch_counts[m["architecture"]]["total"] += 1
    if m["quality_passed"]: arch_counts[m["architecture"]]["pass"] += 1

archs = ["GCN", "GraphSAGE", "GAT", "TAGCN"]
pass_counts = [arch_counts[a]["pass"] for a in archs]
totals = [arch_counts[a]["total"] for a in archs]

fig, ax = plt.subplots(figsize=(8, 5))
colors = [ARCH_COLORS[a] for a in archs]
bars = ax.bar(archs, pass_counts, color=colors, edgecolor='black', linewidth=1)
for b, p, t in zip(bars, pass_counts, totals):
    pct = 100 * p / t if t > 0 else 0
    ax.annotate(f'{p}/{t}\n({pct:.0f}%)', (b.get_x() + b.get_width()/2, b.get_height()),
                ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylabel('Configs passing val gate', fontsize=12)
ax.set_title('R7 — Native 1:30 pass rate by architecture\n(replicates Weber 2019 baseline)', fontsize=13, fontweight='bold')
ax.set_ylim(0, max(totals) + 1)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "R7_native_pass_rate_by_arch.png", dpi=300, bbox_inches='tight')
plt.close()
pd.DataFrame({"arch": archs, "pass": pass_counts, "total": totals}).to_csv(
    RESULTS_DIR / "R7_native_pass_rate_by_arch.data.csv", index=False)
print("Saved R7")

# =========================
# A11 — Spearman comparison: Native vs Forced scenarios
# =========================
by_scenario = defaultdict(list)
for r in deduped:
    if r["explainer"] != "GNNExplainer": continue
    try:
        by_scenario[r["scenario"]].append(float(r["stab_spearman_mean"]))
    except: pass

# Order: 1:1 → 1:10 → 1:30_native → 1:50 → 1:100
order = ["1:1", "1:10", "1:30_native", "1:50", "1:100"]
means = [np.mean(by_scenario[s]) if by_scenario[s] else np.nan for s in order]
counts = [len(by_scenario[s]) for s in order]
stds = [np.std(by_scenario[s]) if len(by_scenario[s]) > 1 else 0 for s in order]

fig, ax = plt.subplots(figsize=(10, 6))
x_pos = np.arange(len(order))
bar_colors = ["#6366f1", "#10b981", "#fbbf24", "#059669", "#ef4444"]
bars = ax.bar(x_pos, means, color=bar_colors, edgecolor='black', linewidth=1, width=0.65)
# error bars
for i, (m, s, n) in enumerate(zip(means, stds, counts)):
    if not np.isnan(m):
        ax.errorbar(i, m, yerr=s, fmt='none', ecolor='black', capsize=6, alpha=0.6)
        ax.annotate(f'{m:.3f}\n(n={n})', (i, m), xytext=(0, 8), textcoords='offset points',
                    ha='center', fontsize=10, fontweight='bold')

ax.set_xticks(x_pos)
ax.set_xticklabels(order, fontsize=11)
ax.set_ylabel('GNNExplainer Spearman', fontsize=12)
ax.set_title('A11 — GNNExplainer stability: native 1:30 (Weber baseline) vs controlled scenarios\n'
             'Native scenario has MOST training data but LESS varied XAI patterns',
             fontsize=12, fontweight='bold')
ax.axvline(x=2, color='gray', linestyle='--', alpha=0.4)
ax.annotate('Native\n(Weber 2019)', (2, max(m for m in means if not np.isnan(m)) * 1.08),
            ha='center', fontsize=9, style='italic', color='#d97706')
ax.set_ylim(0, 1.0)
plt.tight_layout()
plt.savefig(ANALYSIS_DIR / "A11_spearman_native_vs_forced.png", dpi=300, bbox_inches='tight')
plt.close()
pd.DataFrame({"scenario": order, "mean_spearman": means, "std": stds, "n": counts}).to_csv(
    ANALYSIS_DIR / "A11_spearman_native_vs_forced.data.csv", index=False)
print("Saved A11")

# =========================
# A12 — Native scenario arch × explainer heatmap
# =========================
data_matrix = defaultdict(dict)
for r in native:
    key = (r["architecture"], r["explainer"])
    try:
        data_matrix[r["architecture"]][r["explainer"]] = float(r["stab_spearman_mean"])
    except: pass

archs_present = sorted(set(r["architecture"] for r in native))
explainers = ["GNNExplainer", "PGExplainer", "GNNShap"]
heatmap = np.full((len(archs_present), len(explainers)), np.nan)
for i, a in enumerate(archs_present):
    for j, e in enumerate(explainers):
        heatmap[i, j] = data_matrix[a].get(e, np.nan)

fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(heatmap, cmap='RdYlGn', vmin=0, vmax=0.4, aspect='auto')
ax.set_xticks(range(len(explainers)))
ax.set_xticklabels(explainers, fontsize=11)
ax.set_yticks(range(len(archs_present)))
ax.set_yticklabels(archs_present, fontsize=11)
for i in range(len(archs_present)):
    for j in range(len(explainers)):
        val = heatmap[i, j]
        if not np.isnan(val):
            color = "white" if val < 0.15 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=color, fontweight='bold')
        else:
            ax.text(j, i, "—", ha="center", va="center", color="gray")
ax.set_title('A12 — Native 1:30 stability heatmap (Spearman)\n'
             'Averaged across passing configs per (architecture × explainer)',
             fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax, label='Spearman')
plt.tight_layout()
plt.savefig(ANALYSIS_DIR / "A12_native_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()
df = pd.DataFrame(heatmap, index=archs_present, columns=explainers)
df.to_csv(ANALYSIS_DIR / "A12_native_heatmap.data.csv")
print("Saved A12")

# =========================
# D6 — F1 comparison native vs literature
# =========================
# Get native F1 by arch
native_f1 = {a: [m["val_f1_best_epoch"] for m in native_metas.values() if m["architecture"]==a]
             for a in archs}

# Literature baselines
lit_data = [
    ("Weber 2019 GCN", 0.41, 0.628, "test / native"),
    ("Weber 2019 RF", 0.79, 0.79, "test / native"),
    ("Pareja 2020 EvolveGCN", 0.89, 0.89, "test / temporal"),
    ("Bellei 2024 Elliptic2", 0.93, 0.93, "test / subgraph"),
    ("arXiv:2602 GraphSAGE+norm", 0.85, 0.85, "val / native"),
]
our_data = [
    (f"Ours GCN native", np.nan, max(native_f1["GCN"]) if native_f1["GCN"] else 0, "val / native"),
    (f"Ours GraphSAGE native", np.nan, max(native_f1["GraphSAGE"]) if native_f1["GraphSAGE"] else 0, "val / native"),
    (f"Ours GAT native", np.nan, max(native_f1["GAT"]) if native_f1["GAT"] else 0, "val / native"),
    (f"Ours TAGCN native", np.nan, max(native_f1["TAGCN"]) if native_f1["TAGCN"] else 0, "val / native"),
]

fig, ax = plt.subplots(figsize=(12, 7))
all_data = lit_data + our_data
labels = [x[0] for x in all_data]
vals = [x[2] for x in all_data]
colors_bars = ['#6b7280'] * len(lit_data) + ['#059669'] * len(our_data)
bars = ax.barh(range(len(labels)), vals, color=colors_bars, edgecolor='black')
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel('F1 illicit', fontsize=12)
ax.set_title('D6 — Native 1:30 vs literature SOTA on Elliptic\n'
             '(Green = our results; Gray = literature)',
             fontsize=12, fontweight='bold')
for i, v in enumerate(vals):
    ax.annotate(f'{v:.3f}', (v + 0.01, i), va='center', fontsize=9)
ax.axvline(0.70, color='green', linestyle='--', alpha=0.5, label='Competitive threshold')
ax.legend(loc='lower right')
ax.set_xlim(0, 1.0)
plt.tight_layout()
plt.savefig(DISCUSSION_DIR / "D6_native_vs_literature.png", dpi=300, bbox_inches='tight')
plt.close()
pd.DataFrame({"source": labels, "val_or_test_F1": vals, "type": [x[3] for x in all_data]}).to_csv(
    DISCUSSION_DIR / "D6_native_vs_literature.data.csv", index=False)
print("Saved D6")

print("\nAll native figures added:")
for f in ["R7_native_pass_rate_by_arch", "A11_spearman_native_vs_forced",
          "A12_native_heatmap", "D6_native_vs_literature"]:
    print(f"  - {f}.png + .data.csv")
