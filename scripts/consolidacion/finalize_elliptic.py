#!/usr/bin/env python3
"""Finalizacion Paso D: regenera tablas y figuras de estabilidad Elliptic con
el metrico R1 corregido, desde el CSV completo (23 passed x3 + 37 no-passed GNNExpl).

Todas las tablas de estabilidad Elliptic usan GNNExplainer Spearman/Jaccard.
Escribe: elliptic_full.tex, elliptic_stab_scenario.tex, elliptic_jaccard.tex,
y las figuras ranking_khop.png, estabilidad_escenario.png, contraste_regimen.png.
Imprime los valores para las ediciones de prosa (tab:ranking completa + por escenario).
"""
import csv, json, glob, os
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = "/home/juan/Escritorio/gnn_thesis/gnns_thesis"
TEX = f"{ROOT}/tesis_latex"
CSV = f"{ROOT}/results_v3/xai-gnn-stability-B-v3.csv"

def f(x):
    try: return float(x)
    except: return None

rows = list(csv.DictReader(open(CSV)))
ge = [r for r in rows if r["explainer"] == "GNNExplainer"]

# quality_passed set desde meta.json
metas = [json.load(open(p)) for p in glob.glob(f"{ROOT}/results_models_v3/*_meta.json")]
passed = {(m["architecture"], m["scenario"], m["balancing"]) for m in metas if m.get("quality_passed")}

SCEN_ORDER = ["1:1", "1:10", "1:50", "1:100", "1:30_native"]
SCEN_LABEL = {"1:1":"1:1","1:10":"1:10","1:50":"1:50","1:100":"1:100","1:30_native":"nativo"}
BAL_LABEL = {"class_weighting":"ponderacion","focal_loss":"focal loss","none":"ninguno"}
BAL_ORDER = ["class_weighting","focal_loss","none"]
ARCHS = ["GraphSAGE","GAT","GCN","TAGCN"]

# index GNNExplainer por (scen,arch,bal)
idx = {}
for r in ge:
    idx[(r["scenario"], r["architecture"], r["balancing"])] = r

def fmt(x, nd=3):
    return ("%.*f" % (nd, x)).replace(".", ",") if x is not None else "n/d"

# ---------- 1) elliptic_full.tex (60 filas) ----------
lines = [
 r"\begin{longtable}{lllccc}",
 r"\caption{Resultados completos por configuracion sobre el \textit{Elliptic Dataset} con la metrica de estabilidad corregida: estabilidad (Spearman de GNNExplainer), F1 en test y numero de verdaderos positivos de validacion empleados. Una fila por combinacion de escenario, arquitectura y balanceo.}\label{tab:elliptic-full}\\",
 r"\toprule",
 r"\textbf{Escenario} & \textbf{Arquitectura} & \textbf{Balanceo} & \textbf{Spearman} & \textbf{F1 test} & \textbf{n TP} \\",
 r"\midrule", r"\endfirsthead",
 r"\toprule",
 r"\textbf{Escenario} & \textbf{Arquitectura} & \textbf{Balanceo} & \textbf{Spearman} & \textbf{F1 test} & \textbf{n TP} \\",
 r"\midrule", r"\endhead",
 r"\bottomrule", r"\endfoot",
]
for scen in SCEN_ORDER:
    for arch in ["GraphSAGE","GAT","GCN","TAGCN"]:
        for bal in BAL_ORDER:
            r = idx.get((scen, arch, bal))
            sp = fmt(f(r["stab_spearman_mean"])) if r else "n/d"
            f1 = fmt(f(r["model_test_f1"]), 3) if r and f(r["model_test_f1"]) is not None else "n/d"
            ntp = r["stab_n_tp"] if r and r.get("stab_n_tp") not in (None,"") else "0"
            lines.append(f"{SCEN_LABEL[scen]} & {arch} & {BAL_LABEL[bal]} & {sp} & {f1} & {ntp} \\\\")
lines.append(r"\end{longtable}")
open(f"{TEX}/tables/elliptic_full.tex","w").write("\n".join(lines)+"\n")
print("[ok] elliptic_full.tex")

# ---------- agregados por arquitectura ----------
def arch_means(cond):
    sp = defaultdict(list); jc = defaultdict(list)
    for (scen,arch,bal), r in idx.items():
        if not cond(scen,arch,bal): continue
        s = f(r["stab_spearman_mean"]); j = f(r["stab_jaccard_mean"])
        if s is not None: sp[arch].append(s)
        if j is not None: jc[arch].append(j)
    return sp, jc

sp_full, jc_full = arch_means(lambda s,a,b: True)
sp_filt, _ = arch_means(lambda s,a,b: (a,s,b) in passed)

print("\n=== tab:ranking (Spearman GNNExplainer) ===")
print(f"{'arch':10s} {'completa':>14s} {'filtro':>12s}")
completa = {}
for a in ARCHS:
    cm = np.mean(sp_full[a]) if sp_full[a] else None
    fm = np.mean(sp_filt[a]) if sp_filt[a] else None
    completa[a] = (cm, len(sp_full[a]))
    print(f"{a:10s} {fmt(cm):>10s} ({len(sp_full[a])})   {fmt(fm):>7s} ({len(sp_filt[a])})")

# ---------- 2) elliptic_jaccard.tex (per-arch Jaccard + Spearman, full 60) ----------
lines = [
 r"\begin{table}[htbp]", r"\centering", r"\renewcommand{\arraystretch}{1.3}",
 r"\begin{tabular}{lcc}", r"\toprule",
 r"\textbf{Arquitectura} & \textbf{Jaccard medio} & \textbf{Spearman medio} \\", r"\midrule",
]
for a in ARCHS:
    jm = np.mean(jc_full[a]) if jc_full[a] else None
    sm = np.mean(sp_full[a]) if sp_full[a] else None
    lines.append(f"{a} & {fmt(jm)} & {fmt(sm)} \\\\")
lines += [
 r"\bottomrule", r"\end{tabular}",
 r"\caption{Comparacion entre el indice de Jaccard de aristas y la correlacion de Spearman de features (GNNExplainer) como metricas de estabilidad sobre el \textit{Elliptic Dataset}, con la metrica corregida. El Jaccard se satura hacia valores altos por la dispersion del grafo, mientras que la correlacion de Spearman conserva capacidad de discriminacion entre arquitecturas.}",
 r"\label{tab:jaccard}", r"\end{table}",
]
open(f"{TEX}/tables/elliptic_jaccard.tex","w").write("\n".join(lines)+"\n")
print("[ok] elliptic_jaccard.tex")

# ---------- 3) elliptic_stab_scenario.tex (arch x scenario) ----------
cell = defaultdict(dict)
scen_all = defaultdict(list)
for scen in SCEN_ORDER:
    for arch in ARCHS:
        vals = [f(idx[(scen,arch,bal)]["stab_spearman_mean"]) for bal in BAL_ORDER
                if (scen,arch,bal) in idx and f(idx[(scen,arch,bal)]["stab_spearman_mean"]) is not None]
        cell[arch][scen] = np.mean(vals) if vals else None
        scen_all[scen] += vals
lines = [
 r"\begin{table}[htbp]", r"\centering", r"\renewcommand{\arraystretch}{1.3}",
 r"\begin{tabular}{lccccc}", r"\toprule",
 r"\textbf{Arquitectura} & 1:1 & 1:10 & 1:50 & 1:100 & nativo \\", r"\midrule",
]
for a in ARCHS:
    cells = " & ".join(fmt(cell[a][s]) for s in SCEN_ORDER)
    lines.append(f"{a} & {cells} \\\\")
lines += [
 r"\bottomrule", r"\end{tabular}",
 r"\caption{Estabilidad de las explicaciones (correlacion de Spearman de GNNExplainer) por arquitectura y escenario de desbalance sobre el subgrafo receptivo, con la metrica corregida. GCN y GAT encabezan la estabilidad de forma transversal a los escenarios.}",
 r"\label{tab:elliptic-stab-scen}", r"\end{table}",
]
open(f"{TEX}/tables/elliptic_stab_scenario.tex","w").write("\n".join(lines)+"\n")
print("[ok] elliptic_stab_scenario.tex")

print("\n=== por escenario (media Spearman sobre arqs/balanceos, full) ===")
scen_means = {}
for scen in SCEN_ORDER:
    m = np.mean(scen_all[scen]) if scen_all[scen] else None
    scen_means[scen] = m
    print(f"  {SCEN_LABEL[scen]:8s} {fmt(m)}  (n={len(scen_all[scen])})")

# ---------- 4) FIGURAS ----------
IMG4 = f"{TEX}/chapter_4/images_ch4"; IMG5 = f"{TEX}/chapter_5/images_ch5"
plt.rcParams.update({"font.size": 12, "axes.spontaneous": False} if False else {"font.size": 12})

# ranking_khop: barras por arq (completa vs filtro)
fig, ax = plt.subplots(figsize=(8,5))
x = np.arange(len(ARCHS)); w = 0.38
comp = [completa[a][0] or 0 for a in ARCHS]
filt = [ (np.mean(sp_filt[a]) if sp_filt[a] else 0) for a in ARCHS]
ax.bar(x-w/2, comp, w, label="Corrida completa (60)", color="#4C72B0")
ax.bar(x+w/2, filt, w, label="Corrida con filtro (23)", color="#DD8452")
ax.set_xticks(x); ax.set_xticklabels(ARCHS); ax.set_ylabel("Spearman medio")
ax.set_ylim(0,1); ax.legend(); ax.set_title("Estabilidad por arquitectura (metrica corregida)")
for i,(c,fl) in enumerate(zip(comp,filt)):
    ax.text(i-w/2,c+0.01,f"{c:.2f}".replace(".",","),ha="center",fontsize=9)
    ax.text(i+w/2,fl+0.01,f"{fl:.2f}".replace(".",","),ha="center",fontsize=9)
plt.tight_layout(); plt.savefig(f"{IMG4}/ranking_khop.png", dpi=150); plt.close()
print("[ok] ranking_khop.png")

# estabilidad_escenario
fig, ax = plt.subplots(figsize=(8,5))
xs = [SCEN_LABEL[s] for s in SCEN_ORDER]
ys = [scen_means[s] or 0 for s in SCEN_ORDER]
ax.plot(xs, ys, "o-", color="#4C72B0", linewidth=2, markersize=8)
ax.set_ylabel("Spearman medio"); ax.set_xlabel("Escenario de desbalance")
ax.set_ylim(0,1); ax.grid(alpha=0.3); ax.set_title("Estabilidad por escenario (metrica corregida)")
for xi,yi in zip(xs,ys): ax.text(xi,yi+0.02,f"{yi:.2f}".replace(".",","),ha="center",fontsize=9)
plt.tight_layout(); plt.savefig(f"{IMG4}/estabilidad_escenario.png", dpi=150); plt.close()
print("[ok] estabilidad_escenario.png")

# contraste_regimen: Elliptic (filtro) vs sintetico
SYN = {"GCN":0.966,"GAT":0.964,"GraphSAGE":0.884,"TAGCN":0.888}
fig, ax = plt.subplots(figsize=(8,5))
order = ["GCN","GAT","GraphSAGE","TAGCN"]
x = np.arange(len(order)); w=0.38
ell = [ (np.mean(sp_filt[a]) if sp_filt[a] else 0) for a in order]
syn = [SYN[a] for a in order]
ax.bar(x-w/2, ell, w, label="Elliptic (disperso, corregido)", color="#4C72B0")
ax.bar(x+w/2, syn, w, label="Sintetico (denso)", color="#55A868")
ax.set_xticks(x); ax.set_xticklabels(order); ax.set_ylabel("Spearman medio (GNNExplainer)")
ax.set_ylim(0,1.05); ax.legend(); ax.set_title("Concordancia del orden de estabilidad entre regimenes")
for i,(e,s) in enumerate(zip(ell,syn)):
    ax.text(i-w/2,e+0.01,f"{e:.2f}".replace(".",","),ha="center",fontsize=9)
    ax.text(i+w/2,s+0.01,f"{s:.2f}".replace(".",","),ha="center",fontsize=9)
plt.tight_layout(); plt.savefig(f"{IMG5}/contraste_regimen.png", dpi=150); plt.close()
print("[ok] contraste_regimen.png")

print("\n=== LISTO. Valores para prosa: ===")
print("tab:ranking completa:", {a: (fmt(completa[a][0]), completa[a][1]) for a in ARCHS})
print("por escenario:", {SCEN_LABEL[s]: fmt(scen_means[s]) for s in SCEN_ORDER})
