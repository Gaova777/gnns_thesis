"""
Fase 1 FACTORIAL — los 5 análisis del runbook.
1) estabilidad por arquitectura Y explicador; 2) plausibilidad por explicador (edges y
features): ¿cuál señala mejor el patrón real?; 3) Fidelity± por explicador/arquitectura;
4) puente estabilidad↔plausibilidad por tipología Y explicador; 5) contraste Elliptic vs sintético.
Salvedades: 1 modelo/celda → descriptivo; N/A donde el explicador no produce esa métrica.
"""
import sys, csv, math, statistics as st
import numpy as np
sys.path.insert(0, ".")
TYP = {1: "STRUCTURING", 2: "LAYERING", 3: "FAN_IN", 4: "FAN_OUT"}


def f(x):
    try: return float(x)
    except: return None
def num(x): return isinstance(x, float) and not math.isnan(x)
def load(p): return list(csv.DictReader(open(p)))
def mean(xs): xs=[x for x in xs if num(x)]; return st.mean(xs) if xs else float("nan")
def corr(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if num(x) and num(y)]
    if len(pairs) < 3:
        return float("nan")
    return float(np.corrcoef([p[0] for p in pairs], [p[1] for p in pairs])[0, 1])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="phase1/results_factorial.csv")
    ap.add_argument("--node", default="phase1/results_factorial_pernode.csv")
    a = ap.parse_args()
    cell, node = load(a.cell), load(a.node)
    EX = ["GNNExplainer", "PGExplainer", "GNNShap"]
    AR = ["GraphSAGE", "GAT", "GCN", "TAGCN"]

    print("="*78)
    print("1) ESTABILIDAD (Spearman features) por ARQUITECTURA × EXPLICADOR")
    print("="*78)
    print(f"{'arch':11}" + "".join(f"{e:>14}" for e in EX))
    for ar in AR:
        vals = [mean([f(r["spearman_mean"]) for r in cell if r["architecture"]==ar and r["explainer"]==e]) for e in EX]
        print(f"{ar:11}" + "".join(f"{v:>14.3f}" for v in vals))
    print("  (PGExplainer determinista tras entrenar → estabilidad N/A por diseño)")

    print("\n" + "="*78)
    print("2) PLAUSIBILIDAD por EXPLICADOR — ¿cuál señala mejor el patrón real?")
    print("="*78)
    print(f"{'explicador':13}{'plaus_edge':>12}{'plaus_feat':>12}{'shap_conc':>12}{'celdas':>8}")
    for e in EX:
        pe = mean([f(r["plaus_edge_mean"]) for r in cell if r["explainer"]==e])
        pf = mean([f(r["plaus_feat_mean"]) for r in cell if r["explainer"]==e])
        pc = mean([f(r["shap_conc_mean"]) for r in cell if r["explainer"]==e])
        n = sum(1 for r in cell if r["explainer"]==e and f(r["n_tp"]) and f(r["n_tp"])>0)
        print(f"{e:13}{pe:>12.3f}{pf:>12.3f}{pc:>12.3f}{n:>8}")
    # PGExplainer degenerado?
    pgdeg = [r["pg_degenerate"] for r in cell if r["explainer"]=="PGExplainer"]
    from collections import Counter
    print(f"  PGExplainer pg_degenerate flags: {dict(Counter(pgdeg))} | plaus_edge medio="
          f"{mean([f(r['plaus_edge_mean']) for r in cell if r['explainer']=='PGExplainer']):.3f}")

    print("\n" + "="*78)
    print("3) FIDELITY± por EXPLICADOR × ARQUITECTURA (fid+ / fid-)")
    print("="*78)
    print(f"{'arch':11}" + "".join(f"{e:>16}" for e in EX))
    for ar in AR:
        cells_=[]
        for e in EX:
            fp = mean([f(r["fid_plus_mean"]) for r in cell if r["architecture"]==ar and r["explainer"]==e])
            fm = mean([f(r["fid_minus_mean"]) for r in cell if r["architecture"]==ar and r["explainer"]==e])
            cells_.append(f"{fp:.2f}/{fm:.2f}" if num(fp) else "  n/a  ")
        print(f"{ar:11}" + "".join(f"{c:>16}" for c in cells_))

    print("\n" + "="*78)
    print("4) PUENTE estabilidad↔plausibilidad por TIPOLOGÍA × EXPLICADOR (Pearson pernode)")
    print("="*78)
    for e in ["GNNExplainer", "GNNShap"]:   # PGExplainer sin estabilidad
        print(f"\n[{e}]  tipología : r(estab↔edges) | r(estab↔features)")
        for t, tn in TYP.items():
            rows=[r for r in node if r["explainer"]==e and int(r["node_typ"])==t]
            re_=corr([f(r["spearman"]) for r in rows],[f(r["plaus_edge"]) for r in rows])
            rf_=corr([f(r["spearman"]) for r in rows],[f(r["plaus_feat"]) for r in rows])
            print(f"  {tn:13} n={len(rows):3}  edges={re_:+.3f}   features={rf_:+.3f}")
    # global
    for e in ["GNNExplainer","GNNShap"]:
        rows=[r for r in node if r["explainer"]==e]
        rg=corr([f(r["spearman"]) for r in rows],[f(r["plaus_feat"]) for r in rows])
        print(f"  GLOBAL [{e}] estab↔features: {rg:+.3f} (n={len(rows)})")

    print("\n" + "="*78)
    print("5) CONTRASTE Elliptic vs Sintético")
    print("="*78)
    pe_all=mean([f(r["plaus_edge_mean"]) for r in cell])
    pf_all=mean([f(r["plaus_feat_mean"]) for r in cell])
    rows_=[("plausibilidad de subgrafo","NO medible (~2-3 nodos)","SÍ medible (~30 nodos)"),
           ("plaus edges (media)","no aplicable",f"{pe_all:.2f}"),
           ("plaus features (media)","no aplicable",f"{pf_all:.2f}"),
           ("Fidelity±","no medida","medida (tabla 3)"),
           ("comparar explicadores por plausibilidad","IMPOSIBLE","POSIBLE (tabla 2)")]
    print(f"{'':40}{'Elliptic':>18}{'Sintético':>18}")
    for k,el,sy in rows_: print(f"{k:40}{el:>18}{sy:>18}")


if __name__ == "__main__":
    main()
