"""
Fase 1 ROBUSTO — análisis inferencial (encargo ENCARGO_robustez_estadistica.md).
Con S semillas de modelo por celda hay varianza entre entrenamientos → inferencia con potencia.

Produce:
  - results_robust_agg.csv : media Y dispersión (sd) por celda (agregado sobre model_seeds).
  - reporte impreso con: (0) sanity + reproducción del slice seed=42 vs factorial; (1) media±sd e
    IC 95% por arquitectura×explicador; (2) Kruskal-Wallis + ANOVA con tamaños de efecto
    (epsilon²/eta²) por factor; (3) test pareado Wilcoxon PGExplainer vs GNNExplainer (plaus_edge);
    (4) puente estabilidad↔plausibilidad con IC bootstrap; (5) robustez a graph_seed (PIEZA 3).
No tunea nada: solo cuantifica incertidumbre sobre los mismos hallazgos.
"""
import sys, csv, math, argparse
import numpy as np
from scipy import stats

MET = ["spearman_mean", "plaus_edge_mean", "plaus_feat_mean", "shap_conc_mean", "fid_plus_mean", "fid_minus_mean"]
AR = ["GraphSAGE", "GAT", "GCN", "TAGCN"]
EX = ["GNNExplainer", "PGExplainer", "GNNShap"]
BAL = ["none", "class_weighting", "focal_loss"]
TYP = {1: "STRUCTURING", 2: "LAYERING", 3: "FAN_IN", 4: "FAN_OUT"}


def f(x):
    try: return float(x)
    except: return None
def num(x): return isinstance(x, float) and not math.isnan(x)
def load(*paths):
    rows = []
    for p in paths:
        if p:
            try: rows += list(csv.DictReader(open(p)))
            except FileNotFoundError: pass
    return rows
def col(rows, m): return [v for v in (f(r[m]) for r in rows) if num(v)]
def key_seed(r): return (r["graph_seed"], r["architecture"], r["scenario"], r["balancing"], r["explainer"])


def aggregate(cell, out_csv):
    """Media ± sd sobre model_seeds por celda (graph,arch,scen,bal,explainer)."""
    groups = {}
    for r in cell:
        groups.setdefault(key_seed(r), []).append(r)
    fields = ["graph_seed", "architecture", "scenario", "balancing", "explainer", "n_seeds"]
    for m in MET: fields += [m, m.replace("_mean", "_sd")]
    rows = []
    for k, rs in sorted(groups.items()):
        d = dict(zip(["graph_seed", "architecture", "scenario", "balancing", "explainer"], k), n_seeds=len(rs))
        for m in MET:
            vals = col(rs, m)
            d[m] = round(float(np.mean(vals)), 4) if vals else float("nan")
            d[m.replace("_mean", "_sd")] = round(float(np.std(vals, ddof=1)), 4) if len(vals) > 1 else float("nan")
        rows.append(d)
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)
    return rows


def ci95(vals):
    n = len(vals)
    if n < 2: return (float("nan"), float("nan"))
    m, sd = float(np.mean(vals)), float(np.std(vals, ddof=1))
    h = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)
    return (m - h, m + h)


def eff_sizes(groups):
    """groups: lista de arrays por nivel. Devuelve (KW H, KW p, epsilon2, ANOVA F, p, eta2, normal?)."""
    groups = [np.asarray(g, float) for g in groups if len(g) >= 2]
    if len(groups) < 2: return None
    N = sum(len(g) for g in groups); k = len(groups)
    H, pkw = stats.kruskal(*groups)
    eps2 = (H - k + 1) / (N - k) if N > k else float("nan")
    F, pan = stats.f_oneway(*groups)
    eta2 = (F * (k - 1)) / (F * (k - 1) + (N - k)) if num(float(F)) else float("nan")
    # normalidad: Shapiro sobre residuales (x - media de su grupo)
    res = np.concatenate([g - g.mean() for g in groups])
    try: normal = stats.shapiro(res).pvalue > 0.05 if 3 <= len(res) <= 5000 else False
    except Exception: normal = False
    return dict(H=H, pkw=pkw, eps2=eps2, F=float(F), pan=pan, eta2=eta2, normal=normal, N=N, k=k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="phase1/results_robust.csv")
    ap.add_argument("--node", default="phase1/results_robust_pernode.csv")
    ap.add_argument("--graph-cell", default="phase1/results_robust_graph.csv")
    ap.add_argument("--graph-node", default="phase1/results_robust_graph_pernode.csv")
    ap.add_argument("--factorial", default="phase1/results_factorial.csv")
    ap.add_argument("--agg-out", default="phase1/results_robust_agg.csv")
    a = ap.parse_args()
    cell = load(a.cell, a.graph_cell)
    node = load(a.node, a.graph_node)
    seeds = sorted(set(r["model_seed"] for r in cell))
    graphs = sorted(set(r["graph_seed"] for r in cell))

    print("=" * 80)
    print("0) SANITY + reproducción del slice model_seed=42 vs factorial")
    print("=" * 80)
    print(f"  filas celda={len(cell)}  filas pernode={len(node)}  model_seeds={seeds}  graph_seeds={graphs}")
    # NaN silencioso: en celdas con n_tp>0, la métrica propia del explicador no debe ser NaN
    silent = 0
    for r in cell:
        if not f(r["n_tp"]) or f(r["n_tp"]) == 0: continue
        e = r["explainer"]
        need = {"GNNExplainer": ["spearman_mean", "plaus_edge_mean", "fid_plus_mean"],
                "GNNShap": ["spearman_mean", "plaus_feat_mean", "fid_plus_mean"],
                "PGExplainer": ["plaus_edge_mean", "fid_plus_mean"]}[e]
        for m in need:
            if not num(f(r[m])): silent += 1
    print(f"  NaN silencioso (métrica propia ausente con n_tp>0): {silent}")
    # reproducción: slice graph0/seed42 debe casar con factorial (estabilidad/plaus, no fidelidad)
    fac = load(a.factorial)
    if fac:
        s42 = [r for r in cell if r["graph_seed"] == "0" and r["model_seed"] == "42"]
        def mp(rows, m):
            v = col(rows, m); return round(float(np.mean(v)), 3) if v else float("nan")
        for m in ["spearman_mean", "plaus_edge_mean", "plaus_feat_mean"]:
            print(f"    {m:18} robust(g0,s42)={mp(s42,m):.3f}  factorial={mp(fac,m):.3f}  "
                  f"(deben coincidir; fidelidad NO, cambió a definición manual uniforme)")

    agg = aggregate(cell, a.agg_out)
    print(f"  → escrito {a.agg_out} ({len(agg)} celdas con media Y sd)")

    print("\n" + "=" * 80)
    print(f"1) MEDIA ± SD (IC95%) sobre {len(seeds)} model_seeds — grafo g0, por arquitectura × explicador")
    print("=" * 80)
    g0 = [r for r in cell if r["graph_seed"] == "0"]
    for m in ["spearman_mean", "plaus_edge_mean", "plaus_feat_mean", "fid_plus_mean", "fid_minus_mean"]:
        print(f"\n  [{m}]")
        for ar in AR:
            parts = []
            for e in EX:
                vals = col([r for r in g0 if r["architecture"] == ar and r["explainer"] == e], m)
                if len(vals) >= 2:
                    lo, hi = ci95(vals)
                    parts.append(f"{e[:4]}={np.mean(vals):.3f}±{np.std(vals,ddof=1):.3f}")
                else:
                    parts.append(f"{e[:4]}=n/a")
            print(f"    {ar:11} " + "  ".join(parts))

    print("\n" + "=" * 80)
    print("2) KRUSKAL-WALLIS (primario) + ANOVA — efecto de cada factor, con tamaño de efecto")
    print("=" * 80)
    print("   epsilon²/eta²: 0.01 pequeño · 0.06 medio · 0.14 grande")
    factor_levels = {"architecture": AR, "explainer": EX, "balancing": BAL}
    for m in ["spearman_mean", "plaus_edge_mean", "plaus_feat_mean"]:
        print(f"\n  métrica: {m}")
        for fac_name, levels in factor_levels.items():
            groups = [col([r for r in g0 if r[fac_name] == lv], m) for lv in levels]
            groups = [g for g in groups if len(g) >= 2]
            r = eff_sizes(groups)
            if not r:
                print(f"    {fac_name:13}: sin datos suficientes"); continue
            flag = "normal" if r["normal"] else "NO-normal→KW"
            print(f"    {fac_name:13}: KW H={r['H']:.2f} p={r['pkw']:.4f} ε²={r['eps2']:.3f} | "
                  f"ANOVA F={r['F']:.2f} p={r['pan']:.4f} η²={r['eta2']:.3f} | {flag} (N={r['N']},k={r['k']})")

    print("\n" + "=" * 80)
    print("3) TEST PAREADO — ¿PGExplainer > GNNExplainer en plaus_edge sobrevive a la varianza de modelo?")
    print("=" * 80)
    pg = {(r["graph_seed"], r["architecture"], r["scenario"], r["balancing"], r["model_seed"]): f(r["plaus_edge_mean"])
          for r in cell if r["explainer"] == "PGExplainer"}
    gn = {(r["graph_seed"], r["architecture"], r["scenario"], r["balancing"], r["model_seed"]): f(r["plaus_edge_mean"])
          for r in cell if r["explainer"] == "GNNExplainer"}
    pairs = [(pg[k], gn[k]) for k in pg if k in gn and num(pg[k]) and num(gn[k])]
    if len(pairs) >= 6:
        pgv = np.array([p[0] for p in pairs]); gnv = np.array([p[1] for p in pairs])
        diff = pgv - gnv
        W, pw = stats.wilcoxon(pgv, gnv)
        print(f"  n pares (celda×model_seed)={len(pairs)}")
        print(f"  PGExplainer plaus_edge = {pgv.mean():.3f} ± {pgv.std(ddof=1):.3f}")
        print(f"  GNNExplainer plaus_edge= {gnv.mean():.3f} ± {gnv.std(ddof=1):.3f}")
        print(f"  diferencia media (PG-GNN) = {diff.mean():+.3f}  (>0 en {100*np.mean(diff>0):.0f}% de los pares)")
        print(f"  Wilcoxon signed-rank W={W:.1f} p={pw:.2e}  → {'PG > GNN robusto' if pw<0.05 and diff.mean()>0 else 'NO significativo'}")
    else:
        print(f"  pares insuficientes ({len(pairs)})")

    print("\n" + "=" * 80)
    print("4) PUENTE estabilidad↔plausibilidad con IC bootstrap (esperado: incluye 0 → nulo)")
    print("=" * 80)
    def boot_ci(xs, ys, n=2000):
        xs, ys = np.asarray(xs), np.asarray(ys)
        r0 = np.corrcoef(xs, ys)[0, 1]
        idx = np.arange(len(xs)); rng = np.random.RandomState(0)
        bs = []
        for _ in range(n):
            s = rng.choice(idx, len(idx), replace=True)
            if np.std(xs[s]) > 0 and np.std(ys[s]) > 0:
                bs.append(np.corrcoef(xs[s], ys[s])[0, 1])
        return r0, np.percentile(bs, 2.5), np.percentile(bs, 97.5)
    for e in ["GNNExplainer", "GNNShap"]:
        rows = [r for r in node if r["explainer"] == e]
        xs = [f(r["spearman"]) for r in rows]; ys = [f(r["plaus_feat"]) for r in rows]
        p = [(x, y) for x, y in zip(xs, ys) if num(x) and num(y)]
        if len(p) >= 10:
            r0, lo, hi = boot_ci([x for x, _ in p], [y for _, y in p])
            incl0 = "incluye 0 → NULO" if lo <= 0 <= hi else "excluye 0"
            print(f"  [{e}] estab↔features  r={r0:+.3f}  IC95%=[{lo:+.3f}, {hi:+.3f}]  n={len(p)}  → {incl0}")
    # por tipología (GNNExplainer, edges)
    print("  por tipología (GNNExplainer, estab↔edges):")
    for t, tn in TYP.items():
        rows = [r for r in node if r["explainer"] == "GNNExplainer" and r["node_typ"] == str(t)]
        p = [(f(r["spearman"]), f(r["plaus_edge"])) for r in rows]
        p = [(x, y) for x, y in p if num(x) and num(y)]
        if len(p) >= 10:
            r0, lo, hi = boot_ci([x for x, _ in p], [y for _, y in p])
            print(f"    {tn:12} r={r0:+.3f} IC95%=[{lo:+.3f},{hi:+.3f}] n={len(p)}")

    print("\n" + "=" * 80)
    print("5) ROBUSTEZ A GRAPH_SEED (PIEZA 3) — ¿los hallazgos dependen de la instancia?")
    print("=" * 80)
    if len(graphs) < 2:
        print("  (solo un graph_seed cargado; corre la matriz reducida sobre g1/g2 para esta sección)")
    else:
        # comparar SOLO las combinaciones (arch,scen,bal) presentes en TODOS los grafos: el run
        # principal (g0) es matriz completa y el run de grafos (g1,g2) reducido → filtrar al común.
        combos = {}
        for g in graphs:
            combos[g] = set((r["architecture"], r["scenario"], r["balancing"])
                            for r in cell if r["graph_seed"] == g)
        common = set.intersection(*combos.values())
        print(f"  combos (arch,scen,bal) comunes a los {len(graphs)} grafos: {len(common)}")
        def gvals(g, e, m):
            return col([r for r in cell if r["graph_seed"] == g and r["explainer"] == e
                        and (r["architecture"], r["scenario"], r["balancing"]) in common], m)
        print("  plaus_edge PGExplainer (media±sd sobre celdas comunes) por graph_seed:")
        for g in graphs:
            v = gvals(g, "PGExplainer", "plaus_edge_mean")
            if v: print(f"    g{g}: {np.mean(v):.3f} ± {np.std(v,ddof=1):.3f} (n={len(v)})")
        print("  plaus_edge GNNExplainer por graph_seed (para el contraste PG>GNN):")
        for g in graphs:
            v = gvals(g, "GNNExplainer", "plaus_edge_mean")
            if v: print(f"    g{g}: {np.mean(v):.3f} ± {np.std(v,ddof=1):.3f} (n={len(v)})")
        print("  estabilidad GNNExplainer por graph_seed:")
        for g in graphs:
            v = gvals(g, "GNNExplainer", "spearman_mean")
            if v: print(f"    g{g}: {np.mean(v):.3f} ± {np.std(v,ddof=1):.3f} (n={len(v)})")


if __name__ == "__main__":
    main()
