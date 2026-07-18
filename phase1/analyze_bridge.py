"""
Fase 1 v2 — Paso 4 (puente) DESCOMPUESTO POR TIPOLOGÍA + Paso 5 (contraste Elliptic).

Reporta, además del global:
  - correlación estabilidad↔plausibilidad por tipología (edges Y features).
  - plausibilidad de features por celda y por tipología (con chequeo de honestidad:
    si sale ~1.0 en todas, la firma es demasiado separable — se reporta, no se esconde).
Salvedades iguales a Elliptic: 1 modelo/celda → descriptivo, no ANOVA con potencia.
"""
import sys, csv, glob, math, statistics as st
import numpy as np
sys.path.insert(0, ".")

TYP = {1: "STRUCTURING", 2: "LAYERING", 3: "FAN_IN", 4: "FAN_OUT"}


def f(x):
    try: return float(x)
    except: return None


def load(p): return list(csv.DictReader(open(p)))


def corr(a, b):
    if len(a) < 3: return (float("nan"), float("nan"))
    from scipy.stats import spearmanr
    return float(np.corrcoef(a, b)[0, 1]), float(spearmanr(a, b)[0])


def paired(rows, xk, yk, filt=None):
    out = []
    for r in rows:
        if filt and not filt(r): continue
        x, y = f(r[xk]), f(r[yk])
        if x is not None and y is not None:
            out.append((x, y))
    return [p[0] for p in out], [p[1] for p in out]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="phase1/results_phase1_v2.csv")
    ap.add_argument("--node", default="phase1/results_phase1_v2_pernode.csv")
    a = ap.parse_args()
    cell, node = load(a.cell), load(a.node)

    print("=" * 74)
    print("PASO 4 — PUENTE ¿la estabilidad predice la plausibilidad? (global + por tipología)")
    print("=" * 74)

    # ---- GLOBAL ----
    for yk, name in [("plaus_f1", "edges"), ("plaus_feat", "features")]:
        sp, pl = paired(node, "spearman", yk)
        if not sp:
            print(f"\n[{name}] sin datos"); continue
        rp, rs = corr(sp, pl)
        print(f"\nGLOBAL [{name}] n={len(sp)} | estab media={st.mean(sp):.3f} plaus media={st.mean(pl):.3f} "
              f"[{min(pl):.2f},{max(pl):.2f}] | corr Pearson={rp:+.3f} Spearman={rs:+.3f}")

    # ---- POR TIPOLOGÍA ----
    print("\n--- DESCOMPUESTO POR TIPOLOGÍA ---")
    print(f"{'tipología':13}{'n':>4}{'estab':>8}{'plE(F1)':>9}{'r(e)':>8}{'plF(rec)':>10}{'r(f)':>8}")
    for t, tn in TYP.items():
        filt = lambda r, t=t: int(r["node_typ"]) == t
        sp_e, pl_e = paired(node, "spearman", "plaus_f1", filt)
        sp_f, pl_f = paired(node, "spearman", "plaus_feat", filt)
        n = len(sp_e)
        re = corr(sp_e, pl_e)[0] if n >= 3 else float("nan")
        rf = corr(sp_f, pl_f)[0] if len(sp_f) >= 3 else float("nan")
        print(f"{tn:13}{n:>4}{st.mean(sp_e) if sp_e else float('nan'):>8.3f}"
              f"{st.mean(pl_e) if pl_e else float('nan'):>9.3f}{re:>+8.3f}"
              f"{st.mean(pl_f) if pl_f else float('nan'):>10.3f}{rf:>+8.3f}")
    print("  r(e)=corr estabilidad↔plausibilidad de EDGES; r(f)=↔ de FEATURES (por tipología)")

    # ---- POR CELDA ----
    print("\n--- POR CELDA ---")
    print(f"{'arch':11}{'scen':6}{'valPRAUC':>9}{'estab':>8}{'plE(F1)':>9}{'plF(rec)':>10}{'n_tp':>6}")
    for r in cell:
        print(f"{r['architecture']:11}{r['scenario']:6}{r['val_pr_auc']:>9}{r['spearman_mean']:>8}"
              f"{r['plaus_f1_mean']:>9}{r['plaus_feat_mean']:>10}{r['n_tp']:>6}")

    # ---- HONESTIDAD: ¿la plausibilidad de features es demasiado fácil (~1.0)? ----
    _, allf = paired(node, "spearman", "plaus_feat")
    if allf:
        frac1 = sum(1 for v in allf if v >= 0.999) / len(allf)
        print(f"\nHONESTIDAD features: media={st.mean(allf):.3f}, fracción==1.0={frac1:.2f}")
        if st.mean(allf) > 0.9:
            print("  ⚠ La firma (+4 sobre ruido) es MUY separable → plausibilidad de features casi trivial.")
            print("    Interpretación honesta: confirma que el explainer detecta la feature discriminante")
            print("    cuando existe una clara; para un test exigente habría que atenuar la firma (bajar +4).")

    print("\n" + "=" * 74)
    print("PASO 5 — CONTRASTE Elliptic (real) vs Sintético")
    print("=" * 74)
    _, ef = paired(node, "spearman", "plaus_f1")
    rows = [("receptive field (mediana)", "~2-3 nodos", "~30 nodos"),
            ("shift temporal", "sí (test colapsa)", "no (test≈val)"),
            ("TEST pr_auc", "~0.01", "~0.9+"),
            ("plausibilidad subgrafo medible", "NO (dispersión)", "SÍ (ground-truth)"),
            ("plausibilidad edges F1", "no aplicable", f"{st.mean(ef):.2f}" if ef else "n/a"),
            ("plausibilidad features rec@3", "no aplicable", f"{st.mean(allf):.2f}" if allf else "n/a")]
    print(f"{'':32}{'Elliptic':>16}{'Sintético':>16}")
    for k, e, s in rows:
        print(f"{k:32}{e:>16}{s:>16}")


if __name__ == "__main__":
    main()
