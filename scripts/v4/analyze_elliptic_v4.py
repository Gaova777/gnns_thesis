"""
Statistical analysis of the v4 Elliptic stability results (CPU only; numpy/scipy/pandas).

UNIT OF ANALYSIS = CONFIGURATION (scenario × arch × balancing, per explainer): the value of
a configuration is the mean over its model seeds. Main analysis: only models that pass
the quality gate (a configuration's mean uses its passing seeds). Sensitivity: the same
without the filter (every row with status "ok"; only differs from the main analysis if
explain_matrix ran with --include-gated).

Stability metric (``--metric``, default ``stability_primary``): Spearman over the full
feature ranking for GNNExplainer / ShapleyFeatures, Spearman of the edge mask for
PGExplainer. Each explainer is analysed separately (never pooled).

  H1 (scenario)     Friedman over the main scenarios, blocks = (arch, balancing);
                    paired TOST (margin ±--margin) of every scenario vs native;
                    control native_size_ctrl vs 1:1 (same size, different ratio → effect of
                    the RATIO) and native_size_ctrl vs native (same ratio, different size →
                    effect of SIZE): paired Wilcoxon + TOST.
  H2 (architecture) Friedman over architectures, blocks = (scenario, balancing), plus
                    Kruskal-Wallis on configurations; paired Wilcoxon for the 6 pairs with
                    Holm; paired TOST per pair; OLS stability ~ arch + val_pr_auc (numpy
                    lstsq, classical SE); support = n configurations per arch (< 5 →
                    "sin soporte", excluded from the tests).
  H3 (balancing)    Friedman over balancings excluding the 1:1 scenario, blocks =
                    (scenario, arch); pairwise Wilcoxon + Holm; paired TOST.
  Control           Spearman(stability, val_pr_auc) over configurations, bootstrap CI.
  Performance       PR-AUC and illicit-class F1 (val, test) per architecture in the native
                    scenario (all models, gate or not), to set against Weber et al. (2019).

Each hypothesis ends in ONE of three branches: «efecto significativo» (omnibus p < alpha),
«equivalencia (TOST)» (no effect AND every TOST comparison inside the margin) or «no se
detectó con esta potencia» (neither).

Outputs ({out_dir}): analysis_v4_units.csv, analysis_v4_tests.csv,
analysis_v4_regression.csv, analysis_v4_performance.csv, analysis_v4_summary.txt and
tables/*.tex (decimal comma, as in the manuscript).

Usage:
  uv run --frozen python scripts/v4/analyze_elliptic_v4.py
  uv run --frozen python scripts/v4/analyze_elliptic_v4.py --csv results_v4/elliptic_v4_stability.csv \
      --out-dir results_v4 --metric spearman_topk
  uv run --frozen python scripts/v4/analyze_elliptic_v4.py --selftest
"""

import argparse
import sys
import tempfile
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
ALIASES = {"GNNShap": "ShapleyFeatures"}
ARCH_ORDER = ["GCN", "GraphSAGE", "GAT", "TAGCN"]
BAL_ORDER = ["none", "class_weighting", "focal_loss"]
SCEN_ORDER = ["native", "1:10", "1:10_os", "1:20", "1:1",
              "native_size_ctrl", "1:10_subil", "1:50_subil", "1:100_subil"]
MAIN_SCEN = ["native", "1:10", "1:10_os", "1:1"]  # label mode C (28-sep design)
SIG = "efecto significativo"
EQ = "equivalencia (TOST)"
ND = "no se detectó con esta potencia"


# ─────────────────────────────────────────────────────────────────────────────
# statistics helpers
# ─────────────────────────────────────────────────────────────────────────────

def tost_paired(d, margin: float, alpha: float = 0.05) -> dict:
    """Paired TOST on differences d against ±margin (two one-sided t tests)."""
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    n = len(d)
    out = {"n": n, "mean_diff": np.nan, "ci90_lo": np.nan, "ci90_hi": np.nan,
           "p_tost": np.nan, "equivalent": False}
    if n < 2:
        return out
    m, sd = d.mean(), d.std(ddof=1)
    out["mean_diff"] = m
    if sd == 0:
        out.update(ci90_lo=m, ci90_hi=m, p_tost=0.0 if abs(m) < margin else 1.0)
    else:
        se = sd / np.sqrt(n)
        p_lo = 1 - stats.t.cdf((m + margin) / se, n - 1)   # H0: diff <= -margin
        p_hi = stats.t.cdf((m - margin) / se, n - 1)       # H0: diff >= +margin
        tq = stats.t.ppf(1 - alpha, n - 1)
        out.update(ci90_lo=m - tq * se, ci90_hi=m + tq * se, p_tost=max(p_lo, p_hi))
    out["equivalent"] = bool(out["p_tost"] < alpha)
    return out


def wilcoxon_paired(a, b) -> tuple[float, float]:
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[~np.isnan(d)]
    if len(d) < 2 or np.all(d == 0):
        return np.nan, np.nan
    r = stats.wilcoxon(d)
    return float(r.statistic), float(r.pvalue)


def holm(pvals) -> list:
    p = np.asarray(pvals, float)
    out = np.full(len(p), np.nan)
    ok = ~np.isnan(p)
    idx = np.where(ok)[0][np.argsort(p[ok])]
    m, running = len(idx), 0.0
    for rank, i in enumerate(idx):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out.tolist()


def friedman(pivot: pd.DataFrame) -> dict:
    """Friedman on a blocks × treatments pivot (complete blocks only) + Kendall's W."""
    comp = pivot.dropna()
    n, k = comp.shape
    if k < 3 or n < 2:
        return {"stat": np.nan, "p": np.nan, "n_blocks": n, "k": k, "W": np.nan,
                "note": f"no aplicable (k={k}, bloques completos={n})"}
    r = stats.friedmanchisquare(*[comp[c].values for c in comp.columns])
    return {"stat": float(r.statistic), "p": float(r.pvalue), "n_blocks": n, "k": k,
            "W": float(r.statistic / (n * (k - 1))), "note": ""}


def ols(y, X, names) -> pd.DataFrame:
    """OLS by numpy lstsq with classical standard errors."""
    y, X = np.asarray(y, float), np.asarray(X, float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    n, p = X.shape
    resid = y - X @ beta
    dof = n - p
    rows = []
    if dof > 0:
        s2 = resid @ resid / dof
        cov = s2 * np.linalg.pinv(X.T @ X)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    else:
        se = np.full(p, np.nan)
    for j, name in enumerate(names):
        t = beta[j] / se[j] if se[j] > 0 else np.nan
        pv = 2 * (1 - stats.t.cdf(abs(t), dof)) if t == t and dof > 0 else np.nan
        rows.append({"term": name, "coef": beta[j], "se": se[j], "t": t, "p": pv})
    r2 = 1 - (resid @ resid) / np.sum((y - y.mean()) ** 2) if n > 1 else np.nan
    df = pd.DataFrame(rows)
    df["n"], df["r2"] = n, r2
    return df


def boot_spearman(x, y, n_boot=2000, seed=0, conf=0.95) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 4:
        return {"rho": np.nan, "p": np.nan, "lo": np.nan, "hi": np.nan, "n": len(x)}
    r = stats.spearmanr(x, y)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        if np.ptp(x[i]) > 0 and np.ptp(y[i]) > 0:
            bs.append(stats.spearmanr(x[i], y[i]).statistic)
    a = (1 - conf) / 2
    return {"rho": float(r.statistic), "p": float(r.pvalue),
            "lo": float(np.quantile(bs, a)) if bs else np.nan,
            "hi": float(np.quantile(bs, 1 - a)) if bs else np.nan, "n": len(x)}


# ─────────────────────────────────────────────────────────────────────────────
# data
# ─────────────────────────────────────────────────────────────────────────────

def _order(values, pref):
    vals = list(dict.fromkeys(values))
    return [v for v in pref if v in vals] + sorted(v for v in vals if v not in pref)


def load_rows(csv: Path, metric: str) -> pd.DataFrame:
    df = pd.read_csv(csv)
    df["explainer"] = df["explainer"].replace(ALIASES)
    if "arch" not in df.columns and "architecture" in df.columns:
        df["arch"] = df["architecture"]
    df = df[df["status"] == "ok"].copy()
    if metric not in df.columns:
        raise SystemExit(f"metric {metric!r} not in {csv}")
    df["y"] = pd.to_numeric(df[metric], errors="coerce")
    df["gate_passed"] = df["gate_passed"].astype(str).str.lower().isin(["true", "1"])
    return df


def config_units(df: pd.DataFrame, gate_only: bool) -> pd.DataFrame:
    d = df[df.gate_passed] if gate_only else df
    d = d.dropna(subset=["y"])
    g = d.groupby(["explainer", "scenario", "arch", "balancing"], as_index=False).agg(
        y=("y", "mean"), y_sd=("y", "std"), n_seeds=("seed", "nunique"),
        val_pr_auc=("val_pr_auc", "mean"))
    g["filter"] = "gate" if gate_only else "sin_filtro"
    return g


# ─────────────────────────────────────────────────────────────────────────────
# hypotheses
# ─────────────────────────────────────────────────────────────────────────────

def _pairwise_block(u, factor, levels, blocks, margin, alpha, hyp, ex, filt, tests):
    piv = u.pivot_table(index=blocks, columns=factor, values="y")
    pairs = list(combinations([l for l in levels if l in piv.columns], 2))
    raw, recs = [], []
    for a, b in pairs:
        both = piv[[a, b]].dropna()
        w, p = wilcoxon_paired(both[a], both[b])
        t = tost_paired(both[a] - both[b], margin, alpha)
        raw.append(p)
        recs.append({"hypothesis": hyp, "explainer": ex, "filter": filt,
                     "test": "wilcoxon+tost", "comparison": f"{a} vs {b}", "stat": w,
                     "p": p, "n": len(both), "mean_diff": t["mean_diff"],
                     "ci90_lo": t["ci90_lo"], "ci90_hi": t["ci90_hi"],
                     "p_tost": t["p_tost"], "equivalent": t["equivalent"]})
    for r, ph in zip(recs, holm(raw)):
        r["p_holm"] = ph
        tests.append(r)
    return recs


def _branch(omnibus_p, tost_recs, alpha):
    if omnibus_p == omnibus_p and omnibus_p < alpha:
        return SIG
    if tost_recs and all(r["equivalent"] for r in tost_recs):
        return EQ
    return ND


def analyse(u: pd.DataFrame, ex: str, filt: str, a: argparse.Namespace, tests: list,
            regs: list, summary: list):
    u = u[(u.explainer == ex) & (u["filter"] == filt)]
    head = f"[{ex} | {filt}] n configuraciones = {len(u)}"
    summary.append(head)
    if u.empty:
        summary.append("  sin datos")
        return
    scen = _order(u.scenario, SCEN_ORDER)
    main_scen = [s for s in scen if s != a.size_ctrl]

    # ── H1 ───────────────────────────────────────────────────────────────────
    um = u[u.scenario.isin(main_scen)]
    fr = friedman(um.pivot_table(index=["arch", "balancing"], columns="scenario", values="y"))
    tests.append({"hypothesis": "H1", "explainer": ex, "filter": filt, "test": "friedman",
                  "comparison": "escenarios (bloques arch×balanceo)", "stat": fr["stat"],
                  "p": fr["p"], "n": fr["n_blocks"], "effect": fr["W"], "note": fr["note"]})
    piv = u.pivot_table(index=["arch", "balancing"], columns="scenario", values="y")
    tost_recs = []
    if a.native in piv.columns:
        for s in main_scen:
            if s == a.native or s not in piv.columns:
                continue
            both = piv[[s, a.native]].dropna()
            t = tost_paired(both[s] - both[a.native], a.margin, a.alpha)
            w, p = wilcoxon_paired(both[s], both[a.native])
            rec = {"hypothesis": "H1", "explainer": ex, "filter": filt,
                   "test": "tost_pareado", "comparison": f"{s} vs {a.native}", "stat": w,
                   "p": p, "n": t["n"], "mean_diff": t["mean_diff"],
                   "ci90_lo": t["ci90_lo"], "ci90_hi": t["ci90_hi"],
                   "p_tost": t["p_tost"], "equivalent": t["equivalent"]}
            tests.append(rec)
            tost_recs.append(rec)
    ctrl_lines = []
    if a.size_ctrl in piv.columns:
        for other, meaning in ((a.balanced, "la razón (mismo tamaño)"),
                               (a.native, "el tamaño (misma razón)")):
            if other not in piv.columns:
                continue
            both = piv[[a.size_ctrl, other]].dropna()
            w, p = wilcoxon_paired(both[a.size_ctrl], both[other])
            t = tost_paired(both[a.size_ctrl] - both[other], a.margin, a.alpha)
            tests.append({"hypothesis": "H1-control", "explainer": ex, "filter": filt,
                          "test": "wilcoxon+tost", "comparison": f"{a.size_ctrl} vs {other}",
                          "stat": w, "p": p, "n": t["n"], "mean_diff": t["mean_diff"],
                          "ci90_lo": t["ci90_lo"], "ci90_hi": t["ci90_hi"],
                          "p_tost": t["p_tost"], "equivalent": t["equivalent"],
                          "note": f"aísla el efecto de {meaning}"})
            ctrl_lines.append(f"    {a.size_ctrl} vs {other} (efecto de {meaning}): "
                              f"Δ={_f(t['mean_diff'])}, Wilcoxon p={_f(p, 4)}, "
                              f"TOST p={_f(t['p_tost'], 4)} (n={t['n']})")
    b1 = _branch(fr["p"], tost_recs, a.alpha)
    summary.append(f"  H1 escenario: Friedman χ²={_f(fr['stat'], 2)} p={_f(fr['p'], 4)} "
                   f"W={_f(fr['W'])} (bloques={fr['n_blocks']}, k={fr['k']}) {fr['note']}")
    for r in tost_recs:
        summary.append(f"    {r['comparison']}: Δ={_f(r['mean_diff'])} IC90 "
                       f"[{_f(r['ci90_lo'])}, {_f(r['ci90_hi'])}] TOST p={_f(r['p_tost'], 4)}"
                       f" → {'equivalente' if r['equivalent'] else 'no equivalente'}")
    summary += ctrl_lines
    summary.append(f"  ⇒ H1: {b1}")

    # ── H2 ───────────────────────────────────────────────────────────────────
    archs = _order(u.arch, ARCH_ORDER)
    support = u.groupby("arch").size().reindex(archs).fillna(0).astype(int)
    supported = [x for x in archs if support[x] >= a.min_support]
    for x in archs:
        tests.append({"hypothesis": "H2", "explainer": ex, "filter": filt,
                      "test": "soporte", "comparison": x, "n": int(support[x]),
                      "note": "" if support[x] >= a.min_support else "sin soporte"})
    u2 = u[u.arch.isin(supported)]
    fr2 = friedman(u2.pivot_table(index=["scenario", "balancing"], columns="arch",
                                  values="y"))
    groups = [g.y.values for _, g in u2.groupby("arch")]
    kw = stats.kruskal(*groups) if len(groups) >= 2 and all(len(g) > 1 for g in groups) \
        else None
    tests.append({"hypothesis": "H2", "explainer": ex, "filter": filt, "test": "friedman",
                  "comparison": "arquitecturas (bloques escenario×balanceo)",
                  "stat": fr2["stat"], "p": fr2["p"], "n": fr2["n_blocks"],
                  "effect": fr2["W"], "note": fr2["note"]})
    if kw is not None:
        tests.append({"hypothesis": "H2", "explainer": ex, "filter": filt,
                      "test": "kruskal", "comparison": "arquitecturas (configuraciones)",
                      "stat": float(kw.statistic), "p": float(kw.pvalue), "n": len(u2)})
    pairs2 = _pairwise_block(u2, "arch", supported, ["scenario", "balancing"], a.margin,
                             a.alpha, "H2", ex, filt, tests)
    omni2 = fr2["p"] if fr2["p"] == fr2["p"] else (float(kw.pvalue) if kw else np.nan)
    b2 = _branch(omni2, pairs2, a.alpha)
    # regression: y ~ arch dummies + val_pr_auc
    ur = u2.dropna(subset=["val_pr_auc"])
    if len(ur) > len(supported) + 1 and len(supported) >= 1:
        ref = supported[0]
        X = [np.ones(len(ur))] + [(ur.arch == x).astype(float).values for x in supported[1:]]
        X.append(ur.val_pr_auc.values)
        names = ["intercepto"] + [f"arch[{x}] (ref {ref})" for x in supported[1:]] + \
                ["val_pr_auc"]
        reg = ols(ur.y.values, np.column_stack(X), names)
        reg.insert(0, "filter", filt)
        reg.insert(0, "explainer", ex)
        regs.append(reg)
    summary.append("  H2 arquitectura: soporte " + ", ".join(
        f"{x}={support[x]}{'' if support[x] >= a.min_support else ' (sin soporte)'}"
        for x in archs))
    summary.append(f"    Friedman χ²={_f(fr2['stat'], 2)} p={_f(fr2['p'], 4)} "
                   f"W={_f(fr2['W'])} (bloques={fr2['n_blocks']}) {fr2['note']}; "
                   + (f"Kruskal H={_f(kw.statistic, 2)} p={_f(kw.pvalue, 4)}" if kw
                      else "Kruskal no aplicable"))
    for r in pairs2:
        summary.append(f"    {r['comparison']}: Δ={_f(r['mean_diff'])} Wilcoxon p_holm="
                       f"{_f(r['p_holm'], 4)} TOST p={_f(r['p_tost'], 4)} (n={r['n']})")
    if regs and regs[-1]["explainer"].iloc[0] == ex and regs[-1]["filter"].iloc[0] == filt:
        for r in regs[-1].itertuples():
            summary.append(f"    OLS {r.term}: β={_f(r.coef)} (SE {_f(r.se)}, p={_f(r.p, 4)})")
        summary.append(f"    OLS n={int(regs[-1]['n'].iloc[0])} R²={_f(regs[-1]['r2'].iloc[0])}")
    summary.append(f"  ⇒ H2: {b2}")

    # ── H3 ───────────────────────────────────────────────────────────────────
    u3 = u[u.scenario != a.balanced]
    bals = _order(u3.balancing, BAL_ORDER)
    fr3 = friedman(u3.pivot_table(index=["scenario", "arch"], columns="balancing",
                                  values="y"))
    tests.append({"hypothesis": "H3", "explainer": ex, "filter": filt, "test": "friedman",
                  "comparison": f"balanceos sin {a.balanced} (bloques escenario×arch)",
                  "stat": fr3["stat"], "p": fr3["p"], "n": fr3["n_blocks"],
                  "effect": fr3["W"], "note": fr3["note"]})
    pairs3 = _pairwise_block(u3, "balancing", bals, ["scenario", "arch"], a.margin, a.alpha,
                             "H3", ex, filt, tests)
    b3 = _branch(fr3["p"], pairs3, a.alpha)
    summary.append(f"  H3 balanceo (sin {a.balanced}): Friedman χ²={_f(fr3['stat'], 2)} "
                   f"p={_f(fr3['p'], 4)} W={_f(fr3['W'])} (bloques={fr3['n_blocks']}) "
                   f"{fr3['note']}")
    for r in pairs3:
        summary.append(f"    {r['comparison']}: Δ={_f(r['mean_diff'])} p_holm="
                       f"{_f(r['p_holm'], 4)} TOST p={_f(r['p_tost'], 4)}")
    summary.append(f"  ⇒ H3: {b3}")

    # ── control: stability vs val PR-AUC ─────────────────────────────────────
    bs = boot_spearman(u.y, u.val_pr_auc, a.bootstrap, a.boot_seed)
    tests.append({"hypothesis": "control", "explainer": ex, "filter": filt,
                  "test": "spearman_bootstrap", "comparison": "estabilidad vs val_pr_auc",
                  "stat": bs["rho"], "p": bs["p"], "n": bs["n"], "ci95_lo": bs["lo"],
                  "ci95_hi": bs["hi"]})
    summary.append(f"  Control: Spearman(estabilidad, val PR-AUC) ρ={_f(bs['rho'])} "
                   f"IC95 boot [{_f(bs['lo'])}, {_f(bs['hi'])}] p={_f(bs['p'], 4)} "
                   f"(n={bs['n']})")
    summary.append("")
    return {"H1": b1, "H2": b2, "H3": b3}


def _f(v, d=3):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "n/d"
    if v != v:
        return "n/d"
    return f"{v:.{d}g}" if abs(v) < 1e-3 and v != 0 else f"{v:.{d}f}"


def performance(df_all: pd.DataFrame, native: str) -> pd.DataFrame:
    d = df_all.drop_duplicates("run_id")
    d = d[d.scenario == native]
    cols = ["val_pr_auc", "val_f1", "test_pr_auc", "test_f1"]
    rows = []
    for arch, g in d.groupby("arch"):
        r = {"arch": arch, "balancing": "todos", "n_models": len(g)}
        for c in cols:
            r[c], r[c + "_sd"] = g[c].mean(), g[c].std()
        rows.append(r)
        for bal, gb in g.groupby("balancing"):
            r = {"arch": arch, "balancing": bal, "n_models": len(gb)}
            for c in cols:
                r[c], r[c + "_sd"] = gb[c].mean(), gb[c].std()
            rows.append(r)
    out = pd.DataFrame(rows)
    if len(out):
        out["_o"] = out.arch.map({a: i for i, a in enumerate(ARCH_ORDER)}).fillna(99)
        out = out.sort_values(["_o", "balancing"]).drop(columns="_o")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX
# ─────────────────────────────────────────────────────────────────────────────

def _c(v, d=3):
    s = _f(v, d)
    return s.replace(".", ",") if s != "n/d" else "n/d"


def _tex_escape(s: str) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def write_tex(path: Path, caption: str, label: str, header: list, rows: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [r"\begin{table}[htbp]", r"\centering\small",
             rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\begin{tabular}{" + "l" * 2 + "r" * (len(header) - 2) + "}", r"\toprule",
             " & ".join(_tex_escape(h) for h in header) + r" \\", r"\midrule"]
    lines += [" & ".join(_tex_escape(c) for c in r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_tables(out: Path, tests: pd.DataFrame, regs: pd.DataFrame, perf: pd.DataFrame,
                 units: pd.DataFrame):
    t = out / "tables"
    g = tests[tests["filter"] == "gate"] if len(tests) else tests
    # H1..H3 omnibus + pairwise
    for hyp, label in (("H1", "h1"), ("H2", "h2"), ("H3", "h3")):
        sub = g[g.hypothesis.str.startswith(hyp) & (g.test != "soporte")]
        rows = [[r.explainer, r.comparison, _c(r.stat, 3), _c(r.p, 3),
                 _c(r.get("p_holm", np.nan), 3), _c(r.get("mean_diff", np.nan)),
                 _c(r.get("p_tost", np.nan), 3), str(int(r.n)) if r.n == r.n else "n/d"]
                for _, r in sub.iterrows()]
        write_tex(t / f"elliptic_v4_{label}.tex",
                  f"{hyp}: contrastes sobre configuraciones (modelos que pasan la compuerta)",
                  f"tab:v4-{label}",
                  ["Explicador", "Contraste", "Estad.", "p", "p Holm", "Δ", "p TOST", "n"],
                  rows)
    if len(regs):
        rg = regs[regs["filter"] == "gate"]
        write_tex(t / "elliptic_v4_regresion.tex",
                  "Estabilidad $\\sim$ arquitectura + PR-AUC de validación (MCO)",
                  "tab:v4-regresion", ["Explicador", "Término", "β", "EE", "p", "n"],
                  [[r.explainer, r.term, _c(r.coef), _c(r.se), _c(r.p), str(int(r.n))]
                   for r in rg.itertuples()])
    if len(perf):
        write_tex(t / "elliptic_v4_rendimiento_nativo.tex",
                  "Rendimiento en el escenario nativo (media sobre semillas; F1 de la clase "
                  "ilícita). Referencia: Weber et al. (2019)",
                  "tab:v4-rendimiento",
                  ["Arquitectura", "Balanceo", "PR-AUC val", "F1 val", "PR-AUC test",
                   "F1 test", "n"],
                  [[r.arch, r.balancing, _c(r.val_pr_auc), _c(r.val_f1), _c(r.test_pr_auc),
                    _c(r.test_f1), str(int(r.n_models))] for r in perf.itertuples()])
    ug = units[units["filter"] == "gate"]
    if len(ug):
        piv = ug.pivot_table(index=["explainer", "arch"], columns="scenario", values="y")
        cols = _order(piv.columns, SCEN_ORDER)
        write_tex(t / "elliptic_v4_estabilidad_escenario.tex",
                  "Estabilidad media por arquitectura y escenario (configuraciones que pasan "
                  "la compuerta)", "tab:v4-estab-escenario",
                  ["Explicador", "Arquitectura"] + cols,
                  [[ex, ar] + [_c(piv.loc[(ex, ar), c]) for c in cols]
                   for ex, ar in piv.index])


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def run(a: argparse.Namespace) -> dict:
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = load_rows(Path(a.csv), a.metric)
    allrows = pd.read_csv(a.csv)
    allrows["explainer"] = allrows["explainer"].replace(ALIASES)
    units = pd.concat([config_units(df, True), config_units(df, False)], ignore_index=True)
    units.to_csv(out / "analysis_v4_units.csv", index=False)
    explainers = [e for e in ["GNNExplainer", "PGExplainer", "ShapleyFeatures"]
                  if e in set(units.explainer)] + sorted(
        set(units.explainer) - {"GNNExplainer", "PGExplainer", "ShapleyFeatures"})

    tests, regs = [], []
    summary = [f"Análisis v4 Elliptic — métrica: {a.metric}; unidad: configuración (media de "
               f"semillas); α={a.alpha}; margen TOST ±{a.margin}; soporte mínimo "
               f"{a.min_support} configuraciones por arquitectura.",
               f"Fuente: {a.csv} ({len(df)} filas 'ok'; "
               f"{int(df.gate_passed.sum())} pasan la compuerta).", ""]
    branches = {}
    for filt in ("gate", "sin_filtro"):
        summary.append("=" * 78)
        summary.append("PRINCIPAL: solo modelos que pasan la compuerta" if filt == "gate"
                       else "SENSIBILIDAD: sin filtro de compuerta")
        summary.append("=" * 78)
        if filt == "sin_filtro" and not (~df.gate_passed).any():
            summary.append("(ninguna fila explicada falla la compuerta: la sensibilidad "
                           "coincide con el análisis principal; corre explain_matrix con "
                           "--include-gated para que difiera)")
        for ex in explainers:
            branches[(ex, filt)] = analyse(units, ex, filt, a, tests, regs, summary)

    tests_df = pd.DataFrame(tests)
    regs_df = pd.concat(regs, ignore_index=True) if regs else pd.DataFrame()
    perf = performance(allrows[allrows.get("val_pr_auc").notna()] if "val_pr_auc" in
                       allrows else allrows, a.native)
    tests_df.to_csv(out / "analysis_v4_tests.csv", index=False)
    regs_df.to_csv(out / "analysis_v4_regression.csv", index=False)
    perf.to_csv(out / "analysis_v4_performance.csv", index=False)
    write_tables(out, tests_df, regs_df, perf, units)

    summary.append("=" * 78)
    summary.append(f"RENDIMIENTO en '{a.native}' (todos los modelos; F1 = clase ilícita; test "
                   "con umbral calibrado en validación)")
    for r in perf[perf.balancing == "todos"].itertuples() if len(perf) else []:
        summary.append(f"  {r.arch:10s} PR-AUC val {_f(r.val_pr_auc)} test {_f(r.test_pr_auc)}"
                       f" | F1 val {_f(r.val_f1)} test {_f(r.test_f1)} (n={r.n_models})")
    summary.append("  Referencia Weber et al. (2019): F1 ilícito en test, GCN 0,628 y "
                   "Random Forest 0,788 (todas las variables); comparar con cautela: otro "
                   "split temporal y umbral.")
    cs = Path(a.cross_seed) if a.cross_seed else Path(a.csv).with_name(
        "elliptic_v4_cross_seed.csv")
    if cs.exists():
        c = pd.read_csv(cs)
        c["explainer"] = c["explainer"].replace(ALIASES)
        summary.append("")
        summary.append(f"ESTABILIDAD ENTRE SEMILLAS ({cs.name}; media de cs_primary por pares):")
        for (ex, ar), v in c.groupby(["explainer", "arch"])["cs_primary"].mean().items():
            summary.append(f"  {ex:16s} {ar:10s} {_f(v)}")
    summary.append("")
    summary.append("RAMAS (análisis principal):")
    for ex in explainers:
        b = branches.get((ex, "gate")) or {}
        summary.append(f"  {ex:16s} H1: {b.get('H1', 'n/d')} | H2: {b.get('H2', 'n/d')} | "
                       f"H3: {b.get('H3', 'n/d')}")
    text = "\n".join(summary)
    (out / "analysis_v4_summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return branches


def make_synthetic(path: Path, seed: int = 0) -> None:
    """Fake elliptic_v4_stability.csv: an ARCH effect (GAT/GCN > SAGE > TAGCN), no
    scenario or balancing effect, ~80% of models pass the gate."""
    rng = np.random.default_rng(seed)
    eff = {"GCN": 0.05, "GAT": 0.05, "GraphSAGE": 0.0, "TAGCN": -0.06}
    rows = []
    for sd in (42, 43, 44):
        for sc in MAIN_SCEN:
            for ar in ARCH_ORDER:
                for bal in BAL_ORDER:
                    pr = rng.uniform(0.2, 0.6)
                    gate = rng.random() < 0.8
                    rid = f"{sc}_{ar}_{bal}" + ("" if sd == 42 else f"_s{sd}")
                    for ex in ("GNNExplainer", "PGExplainer", "ShapleyFeatures"):
                        base = {"GNNExplainer": 0.72, "PGExplainer": 0.55,
                                "ShapleyFeatures": 0.9}[ex]
                        v = float(np.clip(base + eff[ar] + rng.normal(0, 0.015), -1, 1))
                        rows.append({"run_id": rid, "seed": sd, "scenario": sc, "arch": ar,
                                     "balancing": bal, "explainer": ex, "gate_passed": gate,
                                     "status": "ok", "val_pr_auc": pr,
                                     "val_f1": pr * 0.9, "test_pr_auc": pr * 0.5,
                                     "test_f1": pr * 0.4,
                                     "spearman_full": np.nan if ex == "PGExplainer" else v,
                                     "spearman_edges": v if ex != "ShapleyFeatures" else np.nan,
                                     "spearman_topk": v - 0.1,
                                     "stability_primary": v})
    pd.DataFrame(rows).to_csv(path, index=False)


def main(argv=None):
    p = argparse.ArgumentParser(description="v4 Elliptic stability analysis (H1-H3)")
    p.add_argument("--csv", default="./results_v4/elliptic_v4_stability.csv")
    p.add_argument("--cross-seed", default=None,
                   help="default: elliptic_v4_cross_seed.csv next to --csv")
    p.add_argument("--out-dir", default="./results_v4")
    p.add_argument("--metric", default="stability_primary",
                   help="stability_primary | spearman_full | spearman_topk | spearman_edges"
                        " | jaccard_edges_topk")
    p.add_argument("--native", default="native")
    p.add_argument("--size-ctrl", default="native_size_ctrl")
    p.add_argument("--balanced", default="1:1")
    p.add_argument("--margin", type=float, default=0.05)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--min-support", type=int, default=5)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--boot-seed", type=int, default=0)
    p.add_argument("--selftest", action="store_true",
                   help="generate a synthetic CSV in a temp dir and run everything")
    a = p.parse_args(argv)
    if a.selftest:
        tmp = Path(tempfile.mkdtemp(prefix="v4_analysis_selftest_"))
        a.csv, a.out_dir, a.bootstrap = str(tmp / "elliptic_v4_stability.csv"), str(tmp), 300
        # Seed 1: with no planted scenario/balancing effect each omnibus test still has an
        # α = 5 % false-positive rate; seed 0 hits one on H3 with the 4 main scenarios of the
        # 28-sep design (seeds 0-9: 2 false positives in 20 tests, as expected).
        make_synthetic(Path(a.csv), seed=1)
        br = run(a)
        g = br[("GNNExplainer", "gate")]
        assert g["H2"] == SIG, f"planted arch effect not detected: {g}"
        assert g["H1"] in (EQ, ND) and g["H3"] in (EQ, ND), g
        for f in ("analysis_v4_units.csv", "analysis_v4_tests.csv",
                  "analysis_v4_regression.csv", "analysis_v4_performance.csv",
                  "analysis_v4_summary.txt", "tables/elliptic_v4_h1.tex",
                  "tables/elliptic_v4_h2.tex", "tables/elliptic_v4_h3.tex",
                  "tables/elliptic_v4_regresion.tex",
                  "tables/elliptic_v4_rendimiento_nativo.tex"):
            assert (tmp / f).exists(), f
        print(f"\nSELFTEST OK ({tmp})")
        return 0
    run(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
