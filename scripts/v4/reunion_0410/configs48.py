"""
Reunión con Cristian, 04-oct-2026: la unidad es la CONFIGURACIÓN, no el modelo.

Las semillas 42/43/44 no son modelos distintos: son la incertidumbre de una configuración
(arquitectura × escenario × pérdida, 48 en total). Este script:

  1. Tabla de las 48 configuraciones con media, sd e IC 95 % t (n = 3 semillas) de PR-AUC, F1,
     MCC y ROC-AUC en validación y test. F1/MCC de validación = ``val_metrics`` (argmax en la
     mejor época), que es lo que usa la compuerta actual (``scripts/train_matrix.py``:
     ``quality_passed = val_argmax.f1 >= 0.30 and val_argmax.mcc >= 0.15``). F1/MCC de test =
     ``test_metrics`` (umbral calibrado en validación), igual que ``analyze_elliptic_v4.py``.
     Se guardan también ``val_metrics_calibrated`` y ``test_metrics_argmax``.
     ROC-AUC no está en los meta.json: se recalcula en CPU desde los checkpoints con
     ``--roc`` (inferencia de grafo completo, sin entrenar; se cachea en roc_auc_cpu.csv y se
     comprueba contra el PR-AUC del meta.json).
  2. Compuerta por CONFIGURACIÓN (media de val F1 >= 0,30 y media de val MCC >= 0,15) y dos
     variantes conservadoras (media - sd y límite inferior del IC 95 % t), comparadas con la
     compuerta por modelo (alguna / mayoría / todas las semillas).
  3. H1/H2/H3 sobre la estabilidad con la configuración como unidad (media de las 3 semillas de
     ``stability_primary``, SIEMPRE las 3, no solo las que pasan), reutilizando
     ``analyze_elliptic_v4.analyse`` (TOST ±0,05, Friedman/Kruskal, Wilcoxon + Holm). Principal:
     configuraciones que pasan la compuerta de configuración, sin 1:1. Sensibilidades: con 1:1,
     compuerta conservadora, sin compuerta, y la reproducción del análisis v4 original.
  4. Selección del candidato arquitectura × pérdida por val PR-AUC en native, 1:10 y 1:10_os,
     y costo proyectado de un Optuna de 100 trials por escenario a partir de ``duration_s``.

Solo CPU. No modifica nada fuera de results_v4/reunion_0410/configs48/.

Uso:
  ~/.local/bin/uv run --frozen python scripts/v4/reunion_0410/configs48.py --roc   # una vez
  ~/.local/bin/uv run --frozen python scripts/v4/reunion_0410/configs48.py
"""

import argparse
import importlib.util
import json
import sys
from itertools import permutations, product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "analyze_elliptic_v4", ROOT / "scripts" / "v4" / "analyze_elliptic_v4.py")
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)

ARCHS = ["GCN", "GraphSAGE", "GAT", "TAGCN"]
BALS = ["none", "class_weighting", "focal_loss"]
SCENS = ["native", "1:10", "1:10_os", "1:1"]
SEL_SCENS = ["native", "1:10", "1:10_os"]
EXPLAINERS = ["GNNExplainer", "PGExplainer", "ShapleyFeatures"]
F1_MIN, MCC_MIN = 0.30, 0.15
N_TRIALS_V4 = 8
TRIAL_EPOCHS = 50

METRICS = ["val_pr_auc", "val_f1", "val_mcc", "val_roc_auc",
           "test_pr_auc", "test_f1", "test_mcc", "test_roc_auc",
           "val_f1_cal", "val_mcc_cal", "test_f1_argmax", "test_mcc_argmax"]


def _in_matrix(meta: dict) -> bool:
    """Only the 144 models of the v4 matrix (4 scenarios × 3 seeds, label mode C); other runs
    written later to the same models dir (e.g. extra scenarios) are ignored."""
    return (meta.get("scenario") in SCENS and int(meta.get("seed", -1)) in (42, 43, 44)
            and meta.get("label_mode", "licit_unknown") == "licit_unknown")


# ─────────────────────────────────────────────────────────────────────────────
# 0. ROC-AUC en CPU (opcional, se cachea)
# ─────────────────────────────────────────────────────────────────────────────

def compute_roc(models_dir: Path, out_csv: Path, threads: int) -> pd.DataFrame:
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_auc_score

    from src.data.loader import apply_label_mode, load_elliptic
    from src.data.preprocessing import preprocess
    from src.training.trainer import build_model

    torch.set_num_threads(threads)
    data = load_elliptic(root=str(ROOT / "data"))
    apply_label_mode(data, "licit_unknown")
    preprocess(data, train_range=(1, 34), val_range=(35, 42), test_range=(43, 49))
    x, ei = data.x, data.edge_index
    rows = []
    metas = sorted(models_dir.glob("*_meta.json"))
    for i, p in enumerate(metas, 1):
        m = json.loads(p.read_text(encoding="utf-8"))
        if not _in_matrix(m):
            continue
        bp = m.get("best_params", {}) or {}
        kw = {}
        if m["architecture"] == "GAT" and "heads" in bp:
            kw["heads"] = bp["heads"]
        if m["architecture"] == "TAGCN" and "K" in bp:
            kw["K"] = bp["K"]
        model = build_model(m["architecture"], in_channels=data.num_node_features,
                            hidden_channels=bp.get("hidden_dim", 128),
                            num_layers=bp.get("num_layers", 2),
                            dropout=bp.get("dropout", 0.3), **kw)
        model.load_state_dict(torch.load(models_dir / m["checkpoint"], map_location="cpu",
                                         weights_only=True))
        model.eval()
        with torch.no_grad():
            out = model(x, ei)
        probs = F.softmax(out, dim=-1)[:, 1].numpy()
        pred = out.argmax(dim=-1).numpy()
        r = {"run_id": m["run_id"]}
        for split in ("val", "test"):
            mask = getattr(data, f"{split}_mask").numpy()
            y = data.y.numpy()[mask]
            pr, rc, _ = precision_recall_curve(y, probs[mask], pos_label=1)
            r[f"{split}_roc_auc"] = float(roc_auc_score(y, probs[mask]))
            r[f"{split}_pr_auc_cpu"] = float(auc(rc, pr))
            r[f"{split}_f1_argmax_cpu"] = float(f1_score(y, pred[mask], zero_division=0))
            r[f"{split}_n"] = int(mask.sum())
        rows.append(r)
        print(f"  [{i}/{len(metas)}] {m['run_id']}: val ROC-AUC {r['val_roc_auc']:.4f} "
              f"test ROC-AUC {r['test_roc_auc']:.4f}", flush=True)
        del model, out
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 1. modelos y configuraciones
# ─────────────────────────────────────────────────────────────────────────────

def load_models(models_dir: Path, roc_csv: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(models_dir.glob("*_meta.json")):
        m = json.loads(p.read_text(encoding="utf-8"))
        if not _in_matrix(m):
            continue
        vm, vc = m["val_metrics"], m["val_metrics_calibrated"]
        tm, ta = m["test_metrics"], m["test_metrics_argmax"]
        bp = m.get("best_params", {}) or {}
        rows.append({
            "run_id": m["run_id"], "scenario": m["scenario"], "arch": m["architecture"],
            "balancing": m["balancing"], "seed": int(m["seed"]),
            "quality_passed": bool(m["quality_passed"]),
            "val_pr_auc": vm["pr_auc"], "val_f1": vm["f1"], "val_mcc": vm["mcc"],
            "val_f1_cal": vc["f1"], "val_mcc_cal": vc["mcc"],
            "test_pr_auc": tm["pr_auc"], "test_f1": tm["f1"], "test_mcc": tm["mcc"],
            "test_f1_argmax": ta["f1"], "test_mcc_argmax": ta["mcc"],
            "threshold": m["calibrated_threshold"],
            "duration_s": m["duration_s"], "epochs_run": m["epochs_run"],
            "best_epoch": m["best_epoch"], "hp_source": m["hp_source"],
            "optuna_best_score": m.get("optuna_best_score"),
            "hidden_dim": bp.get("hidden_dim"), "num_layers": bp.get("num_layers"),
            "heads": bp.get("heads"), "K": bp.get("K"), "lr": bp.get("lr"),
            "dropout": bp.get("dropout"), "weight_decay": bp.get("weight_decay"),
        })
    df = pd.DataFrame(rows)
    assert len(df) == 144, f"expected the 144 models of the v4 matrix, found {len(df)}"
    # Sanity: the stored gate must be exactly val argmax F1/MCC against 0,30/0,15.
    recomputed = (df.val_f1 >= F1_MIN) & (df.val_mcc >= MCC_MIN)
    assert (recomputed == df.quality_passed).all(), "quality_passed != val_metrics gate"
    if roc_csv.exists():
        roc = pd.read_csv(roc_csv)
        df = df.merge(roc, on="run_id", how="left")
        for s in ("val", "test"):
            df[f"{s}_pr_auc_absdiff_cpu"] = (df[f"{s}_pr_auc"] - df[f"{s}_pr_auc_cpu"]).abs()
    else:
        df["val_roc_auc"] = np.nan
        df["test_roc_auc"] = np.nan
    return df


def tci(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    n = len(v)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan, 0
    m = v.mean()
    if n < 2:
        return m, np.nan, np.nan, np.nan, n
    sd = v.std(ddof=1)
    h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return m, sd, m - h, m + h, n


def config_table(models: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (sc, ar, bal), g in models.groupby(["scenario", "arch", "balancing"]):
        r = {"scenario": sc, "arch": ar, "balancing": bal, "n_seeds": g.seed.nunique(),
             "n_pass_model": int(g.quality_passed.sum()),
             "seeds_pass": ",".join(str(s) for s in sorted(g[g.quality_passed].seed))}
        for c in METRICS:
            m, sd, lo, hi, _ = tci(g[c])
            r[f"{c}_mean"], r[f"{c}_sd"], r[f"{c}_ci_lo"], r[f"{c}_ci_hi"] = m, sd, lo, hi
        for s in (42, 43, 44):
            gs = g[g.seed == s]
            r[f"val_f1_s{s}"] = gs.val_f1.iloc[0] if len(gs) else np.nan
            r[f"val_mcc_s{s}"] = gs.val_mcc.iloc[0] if len(gs) else np.nan
        rows.append(r)
    t = pd.DataFrame(rows)
    # Gate criteria
    t["g_model_any"] = t.n_pass_model >= 1
    t["g_model_majority"] = t.n_pass_model >= 2
    t["g_model_all"] = t.n_pass_model == 3
    t["g_cfg_mean"] = (t.val_f1_mean >= F1_MIN) & (t.val_mcc_mean >= MCC_MIN)
    t["g_cfg_mean_minus_sd"] = ((t.val_f1_mean - t.val_f1_sd >= F1_MIN)
                                & (t.val_mcc_mean - t.val_mcc_sd >= MCC_MIN))
    t["g_cfg_ci_lo"] = (t.val_f1_ci_lo >= F1_MIN) & (t.val_mcc_ci_lo >= MCC_MIN)
    # Sensitivity only: the same mean gate on F1/MCC at the threshold calibrated on
    # validation (optimistic: the threshold maximises val F1 on the same data).
    t["g_cfg_mean_cal"] = (t.val_f1_cal_mean >= F1_MIN) & (t.val_mcc_cal_mean >= MCC_MIN)
    t["_o"] = (t.scenario.map({s: i for i, s in enumerate(SCENS)}) * 100
               + t.arch.map({a: i for i, a in enumerate(ARCHS)}) * 10
               + t.balancing.map({b: i for i, b in enumerate(BALS)}))
    return t.sort_values("_o").drop(columns="_o").reset_index(drop=True)


GATES = ["g_model_any", "g_model_majority", "g_model_all", "g_cfg_mean",
         "g_cfg_mean_minus_sd", "g_cfg_ci_lo", "g_cfg_mean_cal"]


def gate_summary(cfg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for g in GATES:
        r = {"criterio": g, "total_48": int(cfg[g].sum()),
             "sin_1:1_36": int(cfg[cfg.scenario != "1:1"][g].sum())}
        for a in ARCHS:
            r[a] = int(cfg[cfg.arch == a][g].sum())
        for s in SCENS:
            r[f"esc_{s}"] = int(cfg[cfg.scenario == s][g].sum())
        rows.append(r)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 3. hipótesis con la configuración como unidad
# ─────────────────────────────────────────────────────────────────────────────

def stability_units(stab_csv: Path, cfg: pd.DataFrame) -> pd.DataFrame:
    df = A.load_rows(stab_csv, "stability_primary")
    g = df.groupby(["explainer", "scenario", "arch", "balancing"], as_index=False).agg(
        y=("y", "mean"), y_sd=("y", "std"), n_seeds=("seed", "nunique"),
        val_pr_auc=("val_pr_auc", "mean"))
    keep = ["scenario", "arch", "balancing"] + GATES
    return g.merge(cfg[keep], on=["scenario", "arch", "balancing"], how="left")


def friedman_exact(piv: pd.DataFrame, n_mc: int = 200_000, seed: int = 0) -> dict:
    """Permutation (exact when feasible) p-value of Friedman's statistic: under H0 the
    values of each block are exchangeable among treatments, so the within-block ranks are
    permuted. With 2-5 complete blocks the chi-square approximation of scipy is poor."""
    comp = piv.dropna()
    n, k = comp.shape
    if k < 3 or n < 2:
        return {"p_exact": np.nan, "method": "no aplicable", "n_blocks": n, "k": k}
    R = np.vstack([stats.rankdata(r) for r in comp.values])
    obs = np.sum(R.sum(axis=0) ** 2)
    perms = np.array(list(permutations(range(k))))
    n_total = len(perms) ** n
    if n_total <= 2_000_000:
        idx = np.array(list(product(range(len(perms)), repeat=n)))
        sums = np.zeros((len(idx), k))
        for b in range(n):
            sums += R[b][perms[idx[:, b]]]
        method = f"exacto ({n_total} permutaciones)"
    else:
        rng = np.random.default_rng(seed)
        sums = np.zeros((n_mc, k))
        for b in range(n):
            sums += R[b][perms[rng.integers(0, len(perms), n_mc)]]
        method = f"Monte Carlo ({n_mc})"
    stat = np.sum(sums ** 2, axis=1)
    return {"p_exact": float(np.mean(stat >= obs - 1e-9)), "method": method,
            "n_blocks": n, "k": k}


def exact_friedman_rows(units: pd.DataFrame, variants: list) -> pd.DataFrame:
    rows = []
    for name, *_rest, min_support in variants:
        for ex in EXPLAINERS:
            u = units[(units["filter"] == name) & (units.explainer == ex)]
            if u.empty:
                continue
            um = u[u.scenario != "native_size_ctrl"]
            f1 = friedman_exact(um.pivot_table(index=["arch", "balancing"],
                                               columns="scenario", values="y"))
            sup = u.groupby("arch").size()
            u2 = u[u.arch.isin(sup[sup >= min_support].index)]
            f2 = friedman_exact(u2.pivot_table(index=["scenario", "balancing"],
                                               columns="arch", values="y"))
            u3 = u[u.scenario != "1:1"]
            f3 = friedman_exact(u3.pivot_table(index=["scenario", "arch"],
                                               columns="balancing", values="y"))
            for hyp, f in (("H1", f1), ("H2", f2), ("H3", f3)):
                rows.append({"filter": name, "explainer": ex, "hypothesis": hyp, **f})
    return pd.DataFrame(rows)


def run_hypotheses(stab_csv: Path, cfg: pd.DataFrame, out: Path):
    u = stability_units(stab_csv, cfg)
    # Original v4 units (per-model gate, mean over passing seeds) to check reproduction.
    orig = A.config_units(A.load_rows(stab_csv, "stability_primary"), True)
    orig["filter"] = "repro_v4_por_modelo"

    # (name, gate column, drop 1:1, title, min support per architecture)
    variants = [
        ("repro_v4_por_modelo", None, False,
         "CONTROL: reproducción del análisis v4 original (compuerta por modelo, media de las "
         "semillas que pasan)", 5),
        ("cfg_media_sin1:1", "g_cfg_mean", True,
         "PRINCIPAL: compuerta de configuración (media), sin 1:1", 5),
        ("cfg_media_sin1:1_soporte3", "g_cfg_mean", True,
         "SENSIBILIDAD: igual a la principal con soporte mínimo 3 (entra GAT)", 3),
        ("cfg_media_con1:1", "g_cfg_mean", False,
         "SENSIBILIDAD: compuerta de configuración (media), con 1:1 (ninguna 1:1 pasa)", 5),
        ("cfg_media-sd_sin1:1", "g_cfg_mean_minus_sd", True,
         "SENSIBILIDAD: compuerta conservadora (media - sd), sin 1:1", 5),
        ("cfg_ICinf_sin1:1", "g_cfg_ci_lo", True,
         "SENSIBILIDAD: compuerta conservadora (límite inferior IC 95 %), sin 1:1", 5),
        ("cfg_calibrada_sin1:1", "g_cfg_mean_cal", True,
         "SENSIBILIDAD: compuerta de configuración con F1/MCC al umbral calibrado, sin 1:1", 5),
        ("cfg_calibrada_con1:1", "g_cfg_mean_cal", False,
         "SENSIBILIDAD: compuerta de configuración con F1/MCC al umbral calibrado, con 1:1", 5),
        ("sin_compuerta_sin1:1", None, True,
         "SENSIBILIDAD: sin compuerta (36 configuraciones), sin 1:1", 5),
        ("sin_compuerta_con1:1", None, False,
         "SENSIBILIDAD: sin compuerta (48 configuraciones), con 1:1", 5),
    ]
    frames = [orig]
    for name, gate, drop11, _, _ in variants[1:]:
        d = u.copy()
        if gate:
            d = d[d[gate]]
        if drop11:
            d = d[d.scenario != "1:1"]
        frames.append(d.assign(filter=name))
    units = pd.concat(frames, ignore_index=True)
    units.to_csv(out / "configs48_units_estabilidad.csv", index=False)

    tests, regs, summary, branches = [], [], [], {}
    summary.append("Análisis de hipótesis con la CONFIGURACIÓN como unidad (reunión 04-oct). "
                   "Métrica stability_primary; y = media de las 3 semillas de la "
                   "configuración; α=0,05; margen TOST ±0,05 (método de "
                   "analyze_elliptic_v4.py). El p de Friedman que imprime analyse() es el "
                   "asintótico (χ²); el exacto por permutación está en "
                   "configs48_friedman_exacto.csv y al final de este archivo.")
    summary.append("")
    for name, _g, _d, title, ms in variants:
        a = argparse.Namespace(native="native", size_ctrl="native_size_ctrl", balanced="1:1",
                               margin=0.05, alpha=0.05, min_support=ms, bootstrap=2000,
                               boot_seed=0)
        summary.append("=" * 78)
        summary.append(f"{title}  [filter={name}; soporte mínimo {ms}]")
        summary.append("=" * 78)
        for ex in EXPLAINERS:
            branches[(ex, name)] = A.analyse(units, ex, name, a, tests, regs, summary)
    ex_tab = exact_friedman_rows(units, variants)
    ex_tab.to_csv(out / "configs48_friedman_exacto.csv", index=False)
    summary.append("=" * 78)
    summary.append("RAMAS (con el p asintótico de analyze_elliptic_v4.py)")
    for name, *_ in variants:
        for ex in EXPLAINERS:
            b = branches.get((ex, name)) or {}
            summary.append(f"  {name:26s} {ex:16s} H1: {b.get('H1', 'n/d')} | "
                           f"H2: {b.get('H2', 'n/d')} | H3: {b.get('H3', 'n/d')}")
    summary.append("")
    summary.append("FRIEDMAN EXACTO (permutación de rangos dentro de bloque)")
    for r in ex_tab.itertuples():
        summary.append(f"  {r.filter:26s} {r.explainer:16s} {r.hypothesis}: p_exacto="
                       f"{A._f(r.p_exact, 4)} bloques={r.n_blocks} k={r.k} ({r.method})")
    pd.DataFrame(tests).to_csv(out / "configs48_tests.csv", index=False)
    if regs:
        pd.concat(regs, ignore_index=True).to_csv(out / "configs48_regresion.csv", index=False)
    (out / "configs48_hipotesis.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return branches, units


# ─────────────────────────────────────────────────────────────────────────────
# 4. selección del candidato y costo
# ─────────────────────────────────────────────────────────────────────────────

def selection(models: pd.DataFrame, cfg: pd.DataFrame, units: pd.DataFrame,
              cross_seed_csv: Path) -> tuple[pd.DataFrame, dict]:
    c = cfg[cfg.scenario.isin(SEL_SCENS)].copy()
    c["combo"] = c.arch + " + " + c.balancing
    c["rank_val_pr_auc"] = c.groupby("scenario").val_pr_auc_mean.rank(ascending=False)
    piv_rank = c.pivot_table(index="combo", columns="scenario", values="rank_val_pr_auc")
    # Kendall's W of the ranking across the 3 scenarios (k = 12 combos, m = 3 "judges").
    R = piv_rank[SEL_SCENS].values  # combos x scenarios
    m_j, k = R.shape[1], R.shape[0]
    Rs = R.sum(axis=1)
    S = ((Rs - Rs.mean()) ** 2).sum()
    W = 12 * S / (m_j ** 2 * (k ** 3 - k))
    chi2 = m_j * (k - 1) * W
    pW = 1 - stats.chi2.cdf(chi2, k - 1)

    rows = []
    sel_models = models[models.scenario.isin(SEL_SCENS)]
    stab_u = units[units["filter"] == "sin_compuerta_sin1:1"]
    cs = pd.read_csv(cross_seed_csv) if cross_seed_csv.exists() else pd.DataFrame()
    if len(cs):
        cs["explainer"] = cs["explainer"].replace(A.ALIASES)
    for (ar, bal), g in c.groupby(["arch", "balancing"]):
        r = {"combo": f"{ar} + {bal}", "arch": ar, "balancing": bal}
        for sc in SEL_SCENS:
            gs = g[g.scenario == sc].iloc[0]
            r[f"{sc}_val_pr_auc"] = gs.val_pr_auc_mean
            r[f"{sc}_val_pr_auc_ci_lo"] = gs.val_pr_auc_ci_lo
            r[f"{sc}_val_pr_auc_ci_hi"] = gs.val_pr_auc_ci_hi
            r[f"{sc}_rank"] = gs.rank_val_pr_auc
            r[f"{sc}_gate_cfg"] = bool(gs.g_cfg_mean)
        r["rank_mean"] = np.mean([r[f"{s}_rank"] for s in SEL_SCENS])
        r["rank_worst"] = max(r[f"{s}_rank"] for s in SEL_SCENS)
        mm = sel_models[(sel_models.arch == ar) & (sel_models.balancing == bal)]
        for col in ("val_pr_auc", "val_f1", "val_mcc", "val_roc_auc", "test_pr_auc",
                    "test_roc_auc"):
            m, sd, lo, hi, n = tci(mm[col])
            r[f"pooled9_{col}"], r[f"pooled9_{col}_sd"] = m, sd
            r[f"pooled9_{col}_ci_lo"], r[f"pooled9_{col}_ci_hi"] = lo, hi
        r["pooled9_n"] = len(mm)
        # seed-to-seed spread inside a configuration, averaged over the 3 scenarios
        r["sd_intra_val_pr_auc"] = g.val_pr_auc_sd.mean()
        r["n_scen_gate_cfg"] = int(g.g_cfg_mean.sum())
        r["n_models_pass"] = int(g.n_pass_model.sum())
        for ex in EXPLAINERS:
            su = stab_u[(stab_u.explainer == ex) & (stab_u.arch == ar)
                        & (stab_u.balancing == bal)]
            r[f"estab_{ex}"] = su.y.mean() if len(su) else np.nan
            if len(cs):
                cc = cs[(cs.explainer == ex) & (cs.arch == ar) & (cs.balancing == bal)
                        & cs.scenario.isin(SEL_SCENS)]
                r[f"cross_seed_{ex}"] = cc.cs_primary.mean() if len(cc) else np.nan
        rows.append(r)
    sel = pd.DataFrame(rows).sort_values(["rank_mean", "pooled9_val_pr_auc"],
                                         ascending=[True, False]).reset_index(drop=True)
    return sel, {"kendall_W": W, "chi2": chi2, "p": pW, "k": k, "m": m_j}


def cost_table(models: pd.DataFrame) -> pd.DataFrame:
    """Per configuration: seconds per final-training epoch (seeds 43/44, no Optuna), the
    Optuna part of seed 42 (= its duration minus its final training at that rate) and the
    resulting seconds per trial."""
    rows = []
    for (sc, ar, bal), g in models.groupby(["scenario", "arch", "balancing"]):
        g42 = g[g.seed == 42].iloc[0]
        rep = g[g.seed != 42]
        sec_per_epoch = (rep.duration_s / rep.epochs_run).mean()
        final42 = sec_per_epoch * g42.epochs_run
        optuna42 = max(g42.duration_s - final42, 0.0)
        rows.append({"scenario": sc, "arch": ar, "balancing": bal,
                     "dur_s42": g42.duration_s, "epochs_s42": g42.epochs_run,
                     "dur_s43_s44_mean": rep.duration_s.mean(),
                     "epochs_s43_s44_mean": rep.epochs_run.mean(),
                     "sec_per_epoch_final": sec_per_epoch,
                     "optuna8_s_est": optuna42,
                     "sec_per_trial_est": optuna42 / N_TRIALS_V4,
                     "sec_per_trial_max": sec_per_epoch * TRIAL_EPOCHS,
                     "final_train_s_mean": g.duration_s[g.seed != 42].mean()})
    return pd.DataFrame(rows)


def project_cost(cost: pd.DataFrame, arch: str, bal: str, n_trials=100, n_seeds=3) -> dict:
    c = cost[(cost.arch == arch) & (cost.balancing == bal) & cost.scenario.isin(SEL_SCENS)]
    est = (c.sec_per_trial_est * n_trials).sum()
    upper = (c.sec_per_trial_max * n_trials).sum()
    final = (c.final_train_s_mean * n_seeds).sum()
    final_max = (c.sec_per_epoch_final * 150 * n_seeds).sum()
    return {"combo": f"{arch} + {bal}", "n_scen": len(c), "optuna_s_est": est,
            "optuna_s_max": upper, "final_s_est": final, "final_s_max": final_max,
            "total_h_est": (est + final) / 3600, "total_h_max": (upper + final_max) / 3600}


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--models-dir", default=str(ROOT / "results_models_v4"))
    p.add_argument("--stability-csv", default=str(ROOT / "results_v4/elliptic_v4_stability.csv"))
    p.add_argument("--cross-seed-csv",
                   default=str(ROOT / "results_v4/elliptic_v4_cross_seed_sinfiltro.csv"))
    p.add_argument("--out-dir", default=str(ROOT / "results_v4/reunion_0410/configs48"))
    p.add_argument("--roc", action="store_true",
                   help="recalcular ROC-AUC en CPU desde los checkpoints (cache)")
    p.add_argument("--threads", type=int, default=16)
    a = p.parse_args(argv)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    roc_csv = out / "roc_auc_cpu.csv"
    if a.roc:
        compute_roc(Path(a.models_dir), roc_csv, a.threads)

    models = load_models(Path(a.models_dir), roc_csv)
    models.to_csv(out / "configs48_modelos.csv", index=False)
    cfg = config_table(models)
    cfg.to_csv(out / "configs48_metricas.csv", index=False)
    gs = gate_summary(cfg)
    gs.to_csv(out / "configs48_compuerta_resumen.csv", index=False)
    disagree = cfg[cfg.n_pass_model.isin([1, 2])][
        ["scenario", "arch", "balancing", "n_pass_model", "seeds_pass",
         "val_f1_s42", "val_f1_s43", "val_f1_s44", "val_mcc_s42", "val_mcc_s43",
         "val_mcc_s44", "val_f1_mean", "val_f1_sd", "val_mcc_mean", "val_mcc_sd"] + GATES]
    disagree.to_csv(out / "configs48_compuerta_desacuerdo.csv", index=False)

    branches, units = run_hypotheses(Path(a.stability_csv), cfg, out)
    sel, kw = selection(models, cfg, units, Path(a.cross_seed_csv))
    sel.to_csv(out / "configs48_seleccion.csv", index=False)
    cost = cost_table(models)
    cost.to_csv(out / "configs48_costo_por_config.csv", index=False)
    proj = pd.DataFrame([project_cost(cost, r.arch, r.balancing) for r in sel.itertuples()])
    proj.to_csv(out / "configs48_costo_proyectado.csv", index=False)

    print(gs.to_string(index=False))
    print("\nDesacuerdo entre semillas:", len(disagree))
    print(f"Kendall W ranking val PR-AUC entre {SEL_SCENS}: W={kw['kendall_W']:.3f} "
          f"chi2={kw['chi2']:.2f} p={kw['p']:.4g}")
    (out / "configs48_kendall.json").write_text(json.dumps(kw, indent=2), encoding="utf-8")
    print(sel[["combo", "native_rank", "1:10_rank", "1:10_os_rank", "rank_mean",
               "pooled9_val_pr_auc", "n_scen_gate_cfg"]].to_string(index=False))
    print(proj.to_string(index=False))
    if "val_pr_auc_absdiff_cpu" in models:
        print("max |PR-AUC meta - CPU|: val "
              f"{models.val_pr_auc_absdiff_cpu.max():.2e} test "
              f"{models.test_pr_auc_absdiff_cpu.max():.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
