"""
THE-36: optimización completa del candidato top (GraphSAGE + class_weighting) frente a la
versión de la matriz (8 trials). Lee los meta.json de results_models_v4/ (matriz) y de
results_models_v4_top/ (150 trials, scripts/v4/run_optuna_top.sh), las dos variantes con
detención temprana por PR-AUC (scripts/v4/run_top_prauc.sh), los logs de Optuna de la
semilla 42 (runs_v4/TOP_train_*_s42.log) y los CSV de estabilidad de results_v4/ y
results_v4_top/. Recalcula el ROC-AUC en CPU desde los checkpoints (configs48.compute_roc).

Salidas en results_v4/reunion_0410/optuna_top/:
  comparacion_modelos.csv    una fila por modelo (versión, escenario, semilla)
  comparacion_config.csv     media ± sd de 3 semillas por versión y escenario
  convergencia.csv           mejor valor acumulado por trial y escenario
  estabilidad.csv            stability_primary por explicador, versión y escenario
  convergencia.png           figura de convergencia

Solo CPU. Uso: ~/.local/bin/uv run --frozen python scripts/v4/reunion_0410/optuna_top.py
"""

import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results_v4" / "reunion_0410" / "optuna_top"
ARCH, BAL = "GraphSAGE", "class_weighting"
SCENS = ["native", "1:10", "1:10_os", "1:20"]
# Detención temprana: F1 en argmax (protocolo de la matriz) o PR-AUC (run_top_prauc.sh).
VERS = {"matriz_8": (ROOT / "results_models_v4", ROOT / "results_v4"),
        "top_150": (ROOT / "results_models_v4_top", ROOT / "results_v4_top"),
        "matriz_8_prauc": (ROOT / "results_models_v4_mat_prauc", ROOT / "results_v4_mat_prauc"),
        "top_150_prauc": (ROOT / "results_models_v4_top_prauc", ROOT / "results_v4_top_prauc")}
EXPLAINERS = ["GNNExplainer", "PGExplainer", "ShapleyFeatures"]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C48 = _load(ROOT / "scripts/v4/reunion_0410/configs48.py", "configs48")
AEV = _load(ROOT / "scripts/v4/analyze_elliptic_v4.py", "analyze_elliptic_v4")


def models():
    rows = []
    for ver, (mdir, _) in VERS.items():
        if not mdir.exists():
            continue
        roc_csv = OUT / f"roc_auc_cpu_{ver}.csv"
        if ver == "matriz_8":
            roc = pd.read_csv(ROOT / "results_v4/reunion_0410/configs48/roc_auc_cpu.csv")
        else:
            roc = C48.compute_roc(mdir, roc_csv, 16) if not roc_csv.exists() else \
                pd.read_csv(roc_csv)
        roc = roc.set_index("run_id")
        for sc in SCENS:
            for seed in (42, 43, 44):
                fn = f"{sc.replace(':', '-')}_{ARCH}_{BAL}" + ("" if seed == 42 else f"_s{seed}")
                m = json.loads((mdir / f"{fn}_meta.json").read_text(encoding="utf-8"))
                rid = m["run_id"]
                assert m["scenario"] == sc and int(m["seed"]) == seed
                vm, tm = m["val_metrics"], m["test_metrics"]
                rows.append({
                    "version": ver, "scenario": sc, "seed": seed, "run_id": rid,
                    "val_pr_auc": vm["pr_auc"], "val_f1": vm["f1"], "val_mcc": vm["mcc"],
                    "val_roc_auc": roc.loc[rid, "val_roc_auc"],
                    "test_pr_auc": tm["pr_auc"], "test_f1": tm["f1"],
                    "test_roc_auc": roc.loc[rid, "test_roc_auc"],
                    "gate": bool(m["quality_passed"]), "best_epoch": m["best_epoch"],
                    "epochs_run": m["epochs_run"], "optuna_best": m.get("optuna_best_score"),
                    **{f"hp_{k}": v for k, v in (m.get("best_params") or {}).items()},
                })
    return pd.DataFrame(rows)


def convergence():
    pat = re.compile(r"Trial (\d+) finished with value: ([0-9.eE+-]+)")
    rows = []
    for sc in SCENS:
        log = ROOT / "runs_v4" / f"TOP_train_{sc.replace(':', '-')}_s42.log"
        vals = [(int(a), float(b)) for a, b in pat.findall(log.read_text(errors="ignore"))]
        best = -np.inf
        for t, v in sorted(vals):
            best = max(best, v)
            rows.append({"scenario": sc, "trial": t, "value": v, "best_so_far": best})
    return pd.DataFrame(rows)


def stability():
    rows = []
    for ver, (_, rdir) in VERS.items():
        if not (rdir / "elliptic_v4_cross_seed.csv").exists():
            continue
        d = AEV.load_rows(rdir / "elliptic_v4_stability.csv", "stability_primary")
        d = d[(d.arch == ARCH) & (d.balancing == BAL) & d.scenario.isin(SCENS)]
        cs = pd.read_csv(rdir / ("elliptic_v4_cross_seed_sinfiltro.csv" if ver == "matriz_8"
                                 else "elliptic_v4_cross_seed.csv"))
        cs["explainer"] = cs["explainer"].replace(AEV.ALIASES)
        cs = cs[(cs.arch == ARCH) & (cs.balancing == BAL)]
        for (ex, sc), g in d.groupby(["explainer", "scenario"]):
            c = cs[(cs.explainer == ex) & (cs.scenario == sc)]
            rows.append({"version": ver, "explainer": ex, "scenario": sc,
                         "stab_mean": g.y.mean(), "stab_sd": g.y.std(), "n_seeds": len(g),
                         "cross_seed": c.cs_primary.mean() if len(c) else np.nan})
    return pd.DataFrame(rows)


def figure(conv):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    pal = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    lab = {"native": "nativo (≈1:40)", "1:10": "1:10", "1:10_os": "1:10 con SMOTE", "1:20": "1:20"}
    for sc, col in zip(SCENS, pal):
        g = conv[conv.scenario == sc]
        ax.plot(g.trial + 1, g.best_so_far, color=col, lw=2, label=lab[sc])
        ax.scatter(g.trial + 1, g.value, color=col, s=6, alpha=0.25)
    ax.axvline(8, color="0.4", ls=":", lw=1)
    ax.text(8.5, 0.05, "8 trials (matriz)", color="0.3", fontsize=11)
    ax.set_xscale("log")
    ax.set_xlabel("trial de la búsqueda (escala log)", fontsize=12)
    ax.set_ylabel("PR-AUC de validación (mejor hasta ese trial)", fontsize=12)
    ax.set_ylim(0, 0.6)
    ax.grid(alpha=0.3, ls=":")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=11, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "convergencia.png", dpi=200)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mo = models()
    mo.to_csv(OUT / "comparacion_modelos.csv", index=False)
    met = ["val_pr_auc", "val_roc_auc", "val_f1", "val_mcc", "test_pr_auc", "test_roc_auc",
           "test_f1"]
    agg = mo.groupby(["version", "scenario"])[met].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg["n_gate"] = mo.groupby(["version", "scenario"]).gate.sum()
    agg = agg.reset_index()
    agg.to_csv(OUT / "comparacion_config.csv", index=False)
    conv = convergence()
    conv.to_csv(OUT / "convergencia.csv", index=False)
    st = stability()
    st.to_csv(OUT / "estabilidad.csv", index=False)
    figure(conv)

    pd.set_option("display.width", 200)
    print(agg[["version", "scenario", "val_pr_auc_mean", "val_pr_auc_std", "val_roc_auc_mean",
               "val_f1_mean", "test_pr_auc_mean", "test_roc_auc_mean", "n_gate"]].round(3)
          .to_string(index=False))
    for sc in SCENS:
        g = conv[conv.scenario == sc]
        fin = g.best_so_far.iloc[-1]
        t99 = int(g[g.best_so_far >= 0.99 * fin].trial.min()) + 1
        tbest = int(g[g.value == g.value.max()].trial.min()) + 1
        print(f"{sc:8s} best@8={g.best_so_far.iloc[7]:.3f} best@50={g.best_so_far.iloc[49]:.3f} "
              f"best@150={fin:.3f} trial del mejor={tbest} trial al 99 %={t99}")
    print(st.pivot_table(index=["explainer", "scenario"], columns="version",
                         values=["stab_mean", "cross_seed"]).round(3).to_string())
    print(mo[mo.version.str.startswith("top")][["version", "scenario", "seed", "best_epoch", "epochs_run"]
                                      + [c for c in mo if c.startswith("hp_")]].to_string(index=False))


if __name__ == "__main__":
    main()
