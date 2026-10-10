"""Análisis de la reunión del 7-oct (THE-39, THE-40, THE-41, THE-42). Sin GPU.

Junta todo lo que hay en disco y escribe en results_v4/reunion_0710/:

  config_compuerta.csv        rendimiento por configuración (media de semillas) y las dos
                              compuertas: F1/MCC con umbral 0,5 y con el umbral calibrado
  estabilidad_config.csv      estabilidad por (explicador, configuración): entre réplicas
                              (media y sd de las semillas) y entre semillas del modelo
  estabilidad_escenario.csv   lo anterior resumido por explicador y escenario, para tres
                              conjuntos de configuraciones (ver CONJUNTOS)
  acuerdo_escenarios.csv      ¿el modelo de otro escenario señala las mismas variables que
                              el nativo? Spearman por nodo entre escenarios, misma semilla
  acuerdo_semillas_nativo.csv referencia de ruido: mismo escenario (nativo), otra semilla
  acuerdo_explicadores.csv    Spearman por nodo entre explicadores, mismo modelo
  acuerdo_arquitecturas.csv   Spearman por nodo entre arquitecturas, mismo escenario, pérdida
                              y semilla (¿se apoyan en las mismas variables?)

Los explicadores de variables son GNNExplainer, ShapleyFeatures (results_v4/) e
IntegratedGradients, ExpectedGradients (results_v4_grad/). PGExplainer solo entra en las
tablas de estabilidad: ordena aristas y no se puede comparar con los demás.

Uso: uv run --frozen python scripts/v4/reunion_0710/analisis.py
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "v4"))

from cross_seed_stability import _rho, load_rankings  # noqa: E402

MODELS = ROOT / "results_models_v4"
RES = {"base": ROOT / "results_v4", "grad": ROOT / "results_v4_grad"}
OUT = ROOT / "results_v4" / "reunion_0710"

# De menos a más desbalance. 1:1 queda fuera (anexo desde el 4-oct).
SCENS = ["1:10", "1:10_os", "1:20", "native", "1:100_subil", "1:200_subil"]
MAIN = ["1:10", "1:10_os", "1:20", "native"]
STRESS = ["1:100_subil", "1:200_subil"]
ARCHS = ["GCN", "GraphSAGE", "GAT", "TAGCN"]
BALS = ["none", "class_weighting", "focal_loss"]
FEAT_EXPL = ["GNNExplainer", "ShapleyFeatures", "IntegratedGradients", "ExpectedGradients"]
EXPL = ["GNNExplainer", "PGExplainer", "ShapleyFeatures", "IntegratedGradients",
        "ExpectedGradients"]
F1_MIN, MCC_MIN = 0.30, 0.15
KEY = ["scenario", "arch", "balancing"]


def load_models() -> pd.DataFrame:
    rows = []
    for f in sorted(MODELS.glob("*_meta.json")):
        m = json.loads(f.read_text(encoding="utf-8"))
        if m.get("scenario") not in SCENS:
            continue
        vm, vc, tm = m["val_metrics"], m["val_metrics_calibrated"], m["test_metrics"]
        rows.append({
            "run_id": m["run_id"], "scenario": m["scenario"], "arch": m["architecture"],
            "balancing": m["balancing"], "seed": int(m["seed"]),
            "n_train_illicit": m["n_train_illicit"], "n_train_neg": m["n_train_licit"],
            "val_pr_auc": vm["pr_auc"], "val_f1": vm["f1"], "val_mcc": vm["mcc"],
            "val_f1_cal": vc["f1"], "val_mcc_cal": vc["mcc"],
            "test_pr_auc": tm["pr_auc"], "threshold": m["calibrated_threshold"],
            "best_epoch": m.get("best_epoch"), "gate_model": bool(m["quality_passed"]),
        })
    return pd.DataFrame(rows)


def config_table(models: pd.DataFrame) -> pd.DataFrame:
    g = models.groupby(KEY)
    t = g.agg(n_seeds=("seed", "nunique"), n_train_illicit=("n_train_illicit", "first"),
              n_train_neg=("n_train_neg", "first"),
              val_pr_auc=("val_pr_auc", "mean"), val_pr_auc_sd=("val_pr_auc", "std"),
              val_f1=("val_f1", "mean"), val_mcc=("val_mcc", "mean"),
              val_f1_cal=("val_f1_cal", "mean"), val_mcc_cal=("val_mcc_cal", "mean"),
              test_pr_auc=("test_pr_auc", "mean"), threshold=("threshold", "mean"),
              n_gate_model=("gate_model", "sum")).reset_index()
    t["gate_05"] = (t.val_f1 >= F1_MIN) & (t.val_mcc >= MCC_MIN)
    t["gate_cal"] = (t.val_f1_cal >= F1_MIN) & (t.val_mcc_cal >= MCC_MIN)
    return t


def stability_table() -> pd.DataFrame:
    """Estabilidad entre réplicas (por modelo → configuración) y entre semillas."""
    parts = []
    for res in RES.values():
        f = res / "elliptic_v4_stability.csv"
        if f.exists():
            d = pd.read_csv(f)
            parts.append(d[d.status == "ok"])
    s = pd.concat(parts, ignore_index=True)
    s["explainer"] = s.explainer.replace({"GNNShap": "ShapleyFeatures"})
    s = s[s.scenario.isin(SCENS)]
    rep = s.groupby(["explainer"] + KEY).agg(
        y=("stability_primary", "mean"), y_sd=("stability_primary", "std"),
        n_seeds=("seed", "nunique")).reset_index()

    cs_parts = []
    for name, res in RES.items():
        f = res / ("elliptic_v4_cross_seed_sinfiltro.csv" if name == "base"
                   else "elliptic_v4_cross_seed.csv")
        if f.exists():
            cs_parts.append(pd.read_csv(f))
    cs = pd.concat(cs_parts, ignore_index=True)
    cs["explainer"] = cs.explainer.replace({"GNNShap": "ShapleyFeatures"})
    cs = cs.groupby(["explainer"] + KEY).agg(cs=("cs_primary", "mean"),
                                             n_pairs=("cs_primary", "count")).reset_index()
    return rep.merge(cs, on=["explainer"] + KEY, how="left")


# Tres conjuntos de configuraciones para resumir por escenario.
def conjuntos(cfg: pd.DataFrame) -> dict:
    learn = cfg.arch.isin(["GraphSAGE", "GAT", "TAGCN"])
    return {
        # pista que pidió Cristian: pesos por clase en las tres arquitecturas que aprenden
        "pesos_por_clase_3arq": cfg[learn & (cfg.balancing == "class_weighting")],
        # las que pasan la compuerta con el umbral calibrado
        "compuerta_calibrada": cfg[cfg.gate_cal],
        # las que pasan la compuerta vigente (umbral 0,5)
        "compuerta_0.5": cfg[cfg.gate_05],
    }


def scenario_summary(stab: pd.DataFrame, cfg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, sub in conjuntos(cfg).items():
        d = stab.merge(sub[KEY], on=KEY)
        for (ex, sc), g in d.groupby(["explainer", "scenario"]):
            rows.append({"conjunto": name, "explainer": ex, "scenario": sc,
                         "n_config": len(g), "estab_replicas": g.y.mean(),
                         "estab_replicas_min": g.y.min(), "estab_replicas_max": g.y.max(),
                         "estab_semillas": g.cs.mean()})
    r = pd.DataFrame(rows)
    r["scenario"] = pd.Categorical(r.scenario, SCENS, ordered=True)
    return r.sort_values(["conjunto", "explainer", "scenario"]).reset_index(drop=True)


def _npz(res: Path, run_id: str, ex: str) -> Path:
    return res / "rankings" / f"{run_id.replace(':', '-')}__{ex}.npz"


def load_feat(run_id: str, ex: str):
    res = RES["grad"] if ex in ("IntegratedGradients", "ExpectedGradients") else RES["base"]
    f = _npz(res, run_id, ex)
    return load_rankings(f)["node_data"] if f.exists() else None


def node_agreement(a: dict, b: dict) -> float:
    """Media sobre los nodos comunes del Spearman entre los rangos medios de variables."""
    v = [_rho(a[n]["feat"], b[n]["feat"]) for n in sorted(set(a) & set(b))
         if a[n]["ok"] and b[n]["ok"]]
    v = [x for x in v if x == x]
    return float(np.mean(v)) if v else float("nan")


def run_id(scen: str, arch: str, bal: str, seed: int) -> str:
    base = f"{scen}_{arch}_{bal}"
    return base if seed == 42 else f"{base}_s{seed}"


def agreements(models: pd.DataFrame):
    have = set(models.run_id)
    seeds = sorted(models.seed.unique())
    cache: dict = {}

    def get(rid, ex):
        if (rid, ex) not in cache:
            cache[(rid, ex)] = load_feat(rid, ex) if rid in have else None
        return cache[(rid, ex)]

    esc, sem, expl = [], [], []
    for arch in ARCHS:
        for bal in BALS:
            for ex in FEAT_EXPL:
                # mismo escenario (nativo), otra semilla: el piso de ruido
                for sa, sb in combinations(seeds, 2):
                    a, b = get(run_id("native", arch, bal, sa), ex), \
                        get(run_id("native", arch, bal, sb), ex)
                    if a and b:
                        sem.append({"arch": arch, "balancing": bal, "explainer": ex,
                                    "seed_a": sa, "seed_b": sb, "rho": node_agreement(a, b)})
                # otro escenario, misma semilla
                for sc in [s for s in SCENS if s != "native"]:
                    for sd in seeds:
                        a, b = get(run_id("native", arch, bal, sd), ex), \
                            get(run_id(sc, arch, bal, sd), ex)
                        if a and b:
                            esc.append({"arch": arch, "balancing": bal, "explainer": ex,
                                        "scenario": sc, "seed": sd,
                                        "rho": node_agreement(a, b)})
    for r in models.itertuples():
        feats = {ex: get(r.run_id, ex) for ex in FEAT_EXPL}
        for ea, eb in combinations(FEAT_EXPL, 2):
            if feats[ea] and feats[eb]:
                expl.append({"run_id": r.run_id, "scenario": r.scenario, "arch": r.arch,
                             "balancing": r.balancing, "seed": r.seed, "expl_a": ea,
                             "expl_b": eb, "rho": node_agreement(feats[ea], feats[eb])})
    arq = []
    for sc in SCENS:
        for bal in BALS:
            for sd in seeds:
                for ex in FEAT_EXPL:
                    for aa, ab in combinations(ARCHS, 2):
                        a, b = get(run_id(sc, aa, bal, sd), ex), get(run_id(sc, ab, bal, sd), ex)
                        if a and b:
                            arq.append({"scenario": sc, "balancing": bal, "seed": sd,
                                        "explainer": ex, "arch_a": aa, "arch_b": ab,
                                        "rho": node_agreement(a, b)})
    return pd.DataFrame(esc), pd.DataFrame(sem), pd.DataFrame(expl), pd.DataFrame(arq)


def c(x, d=3) -> str:
    return "n/d" if x != x else f"{x:.{d}f}".replace(".", ",")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    models = load_models()
    cfg = config_table(models)
    stab = stability_table()
    cfg.to_csv(OUT / "config_compuerta.csv", index=False)
    stab.to_csv(OUT / "estabilidad_config.csv", index=False)
    summ = scenario_summary(stab, cfg)
    summ.to_csv(OUT / "estabilidad_escenario.csv", index=False)
    esc, sem, expl, arq = agreements(models)
    arq.to_csv(OUT / "acuerdo_arquitecturas.csv", index=False)
    esc.to_csv(OUT / "acuerdo_escenarios.csv", index=False)
    sem.to_csv(OUT / "acuerdo_semillas_nativo.csv", index=False)
    expl.to_csv(OUT / "acuerdo_explicadores.csv", index=False)

    present = [s for s in SCENS if s in set(models.scenario)]
    print(f"{len(models)} modelos, {len(cfg)} configuraciones, escenarios: {present}")

    print("\n== Compuerta: configuraciones que pasan por escenario ==")
    g = cfg.groupby("scenario")[["gate_05", "gate_cal"]].sum().reindex(present)
    print(g.to_string())

    print("\n== Rendimiento, pesos por clase (val PR-AUC media ± sd) ==")
    cw = cfg[cfg.balancing == "class_weighting"]
    pv = cw.pivot(index="arch", columns="scenario", values="val_pr_auc").reindex(
        index=ARCHS, columns=present)
    print(pv.map(c).to_string())

    for name in conjuntos(cfg):
        print(f"\n== Estabilidad entre réplicas por escenario · {name} ==")
        d = summ[summ.conjunto == name]
        pv = d.pivot(index="explainer", columns="scenario", values="estab_replicas")
        print(pv.reindex(index=EXPL, columns=present).map(c).to_string())
        n = d.pivot(index="explainer", columns="scenario", values="n_config")
        print("n:", n.reindex(index=EXPL[:1], columns=present).fillna(0).astype(int)
              .to_string(header=False, index=False))
        print(f"-- entre semillas · {name} --")
        pv = d.pivot(index="explainer", columns="scenario", values="estab_semillas")
        print(pv.reindex(index=EXPL, columns=present).map(c).to_string())

    learn = ["GraphSAGE", "GAT", "TAGCN"]
    if len(esc):
        print("\n== Acuerdo con el nativo (misma semilla), pesos por clase, 3 arquitecturas ==")
        d = esc[esc.arch.isin(learn) & (esc.balancing == "class_weighting")]
        pv = d.groupby(["explainer", "scenario"]).rho.mean().unstack().reindex(
            index=FEAT_EXPL, columns=[s for s in present if s != "native"])
        print(pv.map(c).to_string())
        d = sem[sem.arch.isin(learn) & (sem.balancing == "class_weighting")]
        print("-- referencia: nativo contra nativo con otra semilla --")
        print(d.groupby("explainer").rho.mean().reindex(FEAT_EXPL).map(c).to_string())
    if len(expl):
        print("\n== Acuerdo entre explicadores (mismo modelo), escenarios principales, "
              "3 arquitecturas ==")
        d = expl[expl.arch.isin(learn) & expl.scenario.isin(MAIN)]
        print(d.groupby(["expl_a", "expl_b"]).rho.agg(["mean", "std", "count"]).round(3)
              .to_string())
    if len(arq):
        print("\n== Acuerdo entre arquitecturas (mismo escenario, pérdida y semilla), "
              "pesos por clase, escenarios principales ==")
        d = arq[(arq.balancing == "class_weighting") & arq.scenario.isin(MAIN)]
        print(d.groupby(["explainer", "arch_a", "arch_b"]).rho.mean().unstack(0)
              .reindex(columns=FEAT_EXPL).map(c).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
