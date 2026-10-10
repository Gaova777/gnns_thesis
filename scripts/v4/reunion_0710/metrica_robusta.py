"""¿La estabilidad medida depende del modelo o de los datos de entrada? (7-oct, validación)

El 46 % de las variables de los 30 nodos explicados vale exactamente 0 después del escalado
(son variables iguales a su mediana). Un explicador de variables casi no puede atribuirle
nada a una variable en 0, así que esas variables quedan siempre al fondo del orden, en
cualquier modelo. El Spearman sobre las 165 variables premia ese bloque común y sube aunque
los modelos no se parezcan.

Este script recalcula la estabilidad de los 4 explicadores de variables con tres medidas,
desde los puntajes ya guardados (sin GPU):

  sp_all   Spearman sobre las 165 variables (la medida de la tesis; reproduce el pipeline)
  sp_nz    Spearman solo sobre las variables del nodo distintas de 0
  jac10    Jaccard de las 10 variables más importantes

y con cada una repite las comparaciones de H1 (escenario), H2 (arquitectura) y H3 (pérdida),
la estabilidad entre semillas y el acuerdo entre escenarios.

Salidas en results_v4/reunion_0710/: robusta_modelos.csv, robusta_config.csv,
robusta_semillas.csv. Uso: uv run --frozen python scripts/v4/reunion_0710/metrica_robusta.py
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

MODELS = ROOT / "results_models_v4"
OUT = ROOT / "results_v4" / "reunion_0710"
RK = {"GNNExplainer": ROOT / "results_v4/rankings",
      "ShapleyFeatures": ROOT / "results_v4/rankings",
      "IntegratedGradients": ROOT / "results_v4_grad/rankings",
      "ExpectedGradients": ROOT / "results_v4_grad/rankings"}
SCENS = ["1:10", "1:10_os", "1:20", "native", "1:100_subil", "1:200_subil"]
MAIN = ["1:10", "1:10_os", "1:20", "native"]
APR = ["GraphSAGE", "GAT", "TAGCN"]
KEY = ["scenario", "arch", "balancing"]
MEDIDAS = ["sp_all", "sp_nz", "jac10"]


def rho(a, b):
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan
    return float(spearmanr(a, b).statistic)


def jac(a, b, k=10):
    ta = set(np.argsort(-a, kind="stable")[:k].tolist())
    tb = set(np.argsort(-b, kind="stable")[:k].tolist())
    return len(ta & tb) / len(ta | tb)


def pair(a, b, nz):
    """Las tres medidas entre dos vectores de puntajes (o de rangos medios) de un nodo."""
    return (rho(rankdata(a), rankdata(b)), rho(rankdata(a[nz]), rankdata(b[nz])), jac(a, b))


def load(run_id: str, ex: str):
    f = RK[ex] / f"{run_id.replace(':', '-')}__{ex}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    ok = np.array([not str(r) for r in z["node_reason"]]) & ~np.isnan(z["feat"]).all(axis=(1, 2))
    return {"nodes": z["nodes"], "feat": np.nan_to_num(z["feat"]), "ok": ok}


def main() -> int:
    from src.data.loader import apply_label_mode, load_elliptic
    from src.data.preprocessing import preprocess
    with contextlib.redirect_stdout(io.StringIO()):
        data = load_elliptic(root=str(ROOT / "data"))
        apply_label_mode(data, "licit_unknown")
        preprocess(data)
    nodes = json.loads((ROOT / "results_v4/explain_nodes_v4.json").read_text())["nodes"]
    NZ = {n: (data.x[n] != 0).numpy() for n in nodes}
    print(f"{len(nodes)} nodos; variables distintas de 0 por nodo: "
          f"mediana {int(np.median([v.sum() for v in NZ.values()]))} de {data.x.shape[1]}")

    metas = []
    for f in sorted(MODELS.glob("*_meta.json")):
        m = json.loads(f.read_text(encoding="utf-8"))
        if m.get("scenario") in SCENS:
            metas.append(m)

    # ── estabilidad entre réplicas, por modelo ───────────────────────────────
    rows, mean_scores = [], {}
    for m in metas:
        for ex in RK:
            d = load(m["run_id"], ex)
            if d is None:
                continue
            per = []
            ms = {}
            for i, n in enumerate(d["nodes"]):
                if not d["ok"][i]:
                    continue
                f = d["feat"][i]
                per.append(np.nanmean([pair(a, b, NZ[int(n)]) for a, b in combinations(f, 2)],
                                      axis=0))
                ms[int(n)] = np.mean([rankdata(r) for r in f], axis=0)
            mean_scores[(m["run_id"], ex)] = ms
            v = np.nanmean(per, axis=0) if per else [np.nan] * 3
            rows.append({"run_id": m["run_id"], "scenario": m["scenario"],
                         "arch": m["architecture"], "balancing": m["balancing"],
                         "seed": int(m["seed"]), "explainer": ex,
                         **dict(zip(MEDIDAS, v))})
    mod = pd.DataFrame(rows)
    mod.to_csv(OUT / "robusta_modelos.csv", index=False)
    cfg = mod.groupby(["explainer"] + KEY)[MEDIDAS].mean().reset_index()
    cfg.to_csv(OUT / "robusta_config.csv", index=False)

    # ── entre semillas del modelo y entre escenarios ──────────────────────────
    def rid(sc, a, b, s):
        return f"{sc}_{a}_{b}" + ("" if s == 42 else f"_s{s}")

    def agree(r1, r2, ex):
        a, b = mean_scores.get((r1, ex)), mean_scores.get((r2, ex))
        if not a or not b:
            return None
        return np.nanmean([pair(a[n], b[n], NZ[n]) for n in a if n in b], axis=0)

    seeds = [42, 43, 44]
    sem = []
    for ex in RK:
        for sc in SCENS:
            for a in ["GCN"] + APR:
                for b in ["none", "class_weighting", "focal_loss"]:
                    for s1, s2 in combinations(seeds, 2):
                        v = agree(rid(sc, a, b, s1), rid(sc, a, b, s2), ex)
                        if v is not None:
                            sem.append({"explainer": ex, "scenario": sc, "arch": a,
                                        "balancing": b, "tipo": "mismo escenario, otra semilla",
                                        **dict(zip(MEDIDAS, v))})
                    if sc == "native":
                        continue
                    for s1 in seeds:
                        for s2 in seeds:
                            v = agree(rid("native", a, b, s1), rid(sc, a, b, s2), ex)
                            if v is not None:
                                sem.append({"explainer": ex, "scenario": sc, "arch": a,
                                            "balancing": b,
                                            "tipo": ("otro escenario, misma semilla" if s1 == s2
                                                     else "otro escenario, otra semilla"),
                                            **dict(zip(MEDIDAS, v))})
    sem = pd.DataFrame(sem)
    sem.to_csv(OUT / "robusta_semillas.csv", index=False)

    # ── entre arquitecturas y entre explicadores (mismo escenario, pérdida y semilla) ──
    otros = []
    EXS = list(RK)
    for sc in MAIN:
        for b in ["none", "class_weighting", "focal_loss"]:
            for sd in seeds:
                for ex in EXS:
                    for a1, a2 in combinations(["GCN"] + APR, 2):
                        v = agree(rid(sc, a1, b, sd), rid(sc, a2, b, sd), ex)
                        if v is not None:
                            otros.append({"tipo": "arquitecturas", "a": a1, "b": a2,
                                          "explainer": ex, "scenario": sc, "balancing": b,
                                          **dict(zip(MEDIDAS, v))})
                for a in APR:
                    for e1, e2 in combinations(EXS, 2):
                        x, y = mean_scores.get((rid(sc, a, b, sd), e1)), \
                            mean_scores.get((rid(sc, a, b, sd), e2))
                        if x and y:
                            v = np.nanmean([pair(x[n], y[n], NZ[n]) for n in x if n in y], axis=0)
                            otros.append({"tipo": "explicadores", "a": e1, "b": e2,
                                          "explainer": "", "scenario": sc, "balancing": b,
                                          **dict(zip(MEDIDAS, v))})
    otros = pd.DataFrame(otros)
    otros.to_csv(OUT / "robusta_otros.csv", index=False)

    # ── resumen ───────────────────────────────────────────────────────────────
    pd.set_option("display.width", 220)
    present = [s for s in SCENS if s in set(mod.scenario)]
    c3 = cfg[cfg.arch.isin(APR)]
    for med in MEDIDAS:
        print(f"\n════ {med} ════")
        print("H1 · por escenario (pesos por clase, 3 arquitecturas)")
        d = c3[c3.balancing == "class_weighting"]
        print(d.pivot_table(index="explainer", columns="scenario", values=med)
              .reindex(index=list(RK), columns=present).round(3).to_string())
        print("H2 · por arquitectura (escenarios principales, pesos por clase; incluye GCN)")
        d = cfg[cfg.scenario.isin(MAIN) & (cfg.balancing == "class_weighting")]
        print(d.pivot_table(index="explainer", columns="arch", values=med)
              .reindex(index=list(RK), columns=["GCN"] + APR).round(3).to_string())
        print("H3 · por pérdida (escenarios principales, 3 arquitecturas)")
        d = c3[c3.scenario.isin(MAIN)]
        print(d.pivot_table(index="explainer", columns="balancing", values=med)
              .reindex(index=list(RK)).round(3).to_string())
        print("Acuerdo entre modelos (3 arquitecturas, pesos por clase, escenarios principales)")
        d = sem[sem.arch.isin(APR) & (sem.balancing == "class_weighting")
                & sem.scenario.isin(MAIN)]
        print(d.pivot_table(index="explainer", columns="tipo", values=med)
              .reindex(index=list(RK)).round(3).to_string())
    print("\n════ Acuerdo entre arquitecturas (pesos por clase, misma semilla) ════")
    d = otros[(otros.tipo == "arquitecturas") & (otros.balancing == "class_weighting")]
    print(d.groupby(["explainer", "a", "b"])[MEDIDAS].mean().round(3).to_string())
    print("\n════ Acuerdo entre explicadores (mismo modelo, 3 arquitecturas) ════")
    d = otros[otros.tipo == "explicadores"]
    print(d.groupby(["a", "b"])[MEDIDAS].mean().round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
