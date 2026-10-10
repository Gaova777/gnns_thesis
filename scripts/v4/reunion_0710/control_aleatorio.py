"""Control de pesos al azar (reunión del 7-oct, THE-41). CPU.

Pregunta: cuando dos explicaciones coinciden, ¿coinciden porque los modelos aprendieron lo
mismo o porque la explicación depende sobre todo de los datos del nodo? Es la prueba de
aleatorización de parámetros de Adebayo et al. (2018): se explica la misma arquitectura con
los pesos entrenados y con pesos al azar (sin entrenar) sobre los mismos 30 nodos, y se mide
cuánto coinciden las dos explicaciones. Si un explicador da casi lo mismo con pesos al azar,
su orden de variables no depende de lo que el modelo aprendió.

Para cada modelo (escenario nativo, pesos por clase, 4 arquitecturas, 3 semillas) y cada
explicador de variables se calcula, por nodo y promediado sobre los 30 nodos:

  rho_entrenado_azar   Spearman entre la explicación del modelo entrenado y la de la misma
                       arquitectura con pesos al azar
  rho_azar_azar        Spearman entre dos redes al azar distintas (cuánto coincide el
                       explicador cuando no hay nada aprendido)
  rho_x_entrenado      Spearman entre la explicación del modelo entrenado y |x| (el valor
                       absoluto de las variables del nodo): cuánto del orden lo fija la entrada
  estab_azar           estabilidad entre réplicas del explicador sobre la red al azar
  jac10_*              lo mismo con una medida que solo mira las 10 variables más
                       importantes (Jaccard de los top-10), que no se infla con la cola

Los puntajes crudos quedan en results_v4_grad/control_aleatorio/ para no repetir la corrida.

Salida: results_v4/reunion_0710/control_aleatorio.csv y un resumen por pantalla.
Uso: uv run --frozen python scripts/v4/reunion_0710/control_aleatorio.py [--seeds 42 43 44]
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import scripts.explain_matrix as em  # noqa: E402
from src.data.loader import apply_label_mode, load_elliptic  # noqa: E402
from src.data.preprocessing import preprocess  # noqa: E402
from src.explainability import gradients as gr  # noqa: E402
from src.explainability.explainer_runner import full_graph_logits  # noqa: E402
from src.explainability.v4_explain import (  # noqa: E402
    receptive_hops, run_gnnexplainer, run_shapley_features)
from src.training.trainer import build_model  # noqa: E402

MODELS = ROOT / "results_models_v4"
OUT = ROOT / "results_v4" / "reunion_0710"
ARCHS = ["GCN", "GraphSAGE", "GAT", "TAGCN"]
EXPL = ["GNNExplainer", "ShapleyFeatures", "IntegratedGradients", "ExpectedGradients"]
R = 5


def random_model(meta: dict, in_channels: int, seed: int):
    bp = meta.get("best_params", {}) or {}
    kw = {}
    if meta["architecture"] == "GAT" and "heads" in bp:
        kw["heads"] = bp["heads"]
    if meta["architecture"] == "TAGCN" and "K" in bp:
        kw["K"] = bp["K"]
    torch.manual_seed(seed)
    return build_model(meta["architecture"], in_channels=in_channels,
                       hidden_channels=bp.get("hidden_dim", 128),
                       num_layers=bp.get("num_layers", 2), dropout=bp.get("dropout", 0.3),
                       **kw).eval()


def explain(model, data, nodes, meta, refs, ex: str):
    fl = full_graph_logits(model, data, device="cpu")
    k = receptive_hops(model, meta.get("best_params", {}) or {}, meta["architecture"])
    if ex == "GNNExplainer":
        res = run_gnnexplainer(model, data, nodes, k, fl, R, 100, 0.01, "cpu")
    elif ex == "ShapleyFeatures":
        res = run_shapley_features(model, data, nodes, k, fl, R, 50, "cpu")
    elif ex == "IntegratedGradients":
        res, _ = gr.run_integrated_gradients(model, data, nodes, k, fl, R, 100, "cpu")
    else:
        res, _ = gr.run_expected_gradients(model, data, nodes, k, fl, R, 200, refs, "cpu")
    return {r["node"]: r["feat"] for r in res if r["feat"] is not None}


def mean_rank(f):
    return np.mean([rankdata(np.nan_to_num(r)) for r in f], axis=0)


def rho(a, b):
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan
    return float(spearmanr(a, b).statistic)


def agree(fa: dict, fb: dict) -> float:
    v = [rho(mean_rank(fa[n]), mean_rank(fb[n])) for n in fa if n in fb]
    return float(np.nanmean(v)) if v else np.nan


def top(f, k):
    return set(np.argsort(-mean_rank(f), kind="stable")[:k].tolist())


def jac(a: set, b: set) -> float:
    return len(a & b) / len(a | b)


def agree_top(fa: dict, fb: dict, k: int = 10) -> float:
    v = [jac(top(fa[n], k), top(fb[n], k)) for n in fa if n in fb]
    return float(np.mean(v)) if v else np.nan


def stability_top(f: dict, k: int = 10) -> float:
    v = []
    for n, m in f.items():
        tops = [set(np.argsort(-np.nan_to_num(r), kind="stable")[:k].tolist()) for r in m]
        v.append(np.mean([jac(a, b) for a, b in combinations(tops, 2)]))
    return float(np.mean(v)) if v else np.nan


def stability(f: dict) -> float:
    v = []
    for n, m in f.items():
        rk = [rankdata(np.nan_to_num(r)) for r in m]
        v.append(np.nanmean([rho(a, b) for a, b in combinations(rk, 2)]))
    return float(np.nanmean(v)) if v else np.nan


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    data = load_elliptic(root=str(ROOT / "data"))
    apply_label_mode(data, "licit_unknown")
    preprocess(data)
    nodes = json.loads((ROOT / "results_v4" / "explain_nodes_v4.json").read_text())["nodes"]
    refs = gr.reference_pool(data)
    absx = {n: data.x[n].abs().numpy() for n in nodes}
    zero = float(np.mean([(data.x[n] == 0).float().mean().item() for n in nodes]))
    print(f"{len(nodes)} nodos; fracción de variables exactamente en 0: {zero:.3f}")

    rows = []
    out = OUT / "control_aleatorio.csv"
    for arch in ARCHS:
        for seed in a.seeds:
            rid = f"native_{arch}_class_weighting" + ("" if seed == 42 else f"_s{seed}")
            meta = json.loads((MODELS / f"{rid}_meta.json").read_text(encoding="utf-8"))
            trained = em._build_model(meta, data.num_node_features, MODELS, "cpu")
            rnd_a = random_model(meta, data.num_node_features, 1000 + seed)
            rnd_b = random_model(meta, data.num_node_features, 2000 + seed)
            for ex in EXPL:
                ft = explain(trained, data, nodes, meta, refs, ex)
                fa = explain(rnd_a, data, nodes, meta, refs, ex)
                fb = explain(rnd_b, data, nodes, meta, refs, ex)
                row = {
                    "arch": arch, "seed": seed, "explainer": ex, "n_nodes": len(ft),
                    "val_pr_auc": meta["val_metrics"]["pr_auc"],
                    "rho_entrenado_azar": np.nanmean([agree(ft, fa), agree(ft, fb)]),
                    "rho_azar_azar": agree(fa, fb),
                    "rho_x_entrenado": float(np.nanmean(
                        [rho(mean_rank(ft[n]), rankdata(absx[n])) for n in ft])),
                    "rho_x_azar": float(np.nanmean(
                        [rho(mean_rank(fa[n]), rankdata(absx[n])) for n in fa])),
                    "estab_entrenado": stability(ft), "estab_azar": stability(fa),
                    "jac10_entrenado_azar": np.mean([agree_top(ft, fa), agree_top(ft, fb)]),
                    "jac10_azar_azar": agree_top(fa, fb),
                    "jac10_x_entrenado": float(np.mean(
                        [jac(top(ft[n], 10), set(np.argsort(-absx[n])[:10].tolist()))
                         for n in ft])),
                    "jac10_estab_entrenado": stability_top(ft),
                    "jac10_estab_azar": stability_top(fa),
                }
                raw = ROOT / "results_v4_grad" / "control_aleatorio"
                raw.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    raw / f"{rid}__{ex}.npz", nodes=np.array(sorted(ft)),
                    entrenado=np.stack([ft[n] for n in sorted(ft)]),
                    azar_a=np.stack([fa[n] for n in sorted(ft)]),
                    azar_b=np.stack([fb[n] for n in sorted(ft)]))
                rows.append(row)
                pd.DataFrame(rows).to_csv(out, index=False)
                print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()},
                      flush=True)
    df = pd.DataFrame(rows)
    print("\n== Por explicador (media de arquitecturas y semillas) ==")
    print(df.groupby("explainer")[["rho_entrenado_azar", "rho_azar_azar", "rho_x_entrenado",
                                   "rho_x_azar", "estab_entrenado", "estab_azar"]]
          .mean().reindex(EXPL).round(3).to_string())
    print("\n== Lo mismo con el Jaccard de las 10 variables más importantes ==")
    print(df.groupby("explainer")[[k for k in df.columns if k.startswith("jac10")]]
          .mean().reindex(EXPL).round(3).to_string())
    print("\n== rho_entrenado_azar por arquitectura ==")
    print(df.pivot_table(index="explainer", columns="arch", values="rho_entrenado_azar")
          .reindex(index=EXPL, columns=ARCHS).round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
