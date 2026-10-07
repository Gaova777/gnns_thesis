"""Test-retest de la estabilidad de PGExplainer (reunión 04-10, pregunta A / H3).

Llama EXACTAMENTE a la misma función del pipeline (src.explainability.v4_explain.run_pgexplainer,
mismos nodos, mismas semillas de réplica, misma configuración) varias veces sobre el mismo
checkpoint, y recalcula spearman_edges con las mismas funciones de src.stability.metrics.
Si el valor de un mismo modelo cambia entre corridas tanto como entre balanceos, la diferencia
de H3 en PGExplainer no es interpretable.

El Spearman sobre los logits del MLP (antes de la sigmoide) no se puede calcular aquí porque
run_pgexplainer devuelve la máscara; eso lo hace diag_pgexplainer.py.

Solo escribe en results_v4/reunion_0410/pgexpl_gcn/repro_pg_stability.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import json

import numpy as np
import pandas as pd
import torch
import yaml

from explain_matrix import _build_model, load_settings  # noqa: E402
from src.data.loader import apply_label_mode, load_elliptic  # noqa: E402
from src.data.preprocessing import preprocess  # noqa: E402
from src.explainability.explainer_runner import full_graph_logits  # noqa: E402
from src.explainability.v4_explain import (  # noqa: E402
    receptive_hops, run_pgexplainer, select_common_explain_nodes)
from src.stability.metrics import aggregate_nodes_v4, node_stability_v4  # noqa: E402

OUT = ROOT / "results_v4/reunion_0410/pgexpl_gcn"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="repro")
    a = ap.parse_args()
    torch.set_num_threads(2)
    with open(ROOT / "configs/experiment_v4.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    s = load_settings(cfg, argparse.Namespace(results_dir=None, models_dir=None,
                                              logs_dir=None, n_nodes=None, replicas=None))
    d = s["data"]
    data = load_elliptic(root=d.get("root", "./data"))
    apply_label_mode(data, "licit_unknown")
    preprocess(data, train_range=tuple(d.get("train_timesteps", (1, 34))),
               val_range=tuple(d.get("val_timesteps", (35, 42))),
               test_range=tuple(d.get("test_timesteps", (43, 49))))
    nodes = select_common_explain_nodes(data, s["n_nodes"], s["nodes_seed"],
                                        ROOT / "results_v4/explain_nodes_v4.json")["nodes"]
    metas = {}
    for p in sorted(s["models_dir"].glob("*_meta.json")):
        mm = json.loads(p.read_text(encoding="utf-8"))
        metas[mm["run_id"]] = mm
    orig = pd.read_csv(ROOT / "results_v4/elliptic_v4_stability.csv")
    orig = orig[orig.explainer == "PGExplainer"].set_index("run_id")
    rows = []
    for rid in a.runs:
        m = metas[rid]
        model = _build_model(m, data.num_node_features, s["models_dir"], a.device)
        full = full_graph_logits(model, data, device="cpu")
        base_k = receptive_hops(model, m.get("best_params", {}) or {}, m["architecture"])
        for rep in range(a.repeats):
            t0 = time.time()
            res, info = run_pgexplainer(model, data, nodes, base_k, full, s["replicas"],
                                        s["pg_epochs"], s["pg_lr"], s["pg_train_nodes"],
                                        s["pg_train_min_illicit"], a.device,
                                        nan_abort_threshold=s["pg_nan_abort"], log=lambda *_: None)
            per = [node_stability_v4(None, r["edge"], s["top_k_features"], s["top_k_edges"])
                   for r in res]
            agg = aggregate_nodes_v4(per, "spearman_edges")
            n_const = sum(1 for p in per if p["spearman_edges_reason"] == "vector_constante")
            rows.append({"run_id": rid, "arch": m["architecture"], "scenario": m["scenario"],
                         "balancing": m["balancing"], "repeat": rep, "device": a.device,
                         "spearman_edges": agg["mean"], "n_measurable": agg["n"],
                         "n_nodes_constant": n_const,
                         "original_csv": float(orig.loc[rid, "spearman_edges"]),
                         "original_device": orig.loc[rid, "device"],
                         "seconds": round(time.time() - t0, 1)})
            print(rows[-1], flush=True)
            pd.DataFrame(rows).to_csv(OUT / f"{a.tag}.csv", index=False)


if __name__ == "__main__":
    main()
