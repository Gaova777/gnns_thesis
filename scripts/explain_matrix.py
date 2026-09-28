"""
Explain matrix runner — pipeline v4 (Script 2 of 2). Also runs v3 configs.

For every trained checkpoint that passes the quality gate, run the three explainers
(GNNExplainer, PGExplainer, ShapleyFeatures) with ``num_replicas`` stochastic replicas
over ONE COMMON set of validation illicit nodes, save the raw scores, and compute
stability. See ``src/explainability/v4_explain.py`` for the method changes vs v3.

Inputs (``models_dir``, written by scripts/train_matrix.py):
  {run_id}_best.pt, {run_id}_meta.json  (gate: ``quality_passed``; val_metrics.pr_auc …)

Outputs (``results_dir``):
  explain_nodes_v4.json                   common node set (sampled once, then reused)
  rankings/{run_id}__{explainer}.npz      raw scores per node × replica (see NPZ below)
  elliptic_v4_stability.csv               one row per (run_id, explainer)  — columns below
  elliptic_v4_stability_pernode.csv       one row per (run_id, explainer, node)
  {logs_dir}/progress.jsonl               RunLog events, stage "explain"

CSV columns (elliptic_v4_stability.csv):
  run_id, seed, scenario, arch, balancing, explainer   identity (explainer ∈ GNNExplainer,
                                                         PGExplainer, ShapleyFeatures)
  gate_passed          quality_passed from meta.json (val F1/MCC gate)
  status               ok | gated_out | oom | error | incomplete
  reason               why a unit or its primary metric is NaN (e.g. no_aplica, oom: …)
  val_pr_auc, val_f1, val_mcc, test_pr_auc, test_f1    predictive metrics (val: argmax;
                                                         test: val-calibrated threshold)
  n_nodes              nodes explained (common set)
  n_tp                 of those, predicted illicit by this model (argmax) = true positives
  n_tp_calibrated      same with the val-calibrated threshold
  n_replicas           replicas requested
  top_k_features, top_k_edges
  spearman_full        mean over nodes of the mean pairwise Spearman (all features,
                       average ranks for ties).  NaN for PGExplainer (no_aplica)
  spearman_full_std    std over nodes
  spearman_full_tp     same, TP nodes only
  spearman_topk        v3 metric (top-k positions, rest tied) — sensitivity only
  spearman_edges       Spearman of the edge-mask scores (PGExplainer's stability metric;
                       also for GNNExplainer). NaN for ShapleyFeatures (no_aplica)
  spearman_edges_tp    same, TP nodes only
  jaccard_edges_topk   Jaccard of top-k edges, only nodes with ≥2 edges, k_eff =
                       min(k, n_edges-1)
  stability_primary    spearman_edges for PGExplainer, spearman_full otherwise
  n_measurable         nodes where stability_primary is defined
  n_measurable_feat, n_measurable_edges, n_measurable_jaccard
  spearman_full_reason, spearman_edges_reason, jaccard_reason   NaN reasons
  median_sub_edges     median edges in the node's explanation subgraph
  pg_train_n, pg_train_illicit, pg_failed_replicas   PGExplainer training sample
  seconds, sec_per_node, device, rankings_file, finished_at

NPZ (rankings/{run_id}__{explainer}.npz):
  run_id, seed, scenario, arch, balancing, explainer   (0-d strings / ints)
  nodes [n], tp [n], tp_calibrated [n], prob_illicit [n], node_reason [n] (str, "" = ok)
  seconds [n]
  feat [n, R, F] float32 (NaN = no output)            only if the explainer has features
  edge_ptr [n+1], edge_ids [ΣE] (global edge ids), edge [R, ΣE] float32
                                                       only if the explainer has edges
  Node i's edges are edge_ids[edge_ptr[i]:edge_ptr[i+1]].

Usage:
  uv run --frozen python scripts/explain_matrix.py --config configs/experiment_v4.yaml
  … --seed 43                      only models trained with seed 43
  … --arch GAT --scenario 1:1 --balancing none --explainer PGExplainer
  … --resume                       skip units whose npz + CSV row are complete
  … --include-gated                also explain models that fail the gate (sensitivity)
  … --threads 2                     torch CPU threads (default 2)
"""

import argparse
import gc
import json
import signal
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import yaml

from src.data.loader import load_elliptic
from src.data.preprocessing import preprocess
from src.explainability.explainer_runner import full_graph_logits
from src.explainability.v4_explain import (
    canonical_explainer, receptive_hops, run_gnnexplainer, run_pgexplainer,
    run_shapley_features, select_common_explain_nodes,
)
from src.monitoring.progress import RunLog
from src.stability.metrics import aggregate_nodes_v4, node_stability_v4
from src.training.trainer import build_model

warnings.filterwarnings("ignore", category=UserWarning)

EXPLAINERS = ["GNNExplainer", "PGExplainer", "ShapleyFeatures"]
HAS_FEAT = {"GNNExplainer": True, "PGExplainer": False, "ShapleyFeatures": True}
HAS_EDGE = {"GNNExplainer": True, "PGExplainer": True, "ShapleyFeatures": False}

COLUMNS = [
    "run_id", "seed", "scenario", "arch", "balancing", "explainer", "gate_passed",
    "status", "reason", "val_pr_auc", "val_f1", "val_mcc", "test_pr_auc", "test_f1",
    "n_nodes", "n_tp", "n_tp_calibrated", "n_replicas", "top_k_features", "top_k_edges",
    "spearman_full", "spearman_full_std", "spearman_full_tp", "spearman_topk",
    "spearman_edges", "spearman_edges_tp", "jaccard_edges_topk", "stability_primary",
    "n_measurable", "n_measurable_feat", "n_measurable_edges", "n_measurable_jaccard",
    "spearman_full_reason", "spearman_edges_reason", "jaccard_reason",
    "median_sub_edges", "pg_train_n", "pg_train_illicit", "pg_failed_replicas",
    "seconds", "sec_per_node", "device", "rankings_file", "finished_at",
]
PERNODE_COLUMNS = [
    "run_id", "seed", "scenario", "arch", "balancing", "explainer", "node", "tp",
    "prob_illicit", "node_reason", "n_edges", "spearman_full", "spearman_full_reason",
    "spearman_topk", "spearman_edges", "spearman_edges_reason", "jaccard_edges_topk",
    "jaccard_edges_topk_reason", "seconds",
]

_interrupted = False


def _signal_handler(signum, frame):
    global _interrupted
    _interrupted = True
    print(f"\n[SIGNAL] {signal.Signals(signum).name} — finishing current unit then exiting.",
          flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Explain matrix (v4) over trained checkpoints")
    p.add_argument("--config", required=True)
    p.add_argument("--models-dir", default=None, help="override tracking.models_dir")
    p.add_argument("--results-dir", default=None, help="override tracking.results_dir")
    p.add_argument("--logs-dir", default=None, help="override tracking.logs_dir")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda")
    p.add_argument("--seed", type=int, default=None, help="only models of this seed")
    p.add_argument("--arch", default=None)
    p.add_argument("--scenario", default=None)
    p.add_argument("--balancing", default=None)
    p.add_argument("--explainer", default=None,
                   help="GNNExplainer | PGExplainer | ShapleyFeatures (alias GNNShap)")
    p.add_argument("--include-gated", "--force", dest="include_gated", action="store_true",
                   help="also explain models that FAIL the gate (rows keep gate_passed=False)")
    p.add_argument("--resume", action="store_true",
                   help="skip units with a complete npz AND an 'ok' CSV row")
    p.add_argument("--max-hours", type=float, default=1e9)
    p.add_argument("--n-nodes", type=int, default=None, help="override explain_nodes")
    p.add_argument("--replicas", type=int, default=None, help="override num_replicas")
    p.add_argument("--threads", type=int, default=2,
                   help="torch CPU threads (default 2; 0 = torch default). The explainers "
                        "run on tiny subgraphs: with the default (all cores) on a loaded "
                        "machine GNNExplainer measured 50 s vs 0.8 s per model (3 nodes).")
    return p.parse_args(argv)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

def load_settings(config: dict, args) -> dict:
    """Normalise v4 (explainability.{gnnexplainer,pgexplainer,shapley}) and v3
    (explainability.methods list) configs into one dict."""
    ex = config.get("explainability", {}) or {}
    methods = {canonical_explainer(m["name"]): m for m in ex.get("methods", [])}
    gx = {**methods.get("GNNExplainer", {}), **(ex.get("gnnexplainer") or {})}
    pg = {**methods.get("PGExplainer", {}), **(ex.get("pgexplainer") or {})}
    sh = {**methods.get("ShapleyFeatures", {}), **(ex.get("shapley") or {})}
    st = config.get("stability", {}) or {}
    tr = config.get("tracking", {}) or {}
    gate = (config.get("analysis", {}) or {}).get("quality_gate", {}) or {}
    results_dir = Path(args.results_dir or tr.get("results_dir", "./results_v4"))
    return {
        "n_nodes": int(args.n_nodes or ex.get("explain_nodes", ex.get("nodes_per_class", 30))),
        "nodes_seed": int(ex.get("explain_nodes_seed", 1234)),
        "replicas": int(args.replicas or st.get("num_replicas", 5)),
        "top_k_features": int(st.get("top_k_features", 20)),
        "top_k_edges": int(st.get("top_k_edges", 20)),
        "gx_epochs": int(gx.get("epochs", 100)), "gx_lr": float(gx.get("lr", 0.01)),
        "pg_epochs": int(pg.get("epochs", 100)), "pg_lr": float(pg.get("lr", 0.001)),
        "pg_train_nodes": int(pg.get("train_nodes", 50)),
        "pg_train_min_illicit": int(pg.get("train_min_illicit", 25)),
        "pg_nan_abort": int(pg.get("nan_abort_threshold", 2)),
        "shap_samples": int(sh.get("num_samples", 50)),
        "results_dir": results_dir,
        "models_dir": Path(args.models_dir or tr.get("models_dir", "./results_models_v4")),
        "logs_dir": Path(args.logs_dir or tr.get("logs_dir", "./runs_v4")),
        "gate_f1": float(gate.get("f1_min", 0.30)),
        "gate_mcc": float(gate.get("mcc_min", 0.15)),
        "data": config.get("data", {}) or {},
    }


def meta_gate(meta: dict, s: dict) -> bool:
    """Gate decision of a checkpoint. train_matrix (v3 and v4) writes ``quality_passed``
    (val F1 >= f1_min AND val MCC >= mcc_min, argmax). If absent, recompute it from the
    stored val metrics with the config thresholds."""
    for key in ("quality_passed", "gate_passed"):
        if key in meta and meta[key] is not None:
            return bool(meta[key])
    f1 = meta.get("val_f1_best_epoch", (meta.get("val_metrics") or {}).get("f1"))
    mcc = meta.get("val_mcc_best_epoch", (meta.get("val_metrics") or {}).get("mcc"))
    if f1 is None or mcc is None:
        return False
    return bool(f1 >= s["gate_f1"] and mcc >= s["gate_mcc"])


def meta_perf(meta: dict) -> dict:
    vm = meta.get("val_metrics") or {}
    tm = meta.get("test_metrics") or {}
    nan = float("nan")
    return {
        "val_pr_auc": vm.get("pr_auc", meta.get("val_pr_auc", nan)),
        "val_f1": vm.get("f1", meta.get("val_f1_best_epoch", nan)),
        "val_mcc": vm.get("mcc", meta.get("val_mcc_best_epoch", meta.get("best_val_mcc", nan))),
        "test_pr_auc": tm.get("pr_auc", nan),
        "test_f1": tm.get("f1", nan),
    }


def _safe(s: str) -> str:
    return str(s).replace(":", "-").replace("/", "-")


# ─────────────────────────────────────────────────────────────────────────────
# CSV (upsert by run_id + explainer, atomic rewrite)
# ─────────────────────────────────────────────────────────────────────────────

def _upsert(path: Path, rows: list[dict], columns: list[str], key=("run_id", "explainer")):
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows, columns=columns)
    if path.exists():
        old = pd.read_csv(path)
        if len(old) and len(new):
            k_new = set(map(tuple, new[list(key)].astype(str).values))
            drop = old[list(key)].astype(str).apply(tuple, axis=1).isin(k_new)
            old = old[~drop]
        new = pd.concat([old, new], ignore_index=True) if len(old) else new
    tmp = path.with_suffix(".tmp")
    new.reindex(columns=columns).to_csv(tmp, index=False)
    tmp.replace(path)


def _completed_units(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path)
    if df.empty:
        return set()
    return set(zip(df.loc[df.status == "ok", "run_id"].astype(str),
                   df.loc[df.status == "ok", "explainer"].astype(str)))


# ─────────────────────────────────────────────────────────────────────────────
# NPZ  ⇄  per-node metrics
# ─────────────────────────────────────────────────────────────────────────────

def save_npz(path: Path, ident: dict, explainer: str, results: list[dict], R: int, F: int,
             tp, tp_cal, prob):
    n = len(results)
    arrays = {k: np.array(v) for k, v in ident.items()}
    arrays.update(
        explainer=np.array(explainer), nodes=np.array([r["node"] for r in results]),
        tp=np.asarray(tp, bool), tp_calibrated=np.asarray(tp_cal, bool),
        prob_illicit=np.asarray(prob, np.float32),
        node_reason=np.array([r["reason"] or "" for r in results]),
        seconds=np.array([r["seconds"] for r in results], np.float32),
        n_replicas=np.array(R))
    if HAS_FEAT[explainer]:
        feat = np.full((n, R, F), np.nan, np.float32)
        for i, r in enumerate(results):
            if r["feat"] is not None:
                feat[i, :r["feat"].shape[0]] = r["feat"]
        arrays["feat"] = feat
    if HAS_EDGE[explainer]:
        ptr = [0]
        ids, cols = [], []
        for r in results:
            e = r["edge"]
            if e is None or e.size == 0:
                ptr.append(ptr[-1])
                continue
            pad = np.full((R, e.shape[1]), np.nan, np.float32)
            pad[:e.shape[0]] = e
            ids.append(np.asarray(r["edge_ids"], np.int64))
            cols.append(pad)
            ptr.append(ptr[-1] + e.shape[1])
        arrays["edge_ptr"] = np.array(ptr, np.int64)
        arrays["edge_ids"] = np.concatenate(ids) if ids else np.zeros(0, np.int64)
        arrays["edge"] = (np.concatenate(cols, axis=1) if cols
                          else np.zeros((R, 0), np.float32))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def load_node_scores(npz) -> list[dict]:
    """Per node: feat [R, F] or None, edge [R, E] or None, edge_ids, reason, tp."""
    out = []
    has_f, has_e = "feat" in npz.files, "edge" in npz.files
    for i, node in enumerate(npz["nodes"]):
        d = {"node": int(node), "tp": bool(npz["tp"][i]),
             "prob_illicit": float(npz["prob_illicit"][i]),
             "reason": str(npz["node_reason"][i]) or None,
             "seconds": float(npz["seconds"][i]), "feat": None, "edge": None,
             "edge_ids": None}
        if has_f and not np.all(np.isnan(npz["feat"][i])):
            d["feat"] = npz["feat"][i]
        if has_e:
            a, b = int(npz["edge_ptr"][i]), int(npz["edge_ptr"][i + 1])
            d["edge"] = npz["edge"][:, a:b]
            d["edge_ids"] = npz["edge_ids"][a:b]
        out.append(d)
    return out


def metrics_from_npz(path: Path, explainer: str, kf: int, ke: int):
    with np.load(path, allow_pickle=False) as z:
        nodes = load_node_scores(z)
    per_node = []
    for d in nodes:
        m = node_stability_v4(d["feat"] if HAS_FEAT[explainer] else None,
                              d["edge"] if HAS_EDGE[explainer] else None, kf, ke)
        if d["reason"]:  # node-level failure (subgraph mismatch, PG training failed)
            for k in list(m):
                if k.endswith("_reason"):
                    m[k] = d["reason"]
                elif m[k] == m[k]:
                    m[k] = float("nan")
        m.update(node=d["node"], tp=d["tp"], prob_illicit=d["prob_illicit"],
                 node_reason=d["reason"] or "", seconds=d["seconds"],
                 n_edges=(0 if d["edge"] is None else int(d["edge"].shape[1])))
        per_node.append(m)
    tp = [m["tp"] for m in per_node]
    sf = aggregate_nodes_v4(per_node, "spearman_full")
    se = aggregate_nodes_v4(per_node, "spearman_edges")
    jc = aggregate_nodes_v4(per_node, "jaccard_edges_topk")
    tk = aggregate_nodes_v4(per_node, "spearman_topk")
    primary = se if explainer == "PGExplainer" else sf
    agg = {
        "spearman_full": sf["mean"], "spearman_full_std": sf["std"],
        "spearman_full_tp": aggregate_nodes_v4(per_node, "spearman_full", tp)["mean"],
        "spearman_topk": tk["mean"],
        "spearman_edges": se["mean"],
        "spearman_edges_tp": aggregate_nodes_v4(per_node, "spearman_edges", tp)["mean"],
        "jaccard_edges_topk": jc["mean"],
        "stability_primary": primary["mean"], "n_measurable": primary["n"],
        "n_measurable_feat": sf["n"], "n_measurable_edges": se["n"],
        "n_measurable_jaccard": jc["n"],
        "spearman_full_reason": sf["reason"], "spearman_edges_reason": se["reason"],
        "jaccard_reason": jc["reason"], "reason": primary["reason"],
        "median_sub_edges": float(np.median([m["n_edges"] for m in per_node]))
        if per_node else float("nan"),
        "n_nodes": len(per_node), "n_tp": int(sum(tp)),
    }
    return agg, per_node


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _load_metas(models_dir: Path) -> list[dict]:
    metas = []
    for p in sorted(models_dir.glob("*_meta.json")):
        try:
            m = json.loads(p.read_text(encoding="utf-8"))
            m["_meta_path"] = str(p)
            metas.append(m)
        except Exception as exc:  # noqa: BLE001
            print(f"  WARNING: cannot read {p.name}: {exc}")
    return metas


def _build_model(meta: dict, in_channels: int, models_dir: Path, device: str):
    bp = meta.get("best_params", {}) or {}
    kw = {}
    if meta["architecture"] == "GAT" and "heads" in bp:
        kw["heads"] = bp["heads"]
    if meta["architecture"] == "TAGCN" and "K" in bp:
        kw["K"] = bp["K"]
    model = build_model(meta["architecture"], in_channels=in_channels,
                        hidden_channels=bp.get("hidden_dim", 128),
                        num_layers=bp.get("num_layers", 2),
                        dropout=bp.get("dropout", 0.3), **kw)
    ckpt = models_dir / meta.get("checkpoint", f"{_safe(meta['run_id'])}_best.pt")
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    return model.to(device).eval()


def _is_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or (
        isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower())


def main(argv=None):
    args = parse_args(argv)
    t_start = time.time()
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    s = load_settings(config, args)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" \
        else args.device
    if args.threads:
        torch.set_num_threads(args.threads)
    print(f"Device: {device} | torch threads: {torch.get_num_threads()}")

    only_ex = canonical_explainer(args.explainer) if args.explainer else None
    explainers = [e for e in EXPLAINERS if only_ex is None or e == only_ex]
    if not explainers:
        sys.exit(f"Unknown explainer {args.explainer!r}; choose from {EXPLAINERS}")

    res_dir, rk_dir = s["results_dir"], s["results_dir"] / "rankings"
    csv_path = res_dir / "elliptic_v4_stability.csv"
    pernode_path = res_dir / "elliptic_v4_stability_pernode.csv"
    log = RunLog(s["logs_dir"], "explain")

    metas = _load_metas(s["models_dir"])
    metas = [m for m in metas
             if (args.seed is None or int(m.get("seed", -1)) == args.seed)
             and (args.arch is None or m.get("architecture") == args.arch)
             and (args.scenario is None or m.get("scenario") == args.scenario)
             and (args.balancing is None or m.get("balancing") == args.balancing)]
    if not metas:
        print(f"No checkpoints matched in {s['models_dir']}.")
        return 0

    # ── data + common node set ───────────────────────────────────────────────
    d = s["data"]
    data = load_elliptic(root=d.get("root", "./data"))
    preprocess(data, train_range=tuple(d.get("train_timesteps", (1, 34))),
               val_range=tuple(d.get("val_timesteps", (35, 42))),
               test_range=tuple(d.get("test_timesteps", (43, 49))))
    nodeset = select_common_explain_nodes(data, s["n_nodes"], s["nodes_seed"],
                                          res_dir / "explain_nodes_v4.json")
    nodes = nodeset["nodes"]
    print(f"Common node set: {len(nodes)} illicit val nodes (seed {nodeset['seed']}, "
          f"pool {nodeset['pool_size']}) -> {res_dir / 'explain_nodes_v4.json'}")

    # ── plan ─────────────────────────────────────────────────────────────────
    done = _completed_units(csv_path) if args.resume else set()
    units, gated_rows = [], []
    for m in metas:
        gate = meta_gate(m, s)
        for ex in explainers:
            if not gate and not args.include_gated:
                gated_rows.append((m, ex))
            else:
                units.append((m, ex, gate))
    todo = [(m, ex, g) for m, ex, g in units
            if not (args.resume and (m["run_id"], ex) in done
                    and (rk_dir / f"{_safe(m['run_id'])}__{ex}.npz").exists())]
    log.plan(len(todo), items=[f"{m['run_id']}__{ex}" for m, ex, _ in todo])
    print(f"{len(metas)} checkpoints | {len(units)} units to explain "
          f"({len(units) - len(todo)} already complete) | "
          f"{len({m['run_id'] for m, _ in gated_rows})} checkpoints gated out")

    # Gated-out models: one row per explainer so the analysis knows they exist.
    if gated_rows:
        rows = []
        for m, ex in gated_rows:
            rows.append({"run_id": m["run_id"], "seed": m.get("seed"),
                         "scenario": m.get("scenario"), "arch": m.get("architecture"),
                         "balancing": m.get("balancing"), "explainer": ex,
                         "gate_passed": False, "status": "gated_out",
                         "reason": "quality_gate", **meta_perf(m)})
            log.skip(f"{m['run_id']}__{ex}", "quality_gate")
        # never overwrite a completed row (e.g. an earlier --include-gated run)
        done_all = _completed_units(csv_path)
        rows = [r for r in rows if (r["run_id"], r["explainer"]) not in done_all]
        if rows:
            _upsert(csv_path, rows, COLUMNS)

    # ── run ──────────────────────────────────────────────────────────────────
    n_ok = n_fail = 0
    by_meta: dict = {}
    for m, ex, gate in todo:
        by_meta.setdefault(m["run_id"], (m, gate, []))[2].append(ex)

    for run_id, (m, gate, exs) in by_meta.items():
        if _interrupted:
            break
        if (time.time() - t_start) / 3600 > args.max_hours:
            print("  DEADLINE reached — stopping.")
            break
        ident = {"run_id": run_id, "seed": int(m.get("seed", -1)),
                 "scenario": str(m.get("scenario")), "arch": str(m.get("architecture")),
                 "balancing": str(m.get("balancing"))}
        base_row = {**ident, "gate_passed": gate, **meta_perf(m),
                    "n_replicas": s["replicas"], "top_k_features": s["top_k_features"],
                    "top_k_edges": s["top_k_edges"], "device": device}
        try:
            model = _build_model(m, data.num_node_features, s["models_dir"], device)
        except Exception as exc:  # noqa: BLE001
            for ex in exs:
                log.error(f"{run_id}__{ex}", f"load: {type(exc).__name__}: {exc}")
                _upsert(csv_path, [{**base_row, "explainer": ex, "status": "error",
                                    "reason": f"load: {exc}"}], COLUMNS)
            n_fail += len(exs)
            continue
        full_logits = full_graph_logits(model, data, device="cpu")
        prob = torch.softmax(full_logits[nodes], -1)[:, 1].numpy()
        tp = prob >= 0.5  # argmax for 2 classes; all nodes are illicit → predicted = TP
        thr = m.get("calibrated_threshold")
        tp_cal = prob >= thr if thr is not None else tp
        base_k = receptive_hops(model, m.get("best_params", {}) or {}, ident["arch"])

        for ex in exs:
            if _interrupted:
                break
            unit = f"{run_id}__{ex}"
            npz_path = rk_dir / f"{_safe(run_id)}__{ex}.npz"
            row = {**base_row, "explainer": ex, "n_nodes": len(nodes),
                   "n_tp": int(tp.sum()), "n_tp_calibrated": int(np.sum(tp_cal)),
                   "rankings_file": str(npz_path.relative_to(res_dir))}
            try:
                with log.step(unit) as rec:
                    t0 = time.monotonic()
                    extra = {}
                    if args.resume and npz_path.exists():
                        print(f"  {unit}: npz present — recomputing metrics only")
                    else:
                        beat = lambda i, nd, u=unit: log.beat(u, f"node {i + 1}/{len(nodes)}")  # noqa: E731
                        if ex == "GNNExplainer":
                            res = run_gnnexplainer(model, data, nodes, base_k, full_logits,
                                                   s["replicas"], s["gx_epochs"], s["gx_lr"],
                                                   device, on_node=beat)
                        elif ex == "ShapleyFeatures":
                            res = run_shapley_features(model, data, nodes, base_k,
                                                       full_logits, s["replicas"],
                                                       s["shap_samples"], device,
                                                       on_node=beat)
                        else:
                            res, extra = run_pgexplainer(
                                model, data, nodes, base_k, full_logits, s["replicas"],
                                s["pg_epochs"], s["pg_lr"], s["pg_train_nodes"],
                                s["pg_train_min_illicit"], device, on_node=beat,
                                nan_abort_threshold=s["pg_nan_abort"])
                        save_npz(npz_path, ident, ex, res, s["replicas"],
                                 data.num_node_features, tp, tp_cal, prob)
                    agg, per_node = metrics_from_npz(npz_path, ex, s["top_k_features"],
                                                     s["top_k_edges"])
                    secs = time.monotonic() - t0
                    row.update(agg)
                    row.update({k: extra.get(k) for k in
                                ("pg_train_n", "pg_train_illicit", "pg_failed_replicas")})
                    row.update(status="ok", seconds=round(secs, 1),
                               sec_per_node=round(secs / max(len(nodes), 1), 2),
                               finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
                    kf = s["top_k_features"]
                    rec["metrics"] = {
                        "spearman_full": agg["spearman_full"],
                        f"spearman_top{kf}": agg["spearman_topk"],
                        "spearman_edges": agg["spearman_edges"],
                        "jaccard": agg["jaccard_edges_topk"],
                        "n_nodes": agg["n_nodes"], "n_tp": agg["n_tp"],
                        "n_measurable": agg["n_measurable"], "seconds": round(secs, 1)}
                    _upsert(pernode_path, [{**ident, "explainer": ex, **pn}
                                           for pn in per_node], PERNODE_COLUMNS)
                    _upsert(csv_path, [row], COLUMNS)
                n_ok += 1
                fmt = lambda v: "nan" if v != v else f"{v:.3f}"  # noqa: E731
                print(f"  {unit}: primary={fmt(row['stability_primary'])} "
                      f"full={fmt(row['spearman_full'])} edges={fmt(row['spearman_edges'])} "
                      f"jac={fmt(row['jaccard_edges_topk'])} "
                      f"(n={row['n_measurable']}, tp={row['n_tp']}) {row['seconds']}s",
                      flush=True)
            except Exception as exc:  # noqa: BLE001 — RunLog.step already wrote 'error'
                n_fail += 1
                status = "oom" if _is_oom(exc) else "error"
                print(f"  {status.upper()} {unit}: {type(exc).__name__}: {exc}", flush=True)
                _upsert(csv_path, [{**row, "status": status,
                                    "reason": f"{status}: {str(exc)[:200]}"}], COLUMNS)
            finally:
                if device != "cpu":
                    torch.cuda.empty_cache()
                gc.collect()
        del model
        gc.collect()

    print(f"\nDone: {n_ok} ok, {n_fail} failed, {(time.time() - t_start) / 60:.1f} min. "
          f"CSV: {csv_path}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
