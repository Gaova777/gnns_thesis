"""
Explanation stage of pipeline v4 (Elliptic real axis).

What changed with respect to v3 (see the audit notes in the v4 task):

* COMMON NODE SET. v3 let every model explain ITS OWN 30 validation true positives, so
  architectures were compared on different nodes. v4 samples ONCE 30 illicit validation
  nodes with a fixed seed (``select_common_explain_nodes``), persists them to JSON and
  every model explains exactly those nodes. Each model explains ITS OWN PREDICTION for the
  node; whether the prediction is correct is recorded per node (``tp`` flag), so the
  analysis can restrict to true positives without changing the node set.

* SCORES, NOT ARGSORTS. Every replica yields raw importance SCORES (features and/or
  edges). Rankings with average ranks for ties are computed downstream
  (``src.stability.metrics``). Scores are saved per node and replica so stability across
  model seeds can be computed later without re-running anything.

* PGExplainer. Trained on a STRATIFIED sample of train nodes (default 50, >= 25 illicit)
  drawn with a per-replica seed, and every replica re-trains PGExplainer from scratch.
  Its only output is an EDGE mask: its stability is measured on edges
  (``spearman_edges``), never on a (non-existent) feature ranking.
  ``explanation_type="model"``: PyG 2.7's PGExplainer only accepts
  ``explanation_type="phenomenon"`` (``PGExplainer.supports``), so "model" is implemented
  exactly as PyG's Explainer defines it: the phenomenon target is the MODEL'S PREDICTION
  (argmax of the full-graph logits), not the ground-truth label.

* ShapleyFeatures (formerly mislabelled "GNNShap"): permutation-sampling Shapley values
  over the explained node's 165 FEATURES; it is not the published GNNShap (Akkas & Azad,
  2024), which attributes over EDGES. See ``shap_runner.shapley_features_subgraph``.

* EXACT SUBGRAPHS. Explanations are computed on subgraphs that are verified to
  reproduce the full-graph logits of the relevant nodes (auditor "Condición 1"), which
  also makes the CPU cost independent of the 203k-node graph.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch_geometric.utils import k_hop_subgraph

from src.explainability.explainer_runner import (
    SubgraphPredictionMismatch,
    create_explainer,
    train_pgexplainer,
)
from src.explainability.shap_runner import shapley_features_subgraph

ILLICIT = 1
LICIT = 0

# Canonical explainer labels written to v4 results. "GNNShap" is only accepted on READ.
EXPLAINER_ALIASES = {"GNNShap": "ShapleyFeatures", "ShapleyFeatures": "ShapleyFeatures",
                     "GNNExplainer": "GNNExplainer", "PGExplainer": "PGExplainer"}


def canonical_explainer(name: str) -> str:
    """Map legacy labels (``GNNShap``) to the v4 label (``ShapleyFeatures``)."""
    return EXPLAINER_ALIASES.get(str(name), str(name))


def replica_seed(replica: int) -> int:
    """Seed of replica ``r`` (same convention as v3: 42 + 17 r)."""
    return 42 + 17 * int(replica)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Common node set
# ─────────────────────────────────────────────────────────────────────────────

def _sample_val_illicit(data, n: int, seed: int) -> tuple[list[int], int]:
    pool = torch.where(data.val_mask & (data.y == ILLICIT))[0].numpy()
    pool = np.sort(pool)
    if len(pool) < n:
        raise ValueError(f"only {len(pool)} illicit validation nodes, need {n}")
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(pool, size=n, replace=False))
    return [int(i) for i in chosen], int(len(pool))


def select_common_explain_nodes(data, n: int = 30, seed: int = 1234,
                                path: Optional[str | Path] = None) -> dict:
    """Sample ONCE ``n`` illicit validation nodes (fixed ``seed``) shared by all models.

    The sample depends only on the labels and the validation mask (never on a model),
    so it is identical for every scenario/architecture/balancing/seed. If ``path``
    exists it is reused, after checking that it matches what the sampler produces
    now (same seed, same n, same pool): a mismatch raises ``ValueError`` instead of
    silently mixing node sets.

    Returns {"nodes": [...], "seed", "n", "pool_size", "label_convention", ...}.
    """
    nodes, pool_size = _sample_val_illicit(data, n, seed)
    fresh = {
        "nodes": nodes,
        "seed": int(seed),
        "n": int(n),
        "pool": "val_mask & y == 1 (illicit)",
        "pool_size": pool_size,
        "label_convention": "0 licit, 1 illicit, -1 unknown",
        "sampler": "numpy.default_rng(seed).choice(sorted(pool), n, replace=False), sorted",
    }
    if path is None:
        return fresh
    path = Path(path)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            saved = json.load(f)
        if (saved.get("nodes") != fresh["nodes"] or saved.get("seed") != fresh["seed"]
                or saved.get("pool_size") != fresh["pool_size"]):
            raise ValueError(
                f"{path} does not match the node set sampled now (seed={seed}, n={n}, "
                f"pool={pool_size}). Different data/labels or config? Refusing to mix "
                f"node sets; delete the file only if the change is intended.")
        return saved
    fresh["created"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(fresh, indent=2), encoding="utf-8")
    tmp.replace(path)
    return fresh


def sample_pg_train_nodes(data, n: int = 50, min_illicit: int = 25,
                          seed: int = 0, mask_name: str = "train_mask") -> list[int]:
    """Stratified sample of labelled training nodes for PGExplainer.

    ``min_illicit`` illicit + ``n - min_illicit`` licit nodes (fewer illicit only if the
    pool has fewer). v3 used ``train_indices[:50]``: the first 50 train nodes, which
    contained no illicit node at all.
    """
    mask = getattr(data, mask_name)
    # Only nodes that RECEIVE at least one edge. A node with in-degree 0 has an empty
    # PGExplainer hard mask: its size/entropy regularisers are means over an empty
    # tensor → NaN loss → the whole epoch is rolled back. ~30 % of Elliptic train nodes
    # have in-degree 0, so with 50 training nodes practically EVERY epoch was NaN (the
    # "~99 % NaN epochs" seen in v3 and attributed to the temperature).
    indeg = torch.bincount(data.edge_index[1], minlength=data.num_nodes)
    mask = mask & (indeg > 0)
    il = np.sort(torch.where(mask & (data.y == ILLICIT))[0].numpy())
    lic = np.sort(torch.where(mask & (data.y == LICIT))[0].numpy())
    rng = np.random.default_rng([int(seed), 7])
    n_il = min(int(min_illicit), len(il))
    n_lic = min(int(n) - n_il, len(lic))
    pick = np.concatenate([rng.choice(il, n_il, replace=False),
                           rng.choice(lic, n_lic, replace=False)])
    rng.shuffle(pick)
    return [int(i) for i in pick]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Exact subgraphs
# ─────────────────────────────────────────────────────────────────────────────

def num_mp_layers(model) -> int:
    from torch_geometric.nn import MessagePassing
    return sum(isinstance(m, MessagePassing) for m in model.modules())


def receptive_hops(model, meta_params: dict, arch: str) -> int:
    """Receptive field in hops: num_layers (× K for TAGCN)."""
    nl = int(meta_params.get("num_layers", 2))
    return nl * int(meta_params.get("K", 3)) if arch == "TAGCN" else nl


def exact_subgraph(model, data, nodes, hops: int, full_logits: torch.Tensor,
                   device: str = "cpu", atol: float = 1e-3, rtol: float = 1e-2):
    """Union of the ``hops``-hop neighbourhoods of ``nodes``; verified exact.

    Returns dict(x, edge_index, subset, edge_ids, local) where ``local`` are the
    positions of ``nodes`` in the subgraph and ``edge_ids`` the global edge ids.
    Raises SubgraphPredictionMismatch if the logits of ``nodes`` differ from the
    full-graph ones.
    """
    idx = torch.as_tensor(list(nodes), dtype=torch.long)
    subset, sub_ei, mapping, emask = k_hop_subgraph(
        idx, int(hops), data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes)
    x = data.x[subset].to(device)
    ei = sub_ei.to(device)
    model.eval()
    with torch.no_grad():
        out = model(x, ei)[mapping.to(device)].detach().cpu()
    ref = full_logits[idx]
    if not (torch.equal(out.argmax(-1), ref.argmax(-1))
            and torch.allclose(out, ref, atol=atol, rtol=rtol)):
        raise SubgraphPredictionMismatch(
            f"{len(idx)} node(s), {hops} hops: max|Δlogit|="
            f"{float((out - ref).abs().max()):.4g}")
    return {"x": x, "edge_index": ei, "subset": subset,
            "edge_ids": torch.where(emask)[0], "local": mapping}


def receptive_context(model, data, node: int, base_k: int, full_logits,
                      device: str = "cpu", max_extra_hops: int = 3) -> dict:
    """Smallest verified k-hop subgraph (k in [base_k, base_k+max_extra_hops]).

    Same rule as ``explainer_runner.build_receptive_subgraph`` (used by v3's
    GNNExplainer) but also returns the global edge ids, so edge masks from different
    replicas and different model seeds refer to the same edges.
    """
    last = None
    for extra in range(max_extra_hops + 1):
        try:
            ctx = exact_subgraph(model, data, [node], base_k + extra, full_logits, device)
            ctx["k_used"] = base_k + extra
            ctx["target"] = int(ctx["local"][0])
            return ctx
        except SubgraphPredictionMismatch as exc:
            last = exc
    raise SubgraphPredictionMismatch(f"node {node}: {last}")


def _select_edges(edge_masks: np.ndarray, edge_ids: np.ndarray):
    """Edges with a non-zero mask in at least one replica (the explainer's hard mask:
    PyG zeroes edges outside the message-passing neighbourhood). Returns ids, masks."""
    if edge_masks.size == 0:
        return edge_ids[:0], edge_masks
    keep = np.any(np.nan_to_num(edge_masks) != 0, axis=0)
    return edge_ids[keep], edge_masks[:, keep]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Explainers — each returns scores for ONE model over the common node set
# ─────────────────────────────────────────────────────────────────────────────

def _node_result(node, feat, edge_ids, edge, reason=None, seconds=0.0, n_sub_edges=None):
    return {"node": int(node), "feat": feat, "edge_ids": edge_ids, "edge": edge,
            "reason": reason, "seconds": float(seconds),
            "n_sub_edges": n_sub_edges}


def run_gnnexplainer(model, data, nodes, base_k, full_logits, num_replicas, epochs, lr,
                     device="cpu", on_node=None) -> list[dict]:
    """GNNExplainer on the verified receptive subgraph; feature and edge scores.

    Feature score of a replica = mean |node_mask| over the subgraph nodes (v3
    convention); edge score = edge_mask over the subgraph edges.
    """
    out = []
    F = data.num_node_features
    for i, node in enumerate(nodes):
        t0 = time.monotonic()
        try:
            ctx = receptive_context(model, data, node, base_k, full_logits, device)
        except SubgraphPredictionMismatch as exc:
            out.append(_node_result(node, None, None, None, f"subgrafo_no_exacto: {exc}",
                                    time.monotonic() - t0))
            continue
        E = ctx["edge_index"].shape[1]
        feat = np.full((num_replicas, F), np.nan, dtype=np.float32)
        edge = np.full((num_replicas, E), np.nan, dtype=np.float32)
        for r in range(num_replicas):
            torch.manual_seed(replica_seed(r))
            ex = create_explainer(model, "GNNExplainer", epochs=epochs, lr=lr)
            exp = ex(ctx["x"], ctx["edge_index"], index=ctx["target"])
            nm = getattr(exp, "node_mask", None)
            if nm is not None:
                feat[r] = nm.detach().abs().mean(dim=0).cpu().numpy()
            em = getattr(exp, "edge_mask", None)
            if em is not None and E > 0:
                edge[r] = em.detach().cpu().numpy()
            del ex, exp
        ids, edge_sel = _select_edges(edge, ctx["edge_ids"].numpy())
        out.append(_node_result(node, feat, ids, edge_sel, None, time.monotonic() - t0, E))
        if on_node:
            on_node(i, node)
    return out


def run_shapley_features(model, data, nodes, base_k, full_logits, num_replicas,
                         num_samples, device="cpu", on_node=None) -> list[dict]:
    """ShapleyFeatures: |Shapley value| per feature of the explained node (no edges)."""
    out = []
    for i, node in enumerate(nodes):
        t0 = time.monotonic()
        try:
            ctx = receptive_context(model, data, node, base_k, full_logits, device)
        except SubgraphPredictionMismatch as exc:
            out.append(_node_result(node, None, None, None, f"subgrafo_no_exacto: {exc}",
                                    time.monotonic() - t0))
            continue
        feat = np.stack([
            np.abs(shapley_features_subgraph(model, ctx["x"], ctx["edge_index"],
                                             ctx["target"], num_samples=num_samples,
                                             seed=replica_seed(r), device=device))
            for r in range(num_replicas)]).astype(np.float32)
        out.append(_node_result(node, feat, None, None, None, time.monotonic() - t0,
                                int(ctx["edge_index"].shape[1])))
        if on_node:
            on_node(i, node)
    return out


def run_pgexplainer(model, data, nodes, base_k, full_logits, num_replicas, epochs, lr,
                    train_nodes: int = 50, train_min_illicit: int = 25, device="cpu",
                    on_node=None, nan_abort_threshold: int = 2, log=print) -> tuple[list, dict]:
    """PGExplainer, re-trained per replica on a stratified sample; edge scores only.

    Training and inference run on exact subgraphs of radius 2H+1 (H = max(message
    passing layers, receptive hops)): the embeddings PGExplainer reads (nodes within H
    hops of the target) and the target's logits are then identical to the full-graph
    ones, verified against ``full_logits``. If verification fails, the full graph is
    used. The target is the model's prediction (explanation_type="model").
    """
    H = max(num_mp_layers(model), int(base_k))
    R = 2 * H + 1
    pred = full_logits.argmax(-1)
    info = {"pg_train_n": 0, "pg_train_illicit": 0, "pg_failed_replicas": 0,
            "pg_train_seconds": 0.0}

    # Per-node explanation contexts (built once; deterministic).
    ctxs = {}
    for node in nodes:
        try:
            ctxs[node] = exact_subgraph(model, data, [node], R, full_logits, device)
        except SubgraphPredictionMismatch:
            ctxs[node] = None  # fall back to the full graph for this node

    feats = {n: [] for n in nodes}   # replica → edge vector over the ctx edges
    t_node = {n: 0.0 for n in nodes}
    consecutive_fail = 0
    for r in range(num_replicas):
        seed = replica_seed(r)
        torch.manual_seed(seed)
        tr = sample_pg_train_nodes(data, train_nodes, train_min_illicit, seed=seed)
        info["pg_train_n"] = len(tr)
        info["pg_train_illicit"] = int((data.y[tr] == ILLICIT).sum())
        t0 = time.monotonic()
        try:
            tctx = exact_subgraph(model, data, tr, R, full_logits, device)
            tx, tei = tctx["x"], tctx["edge_index"]
            tidx = [int(v) for v in tctx["local"]]
            ttarget = pred[tctx["subset"]].to(device)
        except SubgraphPredictionMismatch:
            tx, tei = data.x.to(device), data.edge_index.to(device)
            tidx, ttarget = tr, pred.to(device)
        explainer = create_explainer(model, "PGExplainer", epochs=epochs, lr=lr,
                                     explanation_type="model")
        ok = train_pgexplainer(explainer, None, device=device, x=tx, edge_index=tei,
                               target=ttarget, train_nodes=tidx, require_clean_epoch=True)
        info["pg_train_seconds"] += time.monotonic() - t0
        if not ok:
            info["pg_failed_replicas"] += 1
            consecutive_fail += 1
            log(f"    PGExplainer replica {r}: training failed (NaN weights)")
            if consecutive_fail >= nan_abort_threshold:
                log(f"    PGExplainer: {consecutive_fail} consecutive failures — abort")
                break
            continue
        consecutive_fail = 0
        for i, node in enumerate(nodes):
            t1 = time.monotonic()
            ctx = ctxs[node]
            if ctx is not None:
                x, ei, tgt = ctx["x"], ctx["edge_index"], int(ctx["local"][0])
                target = pred[ctx["subset"]].to(device)
            else:
                x, ei, tgt = data.x.to(device), data.edge_index.to(device), int(node)
                target = pred.to(device)
            with torch.no_grad():
                exp = explainer.algorithm(model, x, ei, target=target, index=tgt)
            em = exp.edge_mask.detach().cpu().numpy().astype(np.float32)
            feats[node].append(em)
            t_node[node] += time.monotonic() - t1
            if on_node:
                on_node(i, node)

    out = []
    per_node_train = info["pg_train_seconds"] / max(len(nodes), 1)
    for node in nodes:
        ctx = ctxs[node]
        ids_all = (ctx["edge_ids"].numpy() if ctx is not None
                   else np.arange(data.edge_index.shape[1]))
        if not feats[node]:
            out.append(_node_result(node, None, None, None,
                                    "pg_entrenamiento_fallido", per_node_train))
            continue
        edge = np.stack(feats[node])
        ids, edge_sel = _select_edges(edge, ids_all)
        out.append(_node_result(node, None, ids, edge_sel, None,
                                t_node[node] + per_node_train, int(edge_sel.shape[1])))
    return out, info
