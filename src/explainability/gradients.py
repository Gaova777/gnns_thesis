"""
Gradient-based feature attributions for pipeline v4 (added after the 7-oct meeting).

PGExplainer turned out to be unsuitable for Elliptic (it ranks EDGES and 83 % of the
explained nodes have at most 2 edges; see results_v4/reunion_0410/pgexpl_gcn/REPORTE.md), so
a fourth explainer family is added to compare explainers on the node's FEATURES, where the
fraud signal is:

* ``IntegratedGradients`` (Sundararajan et al. 2017): path integral of the gradient from a
  FIXED baseline (the zero vector = the per-feature median of the train period after the
  RobustScaler) to the node's features. Deterministic: every replica returns the same
  scores, so its replica stability is 1 by construction. Its informative stability is the
  one across MODEL seeds (scripts/v4/cross_seed_stability.py).
* ``ExpectedGradients`` (Erion et al. 2021): the same integral with the baseline drawn from
  a reference distribution (train negatives) and one random point of the path per baseline.
  Stochastic, with the same sample budget per replica as ShapleyFeatures, so the 5-replica
  protocol measures something.

Both attribute the LOG-ODDS of the illicit class (logit_illicit - logit_licit) of the
explained node to ITS OWN features; the neighbours' features stay fixed, the same convention
as ``shap_runner.shapley_features_subgraph``. They run on the verified receptive subgraph.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def _margin_and_grad(model: nn.Module, x: torch.Tensor, ei: torch.Tensor, target: int,
                     states: torch.Tensor, max_nodes_per_pass: int = 60_000):
    """Log-odds of the target for each state of its feature row, and d(log-odds)/d(state).

    ``states`` [S, F]. The S states are evaluated on a disjoint union of copies of the
    subgraph (message passing never crosses copies), in chunks that keep at most
    ``max_nodes_per_pass`` nodes per forward/backward pass.
    """
    n = x.shape[0]
    per_chunk = max(1, int(max_nodes_per_pass // max(n, 1)))
    margins, grads = [], []
    for s0 in range(0, states.shape[0], per_chunk):
        st = states[s0:s0 + per_chunk].clone().requires_grad_(True)
        S = st.shape[0]
        rows = torch.arange(S, device=x.device) * n + target
        xb = x.repeat(S, 1).index_put((rows,), st)
        eib = torch.cat([ei + c * n for c in range(S)], dim=1) if ei.numel() else ei
        out = model(xb, eib)[rows]
        m = out[:, 1] - out[:, 0]
        (g,) = torch.autograd.grad(m.sum(), st)
        margins.append(m.detach())
        grads.append(g.detach())
    return torch.cat(margins), torch.cat(grads)


def integrated_gradients_subgraph(model: nn.Module, sub_x: torch.Tensor,
                                  sub_edge_index: torch.Tensor, target: int,
                                  baseline: torch.Tensor | None = None, steps: int = 50,
                                  device: str = "cpu") -> tuple[np.ndarray, float]:
    """Integrated Gradients of the target's log-odds w.r.t. its own features.

    Midpoint Riemann sum with ``steps`` points. ``baseline`` [F] defaults to zeros.
    Returns (phi [F], completeness_error) where completeness_error is
    |sum(phi) - (f(x) - f(baseline))|, which the axiom says should vanish as steps grow.
    """
    model.eval()
    x = sub_x.to(device)
    ei = sub_edge_index.to(device)
    actual = x[target].clone()
    base = torch.zeros_like(actual) if baseline is None else baseline.to(device).to(actual.dtype)
    alphas = (torch.arange(steps, device=device, dtype=actual.dtype) + 0.5) / steps
    states = base.unsqueeze(0) + alphas.unsqueeze(1) * (actual - base).unsqueeze(0)
    _, g = _margin_and_grad(model, x, ei, target, states)
    phi = (actual - base) * g.mean(dim=0)
    ends, _ = _margin_and_grad(model, x, ei, target, torch.stack([base, actual]))
    err = float((phi.sum() - (ends[1] - ends[0])).abs())
    return phi.cpu().numpy().astype(np.float64), err


def expected_gradients_subgraph(model: nn.Module, sub_x: torch.Tensor,
                                sub_edge_index: torch.Tensor, target: int,
                                references: torch.Tensor, num_samples: int = 50,
                                seed: int = 42, device: str = "cpu") -> np.ndarray:
    """Expected Gradients of the target's log-odds w.r.t. its own features.

    ``references`` [M, F] is the reference distribution (train negatives). Each of the
    ``num_samples`` draws takes one reference b and one alpha ~ U(0, 1):
    phi = mean[(x - b) * grad f(b + alpha (x - b))]. Draws depend only on ``seed``.
    """
    model.eval()
    rng = np.random.RandomState(seed)
    x = sub_x.to(device)
    ei = sub_edge_index.to(device)
    actual = x[target].clone()
    pick = rng.randint(0, references.shape[0], size=num_samples)
    alphas = torch.as_tensor(rng.uniform(0.0, 1.0, size=num_samples), dtype=actual.dtype,
                             device=device).unsqueeze(1)
    base = references[torch.as_tensor(pick, dtype=torch.long)].to(device).to(actual.dtype)
    states = base + alphas * (actual.unsqueeze(0) - base)
    _, g = _margin_and_grad(model, x, ei, target, states)
    phi = ((actual.unsqueeze(0) - base) * g).mean(dim=0)
    return phi.cpu().numpy().astype(np.float64)


def reference_pool(data, max_refs: int = 5000, seed: int = 1234) -> torch.Tensor:
    """Reference distribution for Expected Gradients: train negatives of ``data`` (the
    negative class of its label mode, native train split), a fixed subsample of at most
    ``max_refs`` rows so every model and replica draws from the same pool."""
    idx = torch.where(data.train_mask & (data.y == 0))[0].numpy()
    if len(idx) > max_refs:
        idx = np.sort(np.random.RandomState(seed).choice(idx, size=max_refs, replace=False))
    return data.x[torch.as_tensor(idx, dtype=torch.long)].clone()


# ─────────────────────────────────────────────────────────────────────────────
# Runners: same contract as v4_explain.run_shapley_features (one model, common node set)
# ─────────────────────────────────────────────────────────────────────────────

def run_integrated_gradients(model, data, nodes, base_k, full_logits, num_replicas,
                             steps: int = 100, device="cpu", on_node=None):
    """IntegratedGradients: |IG| per feature of the explained node (no edges).

    Deterministic, so the single attribution is repeated ``num_replicas`` times to keep
    the npz layout. Returns (results, extra) with the completeness error over the nodes.
    """
    import time

    from src.explainability.explainer_runner import SubgraphPredictionMismatch
    from src.explainability.v4_explain import _node_result, receptive_context

    out, errs = [], []
    for i, node in enumerate(nodes):
        t0 = time.monotonic()
        try:
            ctx = receptive_context(model, data, node, base_k, full_logits, device)
        except SubgraphPredictionMismatch as exc:
            out.append(_node_result(node, None, None, None, f"subgrafo_no_exacto: {exc}",
                                    time.monotonic() - t0))
            continue
        phi, err = integrated_gradients_subgraph(model, ctx["x"], ctx["edge_index"],
                                                 ctx["target"], None, steps, device)
        errs.append(err)
        feat = np.repeat(np.abs(phi)[None, :], num_replicas, axis=0).astype(np.float32)
        out.append(_node_result(node, feat, None, None, None, time.monotonic() - t0,
                                int(ctx["edge_index"].shape[1])))
        if on_node:
            on_node(i, node)
    extra = {"ig_completeness_err_max": float(np.max(errs)) if errs else float("nan"),
             "ig_completeness_err_median": float(np.median(errs)) if errs else float("nan")}
    return out, extra


def run_expected_gradients(model, data, nodes, base_k, full_logits, num_replicas,
                           num_samples: int = 200, references: torch.Tensor | None = None,
                           device="cpu", on_node=None):
    """ExpectedGradients: |EG| per feature of the explained node, one draw of
    ``num_samples`` (reference, alpha) pairs per replica (seed = replica_seed(r))."""
    import time

    from src.explainability.explainer_runner import SubgraphPredictionMismatch
    from src.explainability.v4_explain import _node_result, receptive_context, replica_seed

    refs = reference_pool(data) if references is None else references
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
            np.abs(expected_gradients_subgraph(model, ctx["x"], ctx["edge_index"],
                                               ctx["target"], refs, num_samples,
                                               replica_seed(r), device))
            for r in range(num_replicas)]).astype(np.float32)
        out.append(_node_result(node, feat, None, None, None, time.monotonic() - t0,
                                int(ctx["edge_index"].shape[1])))
        if on_node:
            on_node(i, node)
    return out, {}
