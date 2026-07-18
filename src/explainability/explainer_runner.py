"""
Explainer runner using PyG's native torch_geometric.explain API.

Wraps GNNExplainer and PGExplainer for standardized execution
across different models and configurations.
"""

import torch
import torch.nn as nn
from torch_geometric.explain import Explainer, GNNExplainer, PGExplainer
from torch_geometric.explain import Explanation
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
from typing import Optional


class SubgraphPredictionMismatch(Exception):
    """No k-hop subgraph reproduced the full-graph prediction for the node."""
    pass


@torch.no_grad()
def full_graph_logits(model, data, device: str = "cpu"):
    """Full-graph logits [N, C], computed once per cell for the k-hop assert.

    Runs on CPU by default: a full-graph forward for attention models (GAT) can
    OOM an 8 GB GPU. This is one forward per cell, not perf-critical.
    """
    model.eval()
    orig = next(model.parameters()).device
    model.to(device)
    out = model(data.x.to(device), data.edge_index.to(device)).detach().cpu()
    model.to(orig)
    return out


def build_receptive_subgraph(model, data, node_idx, base_k, device, full_logit,
                             max_extra_hops: int = 3, atol: float = 1e-3,
                             rtol: float = 1e-2):
    """Extract the node's receptive-field subgraph and VERIFY it reproduces the
    full-graph prediction (auditor Condición 1).

    Uses an adaptive number of hops: starts at `base_k` (= num_layers, or
    num_layers*K for TAGCN) and expands up to `+max_extra_hops` — GCN/TAGCN
    normalise by degree, so boundary nodes need one extra hop for their degree to
    be correct and the prediction to match. Returns
    (sub_x, sub_edge_index, target_local, k_used). Raises
    SubgraphPredictionMismatch if no k in range reproduces the prediction.
    """
    n = data.num_nodes
    last_diff = float("nan")
    for extra in range(max_extra_hops + 1):
        k = base_k + extra
        subset, sub_ei, mapping, _ = k_hop_subgraph(
            int(node_idx), k, data.edge_index, relabel_nodes=True, num_nodes=n)
        sub_x = data.x[subset].to(device)
        sub_ei = sub_ei.to(device)
        target_local = int(mapping[0])
        model.eval()
        with torch.no_grad():
            sub_logit = model(sub_x, sub_ei)[target_local].detach().cpu()
        last_diff = float((sub_logit - full_logit).abs().max())
        if (sub_logit.argmax().item() == full_logit.argmax().item()
                and torch.allclose(sub_logit, full_logit, atol=atol, rtol=rtol)):
            return sub_x, sub_ei, target_local, k
    raise SubgraphPredictionMismatch(
        f"node {int(node_idx)}: no subgraph in k=[{base_k},{base_k + max_extra_hops}] "
        f"matched the full-graph prediction (max|Δlogit|={last_diff:.4f})")


def create_explainer(
    model: nn.Module,
    method: str = "GNNExplainer",
    epochs: int = 200,
    lr: float = 0.01,
) -> Explainer:
    """
    Create a PyG Explainer with the specified algorithm.

    Args:
        model: Trained GNN model.
        method: "GNNExplainer" or "PGExplainer".
        epochs: Training epochs for the explainer.
        lr: Learning rate for the explainer.

    Returns:
        Configured Explainer instance.
    """
    if method == "GNNExplainer":
        algorithm = GNNExplainer(epochs=epochs, lr=lr)
        explainer = Explainer(
            model=model,
            algorithm=algorithm,
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type="object",
            model_config=dict(
                mode="multiclass_classification",
                task_level="node",
                return_type="raw",
            ),
        )
    elif method == "PGExplainer":
        # CRITICAL FIXES for PyG 2.7 PGExplainer defaults (methodological findings):
        # 1. edge_size=0.05 (default) causes mode collapse (mask=0 everywhere).
        #    Verified on Cora (balanced) and Elliptic. Fix: edge_size=0.005.
        # 2. temp=[5.0, 2.0] (default) causes NaN explosions on large graphs with
        #    extreme logits (Elliptic 234k edges + class_weighting magnitude).
        #    Fix: temp=[1.0, 1.0] for numerical stability.
        # See scripts/debug_pgexplainer_hyperparams.py for empirical evidence.
        algorithm = PGExplainer(
            epochs=epochs,
            lr=min(lr, 0.003),
            edge_size=0.005,    # default 0.05 → mode collapse
            edge_ent=1.0,       # keep default entropy regularizer
            temp=[1.0, 1.0],    # default [5.0, 2.0] → NaN overflow on large graphs
        )
        explainer = Explainer(
            model=model,
            algorithm=algorithm,
            explanation_type="phenomenon",
            edge_mask_type="object",
            model_config=dict(
                mode="multiclass_classification",
                task_level="node",
                return_type="raw",
            ),
        )
    else:
        raise ValueError(f"Unknown explainer method: {method}")

    return explainer


def explain_nodes(
    explainer: Explainer,
    data: Data,
    node_indices: list,
    device: str = "cpu",
) -> list:
    """
    Generate explanations for a list of nodes.

    Args:
        explainer: Configured PyG Explainer.
        data: PyG Data object.
        node_indices: List of node indices to explain.
        device: Target device.

    Returns:
        List of Explanation objects.
    """
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    explanations = []

    # PGExplainer with 'phenomenon' explanation_type needs a target
    # PyG stores explanation_type as an enum, so compare via str()
    needs_target = "phenomenon" in str(explainer.explanation_type).lower()

    for idx in node_indices:
        kwargs = {"index": idx}
        if needs_target:
            kwargs["target"] = data.y[idx].to(device)
        explanation = explainer(x, edge_index, **kwargs)
        explanations.append(explanation)

    return explanations


def train_pgexplainer(
    explainer: Explainer,
    data: Data,
    device: str = "cpu",
) -> bool:
    """
    Train PGExplainer's parametric model on training nodes.

    Uses a rollback strategy: saves weights before each step and restores
    them if the step produces NaN loss, preventing weight corruption cascades.

    Returns True if training produced usable weights, False otherwise.
    """
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    train_indices = torch.where(data.train_mask)[0]
    explainer.algorithm.to(device)

    # CRITICAL FIX — gradient clipping via monkey-patching the optimizer.
    # PyG PGExplainer.train() has no hook for gradient clipping between .backward()
    # and .step(). On large graphs (Elliptic 234k edges) with class_weighted logits,
    # gradients explode → NaN loss. Wrap optimizer.step() to clip norm first.
    _pg_optimizer = explainer.algorithm.optimizer
    _orig_step = _pg_optimizer.step

    def _step_with_clip(*args, **kwargs):
        torch.nn.utils.clip_grad_norm_(
            explainer.algorithm.parameters(), max_norm=1.0
        )
        return _orig_step(*args, **kwargs)

    _pg_optimizer.step = _step_with_clip

    # PGExplainer loss = cross_entropy(y_hat, y).
    # target must be class indices (long scalars), NOT raw logits.
    target = data.y.to(device)

    loss = 0.0
    nan_epochs = 0
    valid_steps = 0

    for epoch in range(explainer.algorithm.epochs):
        # Save state once per epoch — rollback entire epoch on NaN (50 nodes × 100 epochs
        # = only 100 snapshots max instead of 10,000 per-step snapshots).
        epoch_state = None
        try:
            epoch_state = {k: v.clone() for k, v in explainer.algorithm.state_dict().items()}
        except Exception:
            pass  # LazyModule not yet initialized on first epoch

        epoch_loss = 0.0
        epoch_valid = 0
        had_nan = False

        for idx in train_indices[:50]:  # 100→50 nodes: faster, still representative
            step_loss = explainer.algorithm.train(
                epoch, explainer.model, x, edge_index,
                target=target,
                index=idx.item(),
            )
            if torch.is_tensor(step_loss):
                step_loss = step_loss.item()

            if not (step_loss == step_loss):  # NaN
                had_nan = True
                break  # abort this epoch

            epoch_loss += step_loss
            epoch_valid += 1

        if had_nan and epoch_state is not None:
            explainer.algorithm.load_state_dict(epoch_state)  # rollback entire epoch
            nan_epochs += 1
        elif epoch_valid > 0:
            loss += epoch_loss / epoch_valid
            valid_steps += 1

    # Final health check
    has_nan_weights = any(
        torch.isnan(p).any()
        for p in explainer.algorithm.parameters()
        if p is not None
    )

    if has_nan_weights:
        print(f"  PGExplainer: weights contain NaN after training — unusable")
        return False

    total_epochs = nan_epochs + valid_steps
    nan_pct = 100 * nan_epochs / max(total_epochs, 1)
    avg_loss = loss / max(valid_steps, 1)

    if nan_epochs > 0:
        print(f"  PGExplainer: {nan_pct:.0f}% epochs rolled back due to NaN, "
              f"loss={avg_loss:.4f} over {valid_steps} clean epochs — proceeding")
    else:
        print(f"  PGExplainer training complete (loss={avg_loss:.4f})")
    return True


@torch.no_grad()
def _predicted_illicit_mask(model, data, threshold, device="cpu"):
    """Boolean tensor [N], True where the model predicts illicit (class 1).

    The full-graph forward runs on CPU: on small (8GB) GPUs shared with a display,
    a full-graph forward for attention models (GAT) OOMs. This is one forward per
    config and not perf-critical; the explainers keep using the GPU. `device` is
    accepted for signature compatibility but the forward is forced onto CPU.
    """
    import torch.nn.functional as F
    model.eval()
    orig_dev = next(model.parameters()).device
    model.to("cpu")
    try:
        out = model(data.x.cpu(), data.edge_index.cpu())
    finally:
        model.to(orig_dev)  # restore device for the (GPU) explainers
    if threshold is None:
        pred = out.argmax(dim=-1)
    else:
        probs = F.softmax(out, dim=-1)[:, 1]
        pred = (probs >= threshold).long()
    return pred == 1


def select_explanation_nodes(
    data: Data,
    n_per_class: int = 100,
    mask_name: str = "test_mask",
    seed: int = 42,
    model=None,
    threshold=None,
    only_correct: bool = False,
    device: str = "cpu",
) -> dict:
    """
    Select nodes to explain from the mask given by `mask_name`.

    AUDIT FIX (Corrección 1): optionally condition on TRUE POSITIVES. The original
    version chose illicit nodes by ground-truth label only, so on a model that does
    not discriminate it explained mostly wrong predictions, and the measured
    "stability" was an artifact. With `only_correct=True` and a `model`, illicit
    nodes are restricted to those the model predicts illicit (TP) and licit nodes
    to true negatives (TN). Backward compatible: without `model`/`only_correct`
    the behavior is identical to before.

    Args:
        data: PyG Data object.
        n_per_class: Number of nodes per class.
        mask_name: Which mask to select from (default test_mask).
        seed: Random seed.
        model: (optional) trained model in eval mode. Required if only_correct.
        threshold: (optional) decision threshold on P(y=1). If None, argmax.
        only_correct: if True, restrict to correct predictions (needs `model`).
        device: device for the forward pass.

    Returns:
        Dict with "illicit", "licit", "all" (index lists) and "coverage"
        (how many nodes were available vs. survived the filter, per class).
    """
    import numpy as np
    rng = np.random.RandomState(seed)
    mask = getattr(data, mask_name)

    illicit_true = mask & (data.y == 1)
    licit_true = mask & (data.y == 0)

    if only_correct:
        if model is None:
            raise ValueError("only_correct=True requires passing `model`.")
        pred_illicit = _predicted_illicit_mask(model, data, threshold, device)
        illicit_sel_mask = illicit_true & pred_illicit          # true positives
        licit_sel_mask = licit_true & (~pred_illicit)           # true negatives
    else:
        illicit_sel_mask = illicit_true
        licit_sel_mask = licit_true

    illicit = torch.where(illicit_sel_mask)[0].numpy()
    licit = torch.where(licit_sel_mask)[0].numpy()

    n_illicit = min(n_per_class, len(illicit))
    n_licit = min(n_per_class, len(licit))

    selected_illicit = (rng.choice(illicit, size=n_illicit, replace=False).tolist()
                        if n_illicit > 0 else [])
    selected_licit = (rng.choice(licit, size=n_licit, replace=False).tolist()
                      if n_licit > 0 else [])

    coverage = {
        "illicit_available": int(illicit_true.sum()),
        "illicit_after_filter": int(len(illicit)),
        "illicit_selected": n_illicit,
        "licit_available": int(licit_true.sum()),
        "licit_after_filter": int(len(licit)),
        "licit_selected": n_licit,
        "only_correct": only_correct,
    }
    print(f"  Selected {n_illicit} illicit and {n_licit} licit nodes "
          f"(only_correct={only_correct}; illicit TP pool={len(illicit)}/"
          f"{int(illicit_true.sum())})")
    if only_correct and n_illicit == 0:
        print("  WARNING: 0 illicit true positives — model does not discriminate "
              "in this config; stability not applicable here.")

    return {
        "illicit": selected_illicit,
        "licit": selected_licit,
        "all": selected_illicit + selected_licit,
        "coverage": coverage,
    }
