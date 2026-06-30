"""
Debug script — run GNNExplainer and PGExplainer on Cora (standard balanced PyG benchmark).

Purpose: determine if PGExplainer degeneration (Spearman=0) and Jaccard=1.0
observed on Elliptic are bugs in our implementation or dataset-specific findings.

Cora is:
  - Balanced-ish (7 classes, no extreme imbalance)
  - Small (2708 nodes, 10556 edges, 1433 features)
  - Standard baseline for GNN explainability literature

Runs:
  1. Train a simple 2-layer GCN on Cora
  2. For 5 nodes: run GNNExplainer x 5 replicas + PGExplainer x 5 replicas
  3. Compute Jaccard + Spearman across replicas
  4. Print raw edge_mask statistics for introspection

Expected results per literature:
  - GNNExplainer Jaccard: ~0.5-0.8
  - PGExplainer Jaccard: ~0.6-0.85
  - Spearman: > 0 (non-degenerate)
  - Edge mask distributions: non-trivial (not all-0 or all-1)

Usage:
    uv run python scripts/debug_pgexplainer.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.explain import Explainer, GNNExplainer, PGExplainer

from src.explainability.extraction import extract_subgraph, extract_feature_ranking
from src.stability.metrics import pairwise_jaccard, pairwise_spearman

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_NODES_TO_EXPLAIN = 3
N_REPLICAS = 5
TOP_K_EDGES = 20
TOP_K_FEATURES = 20


class SimpleGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def train_model(data, epochs=200):
    """Train a simple GCN on Cora."""
    print(f"Training GCN on Cora ({DEVICE})...")
    model = SimpleGCN(data.num_node_features, 64, int(data.y.max().item()) + 1).to(DEVICE)
    data = data.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                preds = model(data.x, data.edge_index).argmax(dim=-1)
                acc = (preds[data.test_mask] == data.y[data.test_mask]).float().mean().item()
            print(f"  Epoch {epoch+1}: loss={loss.item():.4f} test_acc={acc:.4f}")
            model.train()
    model.eval()
    return model, data


def make_gnnexplainer(model, edge_mask_type="object"):
    algo = GNNExplainer(epochs=100, lr=0.01)
    return Explainer(
        model=model,
        algorithm=algo,
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type=edge_mask_type,
        model_config=dict(mode="multiclass_classification", task_level="node", return_type="raw"),
    )


def make_pgexplainer(model, edge_mask_type="object"):
    algo = PGExplainer(epochs=30, lr=0.003)
    return Explainer(
        model=model,
        algorithm=algo,
        explanation_type="phenomenon",
        edge_mask_type=edge_mask_type,
        model_config=dict(mode="multiclass_classification", task_level="node", return_type="raw"),
    )


def train_pgexplainer_light(explainer, data, seed):
    """Light PGExplainer training on Cora — fewer nodes, no rollback (for diagnosis)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    explainer.algorithm.to(DEVICE)
    target = data.y.to(DEVICE)
    train_nodes = torch.where(data.train_mask)[0]
    nan_count = 0
    for epoch in range(explainer.algorithm.epochs):
        for idx in train_nodes[:50]:
            step_loss = explainer.algorithm.train(
                epoch, explainer.model, data.x.to(DEVICE), data.edge_index.to(DEVICE),
                target=target, index=idx.item(),
            )
            if torch.is_tensor(step_loss):
                step_loss = step_loss.item()
            if not (step_loss == step_loss):
                nan_count += 1
    return nan_count


def mask_stats(mask_tensor, label):
    """Print edge_mask distribution stats."""
    if mask_tensor is None:
        print(f"    {label}: mask is None")
        return
    m = mask_tensor.detach().cpu().numpy().ravel()
    pct_zero = (m < 1e-6).mean() * 100
    pct_one = (m > 0.999).mean() * 100
    print(f"    {label}: len={len(m)}, min={m.min():.4f}, max={m.max():.4f}, "
          f"mean={m.mean():.4f}, std={m.std():.4f}, "
          f"%≈0={pct_zero:.1f}%, %≈1={pct_one:.1f}%")


def run_explainer(explainer, data, node_idx, is_pg=False, verbose_first=False):
    """Run an explainer on one node, return (subgraph_set, ranking_array, mask_tensor)."""
    kwargs = {"index": node_idx}
    if is_pg:
        kwargs["target"] = data.y.to(DEVICE)
    try:
        explanation = explainer(data.x.to(DEVICE), data.edge_index.to(DEVICE), **kwargs)
    except Exception as exc:
        print(f"    ERROR calling explainer: {exc}")
        return set(), np.array([]), None
    subgraph = extract_subgraph(explanation, top_k=TOP_K_EDGES)
    ranking = extract_feature_ranking(explanation=explanation)
    em = explanation.edge_mask if hasattr(explanation, "edge_mask") else None
    if verbose_first:
        mask_stats(em, "edge_mask raw")
    return subgraph, ranking, em


def benchmark_explainer(name, make_fn, data, model, is_pg=False, edge_mask_type="object"):
    """Run an explainer on N_NODES × N_REPLICAS and return stability metrics."""
    print(f"\n{'='*70}\n{name} (edge_mask_type={edge_mask_type!r})\n{'='*70}")

    # Pick test nodes (varied labels to test class-dependency)
    test_node_idx = torch.where(data.test_mask)[0].tolist()
    np.random.seed(42)
    nodes_to_explain = np.random.choice(test_node_idx, size=N_NODES_TO_EXPLAIN, replace=False)

    per_node_metrics = []
    for n_i, node_idx in enumerate(nodes_to_explain):
        print(f"\n[Node {n_i+1}/{N_NODES_TO_EXPLAIN} — idx={node_idx}]")
        subgraphs, rankings = [], []
        for r_i in range(N_REPLICAS):
            torch.manual_seed(1000 * n_i + r_i)
            np.random.seed(1000 * n_i + r_i)
            explainer = make_fn(model, edge_mask_type=edge_mask_type)
            if is_pg:
                nan_count = train_pgexplainer_light(explainer, data, seed=1000*n_i + r_i)
                print(f"    replica {r_i+1}: PGExplainer train NaN steps={nan_count}")
            verbose = (r_i == 0)
            sg, ranking, em = run_explainer(explainer, data, int(node_idx), is_pg=is_pg, verbose_first=verbose)
            subgraphs.append(sg)
            rankings.append(ranking)

        # Metrics
        j = pairwise_jaccard(subgraphs)
        valid_rankings = [r for r in rankings if len(r) > 0]
        sp = pairwise_spearman(valid_rankings, top_k=TOP_K_FEATURES) if len(valid_rankings) >= 2 else {"mean": float("nan")}
        print(f"    -> Jaccard: mean={j['mean']:.4f} std={j['std']:.4f} min={j['min']:.4f} max={j['max']:.4f}")
        print(f"    -> Spearman: mean={sp.get('mean', float('nan')):.4f}")
        per_node_metrics.append((j["mean"], sp.get("mean", float("nan"))))

    # Aggregate
    j_means = [x[0] for x in per_node_metrics]
    sp_means = [x[1] for x in per_node_metrics if not np.isnan(x[1])]
    print(f"\n[AGGREGATE] {name}:")
    print(f"  Jaccard mean across nodes:  {np.mean(j_means):.4f}")
    print(f"  Spearman mean across nodes: {np.mean(sp_means) if sp_means else float('nan'):.4f}")
    return per_node_metrics


def main():
    print(f"Device: {DEVICE}")
    print(f"Loading Cora dataset...")
    dataset = Planetoid(root="./data/cora", name="Cora")
    data = dataset[0]
    print(f"  Nodes: {data.num_nodes}, Edges: {data.num_edges}, Features: {data.num_node_features}")
    print(f"  Classes: {int(data.y.max().item())+1}")
    print(f"  Train/Val/Test: {data.train_mask.sum().item()}/{data.val_mask.sum().item()}/{data.test_mask.sum().item()}")

    # Train model
    model, data = train_model(data, epochs=200)

    # === Diagnostic 1: GNNExplainer with edge_mask_type="object" (as in Elliptic) ===
    benchmark_explainer("GNNExplainer (object mask)", make_gnnexplainer, data, model,
                         is_pg=False, edge_mask_type="object")

    # === Diagnostic 2: GNNExplainer with edge_mask_type="attributes" ===
    # This mode gives a per-edge mask instead of single mask
    # Hypothesis: "attributes" may be less degenerate
    try:
        benchmark_explainer("GNNExplainer (attributes mask)", make_gnnexplainer, data, model,
                             is_pg=False, edge_mask_type="attributes")
    except Exception as exc:
        print(f"\n[WARN] GNNExplainer attributes mask failed: {exc}")

    # === Diagnostic 3: PGExplainer with object mask (as in Elliptic) ===
    benchmark_explainer("PGExplainer (object mask)", make_pgexplainer, data, model,
                         is_pg=True, edge_mask_type="object")

    print(f"\n{'='*70}")
    print("DIAGNOSTIC VERDICT")
    print(f"{'='*70}")
    print("""
If GNNExplainer Jaccard < 1.0 on Cora:
  -> Our Elliptic Jaccard=1.0 is Elliptic-specific (distribution shift causes degeneracy)
  -> NOT a bug in our code, but worth investigating further

If GNNExplainer Jaccard = 1.0 on Cora too:
  -> GNNExplainer with edge_mask_type='object' is ALWAYS deterministic given the model
  -> Our Elliptic result is EXPECTED, not a finding about imbalance

If PGExplainer Spearman > 0 on Cora:
  -> Our PGExplainer code WORKS, the failure is Elliptic-specific
  -> Defensible as novel finding: 'PGExplainer fails on imbalanced fraud graphs'

If PGExplainer Spearman = 0 on Cora too:
  -> There's a bug in our wrapper — investigate further
""")


if __name__ == "__main__":
    main()
