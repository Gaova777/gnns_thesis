"""
Explainability script for a trained GNN model.

Usage:
    uv run python scripts/run_explain.py --model GCN --explainer GNNExplainer --nodes 10
"""

import argparse
import torch
from pathlib import Path

from src.data.loader import load_elliptic
from src.data.preprocessing import preprocess
from src.training.trainer import build_model
from src.explainability.explainer_runner import (
    create_explainer, explain_nodes, select_explanation_nodes
)
from src.explainability.shap_runner import explain_nodes_shap
from src.explainability.extraction import (
    extract_subgraph, extract_feature_ranking, serialize_explanation, save_explanations
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run XAI on a trained GNN")
    parser.add_argument("--model", type=str, default="GCN")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--explainer", type=str, default="GNNExplainer",
                       choices=["GNNExplainer", "PGExplainer", "GNNShap"])
    parser.add_argument("--nodes", type=int, default=10, help="Nodes per class to explain")
    parser.add_argument("--top-k-edges", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="./results/explanations")
    return parser.parse_args()


def main():
    args = parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load data
    print("\n1. Loading data...")
    data = load_elliptic()
    preprocess(data)

    # Build and load model
    print(f"\n2. Loading model: {args.model}")
    model = build_model(args.model, data.num_node_features, args.hidden_dim, args.num_layers)

    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, weights_only=True))
        print(f"   Loaded checkpoint: {args.checkpoint}")
    else:
        print("   ⚠ No checkpoint provided — using random weights (for testing only)")

    model = model.to(device)
    model.eval()

    # Select nodes
    print(f"\n3. Selecting nodes to explain...")
    selected = select_explanation_nodes(data, n_per_class=args.nodes)

    # Run explainer
    print(f"\n4. Explaining with {args.explainer}...")
    all_serialized = []

    if args.explainer in ("GNNExplainer", "PGExplainer"):
        explainer = create_explainer(model, args.explainer)
        explanations = explain_nodes(explainer, data, selected["all"], device)

        for exp, node_idx in zip(explanations, selected["all"]):
            subgraph = extract_subgraph(exp, top_k=args.top_k_edges)
            ranking = extract_feature_ranking(explanation=exp)
            serialized = serialize_explanation(node_idx, subgraph, ranking)
            all_serialized.append(serialized)

    elif args.explainer == "GNNShap":
        shap_results = explain_nodes_shap(model, data, selected["all"], device=device)

        for result, node_idx in zip(shap_results, selected["all"]):
            ranking = result["feature_ranking"]
            serialized = serialize_explanation(
                node_idx, set(), ranking,
                shap_values=result["shap_values"],
                metadata=result["concentrations"],
            )
            all_serialized.append(serialized)

    # Save
    output_path = f"{args.output_dir}/{args.model}_{args.explainer}.json"
    save_explanations(all_serialized, output_path)
    print(f"\n✓ Done. Explained {len(all_serialized)} nodes.")


if __name__ == "__main__":
    main()
