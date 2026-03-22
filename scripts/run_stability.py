"""
Stability testing script.

Runs stochastic and perturbation tests on a trained model's explanations.

Usage:
    uv run python scripts/run_stability.py --model GCN --explainer GNNExplainer --replicas 30
"""

import argparse
import json
import torch
from pathlib import Path

from src.data.loader import load_elliptic
from src.data.preprocessing import preprocess
from src.training.trainer import build_model
from src.explainability.explainer_runner import select_explanation_nodes
from src.stability.stochastic_test import run_stochastic_test_batch
from src.stability.perturbation import run_perturbation_test
from src.stability.metrics import compute_stability_metrics, compute_perturbation_stability


def parse_args():
    parser = argparse.ArgumentParser(description="Run stability tests")
    parser.add_argument("--model", type=str, default="GCN")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--explainer", type=str, default="GNNExplainer")
    parser.add_argument("--nodes", type=int, default=10)
    parser.add_argument("--replicas", type=int, default=30)
    parser.add_argument("--noise-levels", nargs="+", type=float, default=[0.01, 0.05, 0.10])
    parser.add_argument("--top-k-edges", type=int, default=20)
    parser.add_argument("--top-k-features", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="./results/stability")
    return parser.parse_args()


def main():
    args = parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu"

    # Load data & model
    print("1. Loading data...")
    data = load_elliptic()
    preprocess(data)

    print(f"2. Loading model: {args.model}")
    model = build_model(args.model, data.num_node_features, args.hidden_dim, args.num_layers)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, weights_only=True))
    model = model.to(device)
    model.eval()

    # Select nodes
    selected = select_explanation_nodes(data, n_per_class=args.nodes)
    test_nodes = selected["illicit"][:5] + selected["licit"][:5]  # Sample

    # --- Stochastic stability ---
    print(f"\n3. Stochastic stability test ({args.replicas} replicas)...")
    stochastic_results = run_stochastic_test_batch(
        model, data, test_nodes, args.explainer,
        num_replicas=args.replicas, top_k_edges=args.top_k_edges, device=device
    )

    stochastic_metrics = []
    for result in stochastic_results:
        metrics = compute_stability_metrics(result, top_k_features=args.top_k_features)
        stochastic_metrics.append(metrics)
        if "jaccard" in metrics:
            print(f"  Node {metrics['node_idx']}: Jaccard={metrics['jaccard']['mean']:.4f} "
                  f"± {metrics['jaccard']['std']:.4f}")

    # --- Perturbation stability ---
    print(f"\n4. Perturbation stability test (σ = {args.noise_levels})...")
    perturbation_results = []
    for node_idx in test_nodes[:3]:  # Subset for speed
        result = run_perturbation_test(
            model, data, node_idx, args.explainer,
            noise_levels=args.noise_levels, device=device
        )
        metrics = compute_perturbation_stability(result, args.top_k_features)
        perturbation_results.append(metrics)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{args.model}_{args.explainer}_stability.json"
    results_to_save = {
        "stochastic": stochastic_metrics,
        "perturbation": perturbation_results,
    }

    # Convert sets to lists for JSON serialization
    with open(output_file, "w") as f:
        json.dump(results_to_save, f, indent=2, default=str)

    print(f"\n✓ Results saved to {output_file}")


if __name__ == "__main__":
    main()
