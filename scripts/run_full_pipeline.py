"""
Full pipeline orchestrator.

Runs the complete experimental matrix:
  For each scenario × architecture × balancing:
    1. Train model (with hyperopt or fixed params)
    2. For each explainer:
       a. Generate explanations
       b. Run stochastic stability tests
       c. Run perturbation tests
    3. Log all results

Usage:
    uv run python scripts/run_full_pipeline.py
    uv run python scripts/run_full_pipeline.py --config configs/experiment.yaml
"""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from itertools import product

from src.data.loader import load_elliptic, print_dataset_stats
from src.data.preprocessing import preprocess
from src.data.imbalance import create_imbalance_scenario
from src.training.trainer import Trainer, build_model
from src.balancing.losses import get_loss_function
from src.explainability.explainer_runner import (
    create_explainer, explain_nodes, select_explanation_nodes
)
from src.explainability.shap_runner import explain_nodes_shap
from src.explainability.extraction import extract_subgraph, extract_feature_ranking
from src.stability.stochastic_test import run_stochastic_replicas
from src.stability.metrics import compute_stability_metrics
from src.analysis.tracking import ExperimentTracker


def parse_args():
    parser = argparse.ArgumentParser(description="Full experimental pipeline")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--quick", action="store_true", help="Quick test with reduced params")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Quick mode reduces everything for testing
    if args.quick:
        config["training"]["epochs"] = 10
        config["training"]["patience"] = 5
        config["stability"]["num_replicas"] = 3
        config["explainability"]["nodes_per_class"] = 5

    # Initialize tracker
    tracker = ExperimentTracker(
        backend=config["tracking"]["backend"],
        experiment_name=config["tracking"]["experiment_name"],
        results_dir=config["tracking"]["results_dir"],
    )

    # Load and preprocess data ONCE
    print("\n" + "=" * 70)
    print("LOADING ELLIPTIC DATASET")
    print("=" * 70)
    data_raw = load_elliptic(root=config["data"]["root"])
    print_dataset_stats(data_raw)
    preprocess(data_raw)

    # Experimental matrix
    scenarios = {s["name"]: s["illicit_to_licit"] for s in config["scenarios"]["imbalance_ratios"]}
    architectures = [m["name"] for m in config["models"]["architectures"]]
    balancing_techniques = [t["name"] for t in config["balancing"]["techniques"]]
    explainers = [e["name"] for e in config["explainability"]["methods"]]

    total = len(scenarios) * len(architectures) * len(balancing_techniques)
    print(f"\nExperimental matrix: {len(scenarios)} scenarios × {len(architectures)} archs "
          f"× {len(balancing_techniques)} balancing = {total} training configs")
    print(f"With {len(explainers)} explainers: {total * len(explainers)} explanation configs")

    run_count = 0
    for scenario_name, ratio in scenarios.items():
        for arch_name in architectures:
            for balance_name in balancing_techniques:
                run_count += 1
                run_id = f"{scenario_name}_{arch_name}_{balance_name}"

                print(f"\n{'='*70}")
                print(f"RUN {run_count}/{total}: {run_id}")
                print(f"{'='*70}")

                # Create scenario
                data = create_imbalance_scenario(data_raw, ratio, seed=42)

                # Build model
                model = build_model(
                    arch_name,
                    in_channels=data.num_node_features,
                    hidden_channels=128,
                    num_layers=2,
                    dropout=0.3,
                )

                # Loss
                focal_cfg = next(
                    (t for t in config["balancing"]["techniques"] if t["name"] == "focal_loss"),
                    {"gamma": 2.0, "alpha": 0.25}
                )
                loss_fn = get_loss_function(
                    balance_name, data.y, data.train_mask,
                    gamma=focal_cfg.get("gamma", 2.0),
                    alpha=focal_cfg.get("alpha", 0.25),
                    device=device,
                )

                # Train
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                trainer = Trainer(model, loss_fn, optimizer, device,
                                patience=config["training"]["patience"])
                results = trainer.train(
                    data, epochs=config["training"]["epochs"],
                    run_name=run_id,
                )

                pred_metrics = results["test_metrics"]
                print(f"  Test: F1={pred_metrics['f1']:.4f} MCC={pred_metrics['mcc']:.4f}")

                # Run explainers
                model.eval()
                selected = select_explanation_nodes(
                    data, n_per_class=config["explainability"]["nodes_per_class"]
                )
                test_nodes = selected["illicit"][:5]  # Focus on illicit

                for explainer_name in explainers:
                    print(f"\n  Explainer: {explainer_name}")

                    try:
                        # Stochastic stability
                        for node_idx in test_nodes[:3]:
                            stoch = run_stochastic_replicas(
                                model, data, node_idx, explainer_name,
                                num_replicas=config["stability"]["num_replicas"],
                                top_k_edges=config["stability"]["top_k_edges"],
                                device=device,
                            )
                            stab_metrics = compute_stability_metrics(
                                stoch, top_k_features=config["stability"]["top_k_features"]
                            )

                            # Flatten stability metrics for tracking
                            flat_stab = {}
                            if "jaccard" in stab_metrics:
                                flat_stab["jaccard_mean"] = stab_metrics["jaccard"]["mean"]
                                flat_stab["jaccard_std"] = stab_metrics["jaccard"]["std"]
                            if "spearman" in stab_metrics:
                                flat_stab["spearman_mean"] = stab_metrics["spearman"]["mean"]

                            tracker.log_run(
                                scenario=scenario_name,
                                architecture=arch_name,
                                balancing=balance_name,
                                explainer=explainer_name,
                                seed=42,
                                predictive_metrics=pred_metrics,
                                stability_metrics=flat_stab,
                            )
                    except Exception as e:
                        print(f"  ERROR with {explainer_name}: {e}")
                        print(f"  Skipping {explainer_name} for {run_id}")
                        tracker.log_run(
                            scenario=scenario_name,
                            architecture=arch_name,
                            balancing=balance_name,
                            explainer=explainer_name,
                            seed=42,
                            predictive_metrics=pred_metrics,
                            stability_metrics={"error": str(e)},
                        )

    print(f"\n{'='*70}")
    print("PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"Total runs logged: check {config['tracking']['results_dir']}")


if __name__ == "__main__":
    main()
