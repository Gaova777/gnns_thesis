"""
Full pipeline orchestrator with real-time MLflow logging.

Runs the complete experimental matrix:
  For each scenario x architecture x balancing:
    1. Train model (logged per-epoch to MLflow)
    2. For each explainer:
       a. Run stochastic stability tests
       b. Log stability metrics as nested MLflow run
    3. Progress bar with ETA via tqdm

Usage:
    uv run python scripts/run_full_pipeline.py --quick
    uv run python scripts/run_full_pipeline.py --resume
    uv run python scripts/run_full_pipeline.py --clean
"""

import argparse
import yaml
import time
import shutil
import torch
import warnings
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.data.loader import load_elliptic, print_dataset_stats
from src.data.preprocessing import preprocess
from src.data.imbalance import create_imbalance_scenario
from src.training.trainer import Trainer, build_model
from src.balancing.losses import get_loss_function
from src.explainability.explainer_runner import (
    create_explainer, explain_nodes, select_explanation_nodes, train_pgexplainer
)
from src.explainability.shap_runner import explain_nodes_shap
from src.explainability.extraction import extract_subgraph, extract_feature_ranking
from src.stability.stochastic_test import run_stochastic_replicas
from src.stability.metrics import compute_stability_metrics
from src.analysis.tracking import ExperimentTracker

# Suppress sklearn single-label warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


def parse_args():
    parser = argparse.ArgumentParser(description="Full experimental pipeline")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--quick", action="store_true", help="Quick test with reduced params")
    parser.add_argument("--resume", action="store_true", help="Resume: skip already completed configs")
    parser.add_argument("--clean", action="store_true", help="Clean all previous results before starting")
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
        config["training"]["epochs"] = 5
        config["training"]["patience"] = 5
        config["stability"]["num_replicas"] = 2
        config["explainability"]["nodes_per_class"] = 3
        # Limit to 1 arch × 1 balancing → 4 configs total (1 per scenario)
        config["models"]["architectures"] = config["models"]["architectures"][:1]
        config["balancing"]["techniques"] = config["balancing"]["techniques"][:1]
        # Reduce explainer iterations drastically for smoke test (CPU is slow)
        for m in config["explainability"]["methods"]:
            if m["name"] == "GNNShap":
                m["num_samples"] = 5
            if "epochs" in m:
                m["epochs"] = 5  # GNNExplainer: 200→5, PGExplainer: 30→5

    # Initialize tracker
    tracker = ExperimentTracker(
        backend=config["tracking"]["backend"],
        experiment_name=config["tracking"]["experiment_name"],
        results_dir=config["tracking"]["results_dir"],
    )

    # Handle --clean flag
    if args.clean:
        print("\nCleaning previous results...")
        tracker.clean_experiment()
        ckpt_dir = Path("./results/models")
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
            print(f"  Cleaned checkpoints: {ckpt_dir}")

    # Handle --resume flag
    completed_runs = set()
    if args.resume:
        completed_runs = tracker.get_completed_runs()
        if completed_runs:
            print(f"\nResuming: {len(completed_runs)} configs already completed, will skip them.")

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

    # Build list of all configs
    all_configs = []
    for scenario_name, ratio in scenarios.items():
        for arch_name in architectures:
            for balance_name in balancing_techniques:
                run_id = f"{scenario_name}_{arch_name}_{balance_name}"
                all_configs.append((scenario_name, ratio, arch_name, balance_name, run_id))

    total = len(all_configs)
    print(f"\nExperimental matrix: {len(scenarios)} scenarios x {len(architectures)} archs "
          f"x {len(balancing_techniques)} balancing = {total} training configs")
    print(f"With {len(explainers)} explainers: {total * len(explainers)} explanation configs")

    # Main loop with tqdm progress bar
    pbar = tqdm(all_configs, desc="Pipeline", unit="config",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

    for scenario_name, ratio, arch_name, balance_name, run_id in pbar:
        pbar.set_postfix_str(run_id, refresh=True)

        # Skip if already completed (--resume)
        if run_id in completed_runs:
            tqdm.write(f"  SKIP (already completed): {run_id}")
            continue

        tqdm.write(f"\n{'='*70}")
        tqdm.write(f"CONFIG: {run_id}")
        tqdm.write(f"{'='*70}")

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

        # Parent run in MLflow
        train_params = {
            "scenario": scenario_name,
            "architecture": arch_name,
            "balancing": balance_name,
            "hidden_channels": 128,
            "num_layers": 2,
            "dropout": 0.3,
            "lr": 0.001,
            "epochs": config["training"]["epochs"],
            "patience": config["training"]["patience"],
        }

        with tracker.training_run(run_id, params=train_params):
            # Train (with per-epoch MLflow logging)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            trainer = Trainer(
                model, loss_fn, optimizer, device,
                patience=config["training"]["patience"],
                tracker=tracker,
            )
            results = trainer.train(
                data, epochs=config["training"]["epochs"],
                run_name=run_id,
            )

            pred_metrics = results["test_metrics"]
            tqdm.write(f"  Test: F1={pred_metrics['f1']:.4f} MCC={pred_metrics['mcc']:.4f}")

            # Log test metrics to parent run
            tracker.log_test_metrics(pred_metrics)

            # Log checkpoint artifact
            safe_run_name = run_id.replace(":", "-")
            ckpt_path = Path("./results/models") / f"{safe_run_name}_best.pt"
            if ckpt_path.exists():
                tracker.log_model_artifact(str(ckpt_path))

            # Select nodes for explanation
            model.eval()
            selected = select_explanation_nodes(
                data, n_per_class=config["explainability"]["nodes_per_class"]
            )
            n_explain = config["explainability"]["nodes_per_class"]
            test_nodes = selected["illicit"][:n_explain]

            # Explainer loop with sub-progress
            for explainer_name in tqdm(explainers, desc="  Explainers", leave=False, unit="method"):
                # Pull per-explainer hyperparams from config (epochs, lr, num_samples)
                explainer_cfg = next(
                    (e for e in config["explainability"]["methods"] if e["name"] == explainer_name),
                    {}
                )
                explainer_epochs = explainer_cfg.get("epochs", 200)
                explainer_lr = explainer_cfg.get("lr", 0.01)
                shap_samples = explainer_cfg.get("num_samples", 100)

                with tracker.explainer_run(explainer_name, params={"method": explainer_name}):
                    try:
                        node_pbar = tqdm(test_nodes, desc=f"    {explainer_name}", leave=False, unit="node")
                        all_stab = {"jaccard_means": [], "spearman_means": []}

                        for node_idx in node_pbar:
                            stoch = run_stochastic_replicas(
                                model, data, node_idx, explainer_name,
                                num_replicas=config["stability"]["num_replicas"],
                                top_k_edges=config["stability"]["top_k_edges"],
                                device=device,
                                explainer_epochs=explainer_epochs,
                                explainer_lr=explainer_lr,
                                shap_samples=shap_samples,
                            )
                            stab_metrics = compute_stability_metrics(
                                stoch, top_k_features=config["stability"]["top_k_features"]
                            )

                            if "jaccard" in stab_metrics:
                                all_stab["jaccard_means"].append(stab_metrics["jaccard"]["mean"])
                            if "spearman" in stab_metrics:
                                all_stab["spearman_means"].append(stab_metrics["spearman"]["mean"])

                        node_pbar.close()

                        # Aggregate stability over nodes
                        flat_stab = {}
                        if all_stab["jaccard_means"]:
                            flat_stab["jaccard_mean"] = np.mean(all_stab["jaccard_means"])
                            flat_stab["jaccard_std"] = np.std(all_stab["jaccard_means"])
                        if all_stab["spearman_means"]:
                            flat_stab["spearman_mean"] = np.mean(all_stab["spearman_means"])

                        tracker.log_stability(flat_stab)
                        tqdm.write(f"    {explainer_name}: Jaccard={flat_stab.get('jaccard_mean', 'N/A'):.4f}" if flat_stab.get('jaccard_mean') else f"    {explainer_name}: done")

                        # Also log to CSV for backup
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
                        tqdm.write(f"    ERROR {explainer_name}: {e}")
                        tracker.log_stability({"error": str(e)})
                        tracker.log_run(
                            scenario=scenario_name,
                            architecture=arch_name,
                            balancing=balance_name,
                            explainer=explainer_name,
                            seed=42,
                            predictive_metrics=pred_metrics,
                            stability_metrics={"error": str(e)},
                        )

    pbar.close()
    print(f"\n{'='*70}")
    print("PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"Results logged to: {config['tracking']['results_dir']}")
    print(f"View in MLflow UI: uv run mlflow ui --backend-store-uri sqlite:///mlruns.db")


if __name__ == "__main__":
    main()
