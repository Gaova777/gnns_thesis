"""
Explain matrix runner (Pipeline v3 — Script 2 of 2).

Responsibility: load trained model checkpoints produced by train_matrix.py,
enforce the F1/MCC quality gate, and run explainability + stability tests
only on models that actually learned.

Inputs (from `models_dir`):
  - {run_id}_best.pt    — model weights
  - {run_id}_meta.json  — metadata (best_params, test_metrics, quality_passed, ...)

Outputs:
  - {results_dir}/{experiment_name}.csv  (one row per config × explainer)
  - MLflow nested runs under the training parent run (if mlflow_run_id present)

Usage:
    uv run python scripts/explain_matrix.py --config configs/experiment_machineB_v3.yaml
    uv run python scripts/explain_matrix.py --config ... --arch GraphSAGE
    uv run python scripts/explain_matrix.py --config ... --explainer PGExplainer
    uv run python scripts/explain_matrix.py --config ... --force  # ignore quality gate
"""

import argparse
import json
import signal
import sys
import time
import warnings
from pathlib import Path

# Make project root importable (so `src.*` works when run via `uv run python scripts/...`)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import yaml
from tqdm import tqdm

from src.analysis.tracking import ExperimentTracker
from src.data.imbalance import create_imbalance_scenario
from src.data.loader import load_elliptic, print_dataset_stats
from src.data.preprocessing import preprocess
from src.explainability.explainer_runner import select_explanation_nodes
from src.stability.metrics import compute_stability_metrics
from src.stability.stochastic_test import (
    run_stochastic_replicas,
    run_stochastic_test_batch,
)
from src.training.trainer import build_model

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

_interrupted = False


def _signal_handler(signum, frame):
    global _interrupted
    _interrupted = True
    sig_name = signal.Signals(signum).name
    tqdm.write(f"\n[SIGNAL] {sig_name} received — finishing current config then exiting.")


def parse_args():
    p = argparse.ArgumentParser(description="Explain matrix over trained checkpoints")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--models-dir", type=str, default=None,
                   help="Override config's models_dir")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--arch", type=str, default=None)
    p.add_argument("--scenario", type=str, default=None)
    p.add_argument("--balancing", type=str, default=None)
    p.add_argument("--explainer", type=str, default=None,
                   help="Filter to one explainer: GNNExplainer | PGExplainer | GNNShap")
    p.add_argument("--force", action="store_true",
                   help="Run explainers even if quality_passed=False")
    p.add_argument("--resume", action="store_true",
                   help="Skip (run_id, explainer) pairs already in results CSV")
    p.add_argument("--max-hours", type=float, default=10.0)
    return p.parse_args()


def _safe_name(s: str) -> str:
    return s.replace(":", "-").replace("/", "-")


def _load_meta_files(models_dir: Path) -> list[dict]:
    metas = []
    for path in sorted(models_dir.glob("*_meta.json")):
        try:
            with open(path) as f:
                meta = json.load(f)
            meta["_meta_path"] = str(path)
            metas.append(meta)
        except Exception as exc:
            tqdm.write(f"  WARNING: could not read {path.name}: {exc}")
    return metas


def _completed_pairs_from_csv(tracker: ExperimentTracker) -> set[tuple[str, str]]:
    """Return set of (run_id, explainer) already in the results CSV."""
    csv_path = Path(tracker.results_dir) / f"{tracker.experiment_name}.csv"
    if not csv_path.exists():
        return set()
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        if not {"scenario", "architecture", "balancing", "explainer"}.issubset(df.columns):
            return set()
        pairs = set()
        for _, row in df.iterrows():
            rid = f"{row['scenario']}_{row['architecture']}_{row['balancing']}"
            pairs.add((rid, str(row["explainer"])))
        return pairs
    except Exception:
        return set()


def _run_one_explainer(
    model, data, test_nodes, explainer_name, explainer_cfg, stability_cfg, device,
) -> dict:
    """Run one explainer over the test nodes and aggregate stability."""
    ex_epochs = explainer_cfg.get("epochs", 200)
    ex_lr = explainer_cfg.get("lr", 0.01)
    shap_samples = explainer_cfg.get("num_samples", 50)
    nan_abort = explainer_cfg.get("nan_abort_threshold", 2)
    num_replicas = stability_cfg.get("num_replicas", 5)
    top_k_edges = stability_cfg.get("top_k_edges", 20)
    top_k_features = stability_cfg.get("top_k_features", 20)

    agg = {"jaccard_means": [], "spearman_means": [], "shap_oom_retries": 0}

    if explainer_name == "PGExplainer":
        batch_results = run_stochastic_test_batch(
            model, data, test_nodes, method="PGExplainer",
            num_replicas=num_replicas, top_k_edges=top_k_edges, device=device,
            explainer_epochs=ex_epochs, explainer_lr=ex_lr,
            nan_abort_threshold=nan_abort,
        )
        for stoch in batch_results:
            m = compute_stability_metrics(stoch, top_k_features=top_k_features)
            if "jaccard" in m:
                agg["jaccard_means"].append(m["jaccard"]["mean"])
            if "spearman" in m:
                agg["spearman_means"].append(m["spearman"]["mean"])
    else:
        node_pbar = tqdm(test_nodes, desc=f"    {explainer_name}",
                         leave=False, unit="node")
        for node_idx in node_pbar:
            stoch = run_stochastic_replicas(
                model, data, node_idx, explainer_name,
                num_replicas=num_replicas, top_k_edges=top_k_edges, device=device,
                explainer_epochs=ex_epochs, explainer_lr=ex_lr,
                shap_samples=shap_samples,
            )
            m = compute_stability_metrics(stoch, top_k_features=top_k_features)
            if "jaccard" in m:
                agg["jaccard_means"].append(m["jaccard"]["mean"])
            if "spearman" in m:
                agg["spearman_means"].append(m["spearman"]["mean"])
            agg["shap_oom_retries"] += stoch.get("shap_oom_retries", 0)
        node_pbar.close()

    flat = {}
    if agg["jaccard_means"]:
        flat["jaccard_mean"] = float(np.mean(agg["jaccard_means"]))
        flat["jaccard_std"] = float(np.std(agg["jaccard_means"]))
    if agg["spearman_means"]:
        flat["spearman_mean"] = float(np.mean(agg["spearman_means"]))
    if agg["shap_oom_retries"] > 0:
        flat["shap_oom_retries"] = agg["shap_oom_retries"]
    return flat


def main():
    args = parse_args()
    pipeline_start = time.time()
    deadline_sec = args.max_hours * 3600

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tracking_cfg = config["tracking"]
    models_dir = Path(args.models_dir or tracking_cfg.get("models_dir", "./results_models_v3"))
    if not models_dir.exists():
        raise FileNotFoundError(f"Models dir not found: {models_dir}")

    metas = _load_meta_files(models_dir)
    if not metas:
        print(f"No *_meta.json files found in {models_dir}. Run train_matrix.py first.")
        return

    # Apply CLI filters
    def _keep(meta: dict) -> bool:
        if args.scenario and meta.get("scenario") != args.scenario:
            return False
        if args.arch and meta.get("architecture") != args.arch:
            return False
        if args.balancing and meta.get("balancing") != args.balancing:
            return False
        return True

    metas = [m for m in metas if _keep(m)]
    if not metas:
        print("No metas matched filters.")
        return

    # Tracker + resume state
    tracker = ExperimentTracker(
        backend=tracking_cfg["backend"],
        experiment_name=tracking_cfg["experiment_name"],
        results_dir=tracking_cfg["results_dir"],
    )
    completed_pairs = _completed_pairs_from_csv(tracker) if args.resume else set()

    # Dataset — load + preprocess once
    print("\n" + "=" * 70)
    print("LOADING ELLIPTIC DATASET")
    print("=" * 70)
    data_raw = load_elliptic(root=config["data"]["root"])
    print_dataset_stats(data_raw)
    preprocess(data_raw)

    # Explainers to run
    explainer_methods = config["explainability"]["methods"]
    if args.explainer:
        explainer_methods = [m for m in explainer_methods if m["name"] == args.explainer]
        if not explainer_methods:
            print(f"No explainer named {args.explainer!r} in config.")
            return
    nodes_per_class = config["explainability"]["nodes_per_class"]

    gate_cfg = config.get("analysis", {}).get("quality_gate", {})
    gate_f1 = gate_cfg.get("f1_min", 0.70)
    gate_mcc = gate_cfg.get("mcc_min", 0.40)

    seed = config.get("training", {}).get("seeds", [42])[0]

    total_cfgs = len(metas)
    print(f"\nExplain matrix: {total_cfgs} checkpoints × {len(explainer_methods)} explainers "
          f"= {total_cfgs * len(explainer_methods)} runs (before gating)")

    gated_out = []
    pbar = tqdm(metas, desc="Explain matrix", unit="ckpt",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

    for meta in pbar:
        if _interrupted:
            tqdm.write("  Interrupt — stopping.")
            break
        elapsed = time.time() - pipeline_start
        if (deadline_sec - elapsed) < 1800:
            tqdm.write(f"  DEADLINE: {elapsed/3600:.1f}h elapsed, <30 min remaining — stopping.")
            break

        run_id = meta["run_id"]
        pbar.set_postfix_str(run_id, refresh=True)

        # Quality gate
        if not args.force:
            if not meta.get("quality_passed", False):
                tm = meta.get("test_metrics", {})
                tqdm.write(
                    f"  SKIP (quality gate) {run_id}: "
                    f"F1={tm.get('f1', float('nan')):.4f} MCC={tm.get('mcc', float('nan')):.4f} "
                    f"< F1>={gate_f1} AND MCC>={gate_mcc}"
                )
                gated_out.append(run_id)
                # Log a SKIPPED row so downstream analysis sees it
                try:
                    tracker.log_run(
                        scenario=meta["scenario"], architecture=meta["architecture"],
                        balancing=meta["balancing"], explainer="SKIPPED_QUALITY_GATE",
                        seed=seed, predictive_metrics=tm,
                        stability_metrics={"reason": "quality_gate_v3"},
                    )
                except Exception:
                    pass
                continue

        # Build + load model
        arch = meta["architecture"]
        bp = meta["best_params"]
        arch_kwargs = {}
        if arch == "GAT" and "heads" in bp:
            arch_kwargs["heads"] = bp["heads"]
        if arch == "TAGCN" and "K" in bp:
            arch_kwargs["K"] = bp["K"]

        # Recreate data for this scenario
        data = create_imbalance_scenario(data_raw, meta["imbalance_ratio"], seed=seed)

        model = build_model(
            arch,
            in_channels=data.num_node_features,
            hidden_channels=bp.get("hidden_dim", 128),
            num_layers=bp.get("num_layers", 2),
            dropout=bp.get("dropout", 0.3),
            **arch_kwargs,
        )
        ckpt_file = models_dir / meta["checkpoint"]
        if not ckpt_file.exists():
            tqdm.write(f"  ERROR: checkpoint missing: {ckpt_file}. Skipping.")
            continue
        model.load_state_dict(torch.load(ckpt_file, map_location=device, weights_only=True))
        model = model.to(device)
        model.eval()

        # Threshold calibration is applied at reporting time in train_matrix.
        # Explainers (GNNExplainer/PGExplainer/GNNShap) work on raw logits and
        # feature attributions — they don't depend on the decision threshold,
        # so we log it for context but don't need to apply it.
        calib_t = meta.get("calibrated_threshold")
        tm = meta.get("test_metrics", {})
        tm_argmax = meta.get("test_metrics_argmax", {})
        if calib_t is not None:
            tqdm.write(
                f"  Loaded {run_id}: calibrated_threshold={calib_t:.2f} | "
                f"test F1 calib={tm.get('f1', float('nan')):.4f} "
                f"(argmax={tm_argmax.get('f1', float('nan')):.4f})"
            )

        # Select test nodes
        selected = select_explanation_nodes(data, n_per_class=nodes_per_class, seed=seed)
        test_nodes = selected["illicit"][:nodes_per_class]

        # Open a parent run to host the nested explainer runs (explainer_run
        # uses nested=True which requires an active parent).
        pred_metrics = meta["test_metrics"]
        parent_params = {
            "scenario": meta["scenario"], "architecture": meta["architecture"],
            "balancing": meta["balancing"],
            "phase": "explain_only", "seed": seed,
        }
        with tracker.training_run(f"{run_id}__explain", params=parent_params):
            tracker.log_test_metrics(pred_metrics)
            for ex_cfg in explainer_methods:
                ex_name = ex_cfg["name"]
                if args.resume and (run_id, ex_name) in completed_pairs:
                    tqdm.write(f"  SKIP (resume): {run_id} / {ex_name}")
                    continue

                with tracker.explainer_run(ex_name, params={"method": ex_name}):
                    try:
                        flat = _run_one_explainer(
                            model, data, test_nodes, ex_name, ex_cfg,
                            config.get("stability", {}), device,
                        )
                        tracker.log_stability(flat)
                        jmean = flat.get("jaccard_mean")
                        tqdm.write(
                            f"  {run_id} / {ex_name}: "
                            + (f"Jaccard={jmean:.4f}" if jmean is not None else "done")
                            + (f", Spearman={flat['spearman_mean']:.4f}"
                               if 'spearman_mean' in flat else "")
                        )
                        tracker.log_run(
                            scenario=meta["scenario"], architecture=meta["architecture"],
                            balancing=meta["balancing"], explainer=ex_name,
                            seed=seed, predictive_metrics=pred_metrics,
                            stability_metrics=flat,
                        )
                    except Exception as exc:
                        tqdm.write(f"  ERROR {run_id}/{ex_name}: {exc}")
                        tracker.log_stability({"error": str(exc)})
                        tracker.log_run(
                            scenario=meta["scenario"], architecture=meta["architecture"],
                            balancing=meta["balancing"], explainer=ex_name,
                            seed=seed, predictive_metrics=pred_metrics,
                            stability_metrics={"error": str(exc)},
                        )

    pbar.close()
    print("\n" + "=" * 70)
    print("EXPLAIN MATRIX SUMMARY")
    print("=" * 70)
    print(f"  Checkpoints processed: {total_cfgs}")
    print(f"  Gated out (quality):   {len(gated_out)}")
    for r in gated_out:
        print(f"    ✗ {r}")
    total_elapsed_h = (time.time() - pipeline_start) / 3600
    print(f"  Total time: {total_elapsed_h:.2f} h")
    print(f"  CSV: {tracking_cfg['results_dir']}/{tracking_cfg['experiment_name']}.csv")


if __name__ == "__main__":
    main()
