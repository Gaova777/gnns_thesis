"""
Smoke test — validates the full pipeline in ~10 minutes.

Runs 1 minimal config (1:10 / GCN / class_weighting) end-to-end with
drastically reduced parameters to catch all integration errors BEFORE
launching an overnight 10-hour run.

Usage:
    uv run python scripts/smoke_test.py --config configs/experiment_machineB.yaml
    uv run python scripts/smoke_test.py --config configs/experiment_machineC.yaml

Exit codes:
    0 — all checks passed
    1 — one or more checks failed (details printed to stdout)
"""

import argparse
import csv
import shutil
import sys
import time
import traceback
from pathlib import Path

# Ensure project root is in sys.path so `src` is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml


# ── Smoke-test parameters ─────────────────────────────────────────────────────
SMOKE_SCENARIO = "1:10"
SMOKE_RATIO = 0.1
SMOKE_ARCH = "GCN"
SMOKE_BALANCE = "class_weighting"
SMOKE_PARAMS = {
    "epochs": 5,
    "patience": 5,
    "nodes_per_class": 2,
    "num_replicas": 3,
    "gnn_explainer_epochs": 5,
    "pg_explainer_epochs": 5,
    "shap_samples": 5,
}
SMOKE_RESULTS_DIR = "./results_smoke"
SMOKE_EXPERIMENT = "smoke-test"


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Smoke test for pipeline validation")
    parser.add_argument("--config", type=str, required=True,
                        help="Machine config to validate (experiment_machineB/C.yaml)")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


class CheckList:
    def __init__(self):
        self.results = []

    def ok(self, name: str, detail: str = ""):
        self.results.append((name, True, detail))
        print(f"  [PASS] {name}" + (f" — {detail}" if detail else ""))

    def fail(self, name: str, detail: str = ""):
        self.results.append((name, False, detail))
        print(f"  [FAIL] {name}" + (f" — {detail}" if detail else ""))

    def summary(self) -> bool:
        passed = sum(1 for _, p, _ in self.results if p)
        total = len(self.results)
        print(f"\n{'='*60}")
        print(f"SMOKE TEST RESULT: {passed}/{total} checks passed")
        print("="*60)
        for name, p, detail in self.results:
            status = "PASS" if p else "FAIL"
            print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
        print("="*60)
        return passed == total


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    checks = CheckList()
    start = time.time()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu"
    print(f"\nSmoke test starting — device: {device}")
    print(f"Config: {args.config}\n")

    # ── Check 1: Config structure ──────────────────────────────────────────────
    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        required_sections = [
            "data", "scenarios", "models", "training",
            "balancing", "explainability", "stability", "tracking",
        ]
        missing = [s for s in required_sections if s not in config]
        if missing:
            checks.fail("Config structure", f"Missing sections: {missing}")
        else:
            checks.ok("Config structure", "All required sections present")
    except Exception as e:
        checks.fail("Config structure", str(e))
        checks.summary()
        sys.exit(1)

    # ── Check 2: Quality gate present ─────────────────────────────────────────
    has_gate = "quality_gate" in config.get("analysis", {})
    if has_gate:
        gate = config["analysis"]["quality_gate"]
        checks.ok("Quality gate", f"f1_min={gate.get('f1_min')}, mcc_min={gate.get('mcc_min')}")
    else:
        checks.fail("Quality gate", "Missing analysis.quality_gate — models may run without filter")

    # ── Check 3: Dataset loading ───────────────────────────────────────────────
    try:
        from src.data.loader import load_elliptic
        from src.data.preprocessing import preprocess

        data_raw = load_elliptic(root=config["data"]["root"])
        preprocess(data_raw)

        n_nan = torch.isnan(data_raw.x).sum().item()
        n_inf = torch.isinf(data_raw.x).sum().item()
        if n_nan > 0 or n_inf > 0:
            checks.fail("Dataset integrity", f"NaN={n_nan}, Inf={n_inf} in features")
        else:
            checks.ok("Dataset integrity",
                      f"{data_raw.num_nodes:,} nodes, {data_raw.num_edges:,} edges, "
                      f"{data_raw.num_node_features} features — clean")
    except Exception as e:
        checks.fail("Dataset integrity", traceback.format_exc().splitlines()[-1])
        checks.summary()
        sys.exit(1)

    # ── Check 4: Imbalance scenario ────────────────────────────────────────────
    try:
        from src.data.imbalance import create_imbalance_scenario

        data = create_imbalance_scenario(data_raw, SMOKE_RATIO, seed=42)
        train_y = data.y[data.train_mask]
        has_illicit = (train_y == 1).any().item()
        has_licit = (train_y == 0).any().item()
        n_illicit = (train_y == 1).sum().item()
        n_licit = (train_y == 0).sum().item()
        if has_illicit and has_licit:
            checks.ok("Imbalance scenario (1:10)",
                      f"train: {n_illicit} illicit, {n_licit} licit")
        else:
            checks.fail("Imbalance scenario", f"Missing class — illicit={has_illicit}, licit={has_licit}")
    except Exception as e:
        checks.fail("Imbalance scenario", traceback.format_exc().splitlines()[-1])

    # ── Check 5: Model training ────────────────────────────────────────────────
    pred_metrics = None
    model = None
    try:
        from src.training.trainer import Trainer, build_model
        from src.balancing.losses import get_loss_function
        from src.analysis.tracking import ExperimentTracker

        model = build_model(
            SMOKE_ARCH,
            in_channels=data.num_node_features,
            hidden_channels=64,
            num_layers=2,
            dropout=0.3,
        )
        loss_fn = get_loss_function(SMOKE_BALANCE, data.y, data.train_mask, device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        tracker = ExperimentTracker(
            backend="mlflow",
            experiment_name=SMOKE_EXPERIMENT,
            results_dir=SMOKE_RESULTS_DIR,
        )
        trainer = Trainer(
            model, loss_fn, optimizer, device,
            patience=SMOKE_PARAMS["patience"],
            tracker=tracker,
        )
        with tracker.training_run("smoke_train"):
            results = trainer.train(
                data,
                epochs=SMOKE_PARAMS["epochs"],
                run_name="smoke_train",
            )
        pred_metrics = results["test_metrics"]
        f1 = pred_metrics.get("f1", float("nan"))
        mcc = pred_metrics.get("mcc", float("nan"))
        is_numeric = isinstance(f1, (int, float)) and isinstance(mcc, (int, float))
        if is_numeric:
            checks.ok("Model training", f"F1={f1:.4f}, MCC={mcc:.4f} (5 epochs, values expected near 0)")
        else:
            checks.fail("Model training", f"Non-numeric metrics: F1={f1}, MCC={mcc}")
    except Exception as e:
        checks.fail("Model training", traceback.format_exc().splitlines()[-1])

    # ── Check 6: Node selection ────────────────────────────────────────────────
    test_nodes = []
    try:
        from src.explainability.explainer_runner import select_explanation_nodes

        model.eval()
        selected = select_explanation_nodes(
            data, n_per_class=SMOKE_PARAMS["nodes_per_class"]
        )
        test_nodes = selected["illicit"][:SMOKE_PARAMS["nodes_per_class"]]
        if len(test_nodes) > 0:
            checks.ok("Node selection", f"{len(test_nodes)} illicit nodes selected")
        else:
            checks.fail("Node selection", "No illicit nodes available in test set")
    except Exception as e:
        checks.fail("Node selection", traceback.format_exc().splitlines()[-1])

    # ── Check 7: GNNExplainer ─────────────────────────────────────────────────
    if model is not None and len(test_nodes) > 0:
        try:
            from src.stability.stochastic_test import run_stochastic_replicas
            from src.stability.metrics import compute_stability_metrics

            stoch = run_stochastic_replicas(
                model, data, test_nodes[0], "GNNExplainer",
                num_replicas=SMOKE_PARAMS["num_replicas"],
                top_k_edges=10,
                device=device,
                explainer_epochs=SMOKE_PARAMS["gnn_explainer_epochs"],
                explainer_lr=0.01,
            )
            stab = compute_stability_metrics(stoch, top_k_features=10)
            j_mean = stab.get("jaccard", {}).get("mean", None)
            if j_mean is not None:
                checks.ok("GNNExplainer", f"Jaccard mean={j_mean:.4f} (low expected with 5 epochs)")
            else:
                checks.fail("GNNExplainer", "No Jaccard metric returned")
        except Exception as e:
            checks.fail("GNNExplainer", traceback.format_exc().splitlines()[-1])
    else:
        checks.fail("GNNExplainer", "Skipped — model or nodes unavailable")

    # ── Check 8: PGExplainer (no crash) ───────────────────────────────────────
    if model is not None and len(test_nodes) > 0:
        try:
            from src.stability.stochastic_test import run_stochastic_test_batch

            batch_results = run_stochastic_test_batch(
                model, data, test_nodes[:2], method="PGExplainer",
                num_replicas=SMOKE_PARAMS["num_replicas"],
                top_k_edges=10,
                device=device,
                explainer_epochs=SMOKE_PARAMS["pg_explainer_epochs"],
                explainer_lr=0.003,
                nan_abort_threshold=2,
            )
            # PGExplainer may NaN-abort — that's OK, as long as it doesn't crash
            n_with_subgraphs = sum(1 for r in batch_results if len(r["subgraphs"]) > 0)
            checks.ok("PGExplainer (no crash)",
                      f"{len(batch_results)} nodes processed, {n_with_subgraphs} with subgraphs "
                      f"(NaN-abort is expected with 5 epochs)")
        except Exception as e:
            checks.fail("PGExplainer (no crash)", traceback.format_exc().splitlines()[-1])
    else:
        checks.fail("PGExplainer (no crash)", "Skipped — model or nodes unavailable")

    # ── Check 9: GNNShap (no OOM) ─────────────────────────────────────────────
    if model is not None and len(test_nodes) > 0:
        try:
            from src.explainability.shap_runner import explain_node_shap

            shap_result = explain_node_shap(
                model, data, test_nodes[0],
                num_samples=SMOKE_PARAMS["shap_samples"],
                device=device,
                seed=42,
            )
            oom_retries = shap_result.get("shap_oom_retries", 0)
            ranking = shap_result.get("feature_ranking", [])
            if oom_retries < 2 and len(ranking) > 0:
                checks.ok("GNNShap (no OOM)",
                          f"OOM retries={oom_retries}, top feature={ranking[0]}")
            else:
                checks.fail("GNNShap (no OOM)",
                            f"OOM retries={oom_retries}, ranking len={len(ranking)}")
        except Exception as e:
            checks.fail("GNNShap (no OOM)", traceback.format_exc().splitlines()[-1])
    else:
        checks.fail("GNNShap (no OOM)", "Skipped — model or nodes unavailable")

    # ── Check 10: CSV schema consistency ──────────────────────────────────────
    try:
        from src.analysis.tracking import ExperimentTracker, CSV_SCHEMA_FIELDS

        tracker = ExperimentTracker(
            backend="mlflow",
            experiment_name=SMOKE_EXPERIMENT,
            results_dir=SMOKE_RESULTS_DIR,
        )
        dummy_metrics = {"loss": 0.5, "f1": 0.1, "mcc": 0.05, "pr_auc": 0.1}

        # Write a normal row
        tracker.log_run(
            scenario=SMOKE_SCENARIO, architecture=SMOKE_ARCH,
            balancing=SMOKE_BALANCE, explainer="GNNExplainer", seed=42,
            predictive_metrics=dummy_metrics,
            stability_metrics={"jaccard_mean": 0.5, "jaccard_std": 0.1, "spearman_mean": 0.8},
        )
        # Write an error row (simulates PGExplainer crash)
        tracker.log_run(
            scenario=SMOKE_SCENARIO, architecture=SMOKE_ARCH,
            balancing=SMOKE_BALANCE, explainer="PGExplainer", seed=42,
            predictive_metrics=dummy_metrics,
            stability_metrics={"error": "Simulated error for smoke test"},
        )

        csv_path = Path(SMOKE_RESULTS_DIR) / f"{SMOKE_EXPERIMENT}.csv"
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if len(rows) < 2:
            checks.fail("CSV schema", f"Expected ≥2 rows, got {len(rows)}")
        else:
            # All rows must have the same fieldnames (= CSV_SCHEMA_FIELDS)
            all_fields = [set(r.keys()) for r in rows]
            consistent = all(f == all_fields[0] for f in all_fields)

            # Normal row must have numeric stab_jaccard_mean
            normal_row = next((r for r in rows if r.get("explainer") == "GNNExplainer"), None)
            error_row = next((r for r in rows if r.get("explainer") == "PGExplainer"), None)

            jaccard_val = normal_row.get("stab_jaccard_mean", "") if normal_row else ""
            error_val = error_row.get("stab_error", "") if error_row else ""

            try:
                float(jaccard_val)
                jaccard_numeric = True
            except (ValueError, TypeError):
                jaccard_numeric = False

            if consistent and jaccard_numeric and error_val:
                checks.ok("CSV schema",
                          f"{len(rows)} rows, consistent fields={consistent}, "
                          f"jaccard numeric={jaccard_numeric}, error captured={bool(error_val)}")
            else:
                checks.fail("CSV schema",
                            f"consistent={consistent}, jaccard_numeric={jaccard_numeric}, "
                            f"jaccard_val={repr(jaccard_val)}, error_val={repr(error_val)}")
    except Exception as e:
        checks.fail("CSV schema", traceback.format_exc().splitlines()[-1])

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - start
    print(f"\nElapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    all_passed = checks.summary()

    # Cleanup smoke test artifacts
    for path in [Path(SMOKE_RESULTS_DIR)]:
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
    # Clean up MLflow smoke runs (optional — they're tiny)
    mlruns_smoke = Path("mlruns")
    if mlruns_smoke.exists():
        try:
            import mlflow
            exp = mlflow.get_experiment_by_name(SMOKE_EXPERIMENT)
            if exp is not None:
                runs = mlflow.search_runs(
                    experiment_ids=[exp.experiment_id], output_format="list"
                )
                for run in runs:
                    mlflow.delete_run(run.info.run_id)
        except Exception:
            pass  # Non-critical cleanup

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
