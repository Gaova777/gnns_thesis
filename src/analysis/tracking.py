"""
Experiment tracking and logging with real-time MLflow support.

Uses parent runs for training configs and nested runs for explainers.
Logs per-epoch metrics so curves are visible live in MLflow UI.
"""

import csv
import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Fixed CSV schema — prevents column mismatch between normal and error rows.
# All rows always use these exact fieldnames; missing values are written as None.
CSV_SCHEMA_FIELDS = [
    "timestamp", "scenario", "architecture", "balancing", "explainer", "seed",
    # AUDIT FIX (1b): these are the MODEL's global test metrics — renamed from
    # pred_* so they are never mistaken for the explainer's or per-node metrics.
    "model_test_loss", "model_test_f1", "model_test_mcc", "model_test_pr_auc",
    "stab_jaccard_mean", "stab_jaccard_std", "stab_spearman_mean",
    "stab_shap_oom_retries", "stab_n_tp", "stab_n_measurable",
    "stab_subgraph_n_nodes", "stab_subgraph_n_edges",
    "stab_error", "stab_reason",
]


class ExperimentTracker:
    """
    Unified experiment tracker with real-time MLflow logging.

    Architecture:
      - One MLflow experiment per study
      - One parent run per training config (scenario+arch+balance)
      - Nested runs per explainer within each parent

    Usage:
        tracker = ExperimentTracker(backend="mlflow")
        with tracker.training_run("1:1_GCN_none", params={...}) as run:
            # Log per-epoch metrics
            tracker.log_epoch(epoch=1, train_loss=0.5, val_f1=0.3, val_mcc=0.1)
            # After training:
            tracker.log_test_metrics({"f1": 0.8, "mcc": 0.7})
            # Per explainer:
            with tracker.explainer_run("GNNExplainer") as exp_run:
                tracker.log_stability({"jaccard_mean": 0.85})
    """

    def __init__(
        self,
        backend: str = "mlflow",
        experiment_name: str = "xai-gnn-stability",
        results_dir: str = "./results",
        tracking_uri: str = "sqlite:///mlruns.db",
    ):
        self.backend = backend
        self.experiment_name = experiment_name
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._active_parent_run = None
        self._active_child_run = None

        if backend == "mlflow" and MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            print(f"  MLflow experiment: {experiment_name}")
            print(f"  Tracking URI: {tracking_uri}")
        elif backend == "mlflow" and not MLFLOW_AVAILABLE:
            print("  MLflow not installed, falling back to CSV")
            self.backend = "csv"

        # Always init CSV path (used as backup even with MLflow)
        self.csv_path = self.results_dir / f"{experiment_name}.csv"
        self._csv_write_count = 0

    # ── Context managers for MLflow runs ──

    class _RunContext:
        """Context manager for MLflow runs."""
        def __init__(self, tracker, run_name, params=None, nested=False, parent_run_id=None):
            self.tracker = tracker
            self.run_name = run_name
            self.params = params or {}
            self.nested = nested
            self.parent_run_id = parent_run_id
            self.run = None

        def __enter__(self):
            if self.tracker.backend == "mlflow" and MLFLOW_AVAILABLE:
                self.run = mlflow.start_run(
                    run_name=self.run_name,
                    nested=self.nested,
                )
                if self.params:
                    # MLflow limits param values to 500 chars
                    safe_params = {k: str(v)[:500] for k, v in self.params.items()}
                    mlflow.log_params(safe_params)
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.run and MLFLOW_AVAILABLE:
                if exc_type:
                    mlflow.set_tag("status", "FAILED")
                    mlflow.set_tag("error", str(exc_val)[:500])
                else:
                    mlflow.set_tag("status", "COMPLETED")
                mlflow.end_run()
            return False  # Don't suppress exceptions

    def training_run(self, run_name: str, params: dict = None):
        """Start a parent run for a training configuration."""
        ctx = self._RunContext(self, run_name, params, nested=False)
        self._active_parent_run = ctx
        return ctx

    def explainer_run(self, explainer_name: str, params: dict = None):
        """Start a nested run for an explainer within the current training run."""
        ctx = self._RunContext(self, explainer_name, params, nested=True)
        self._active_child_run = ctx
        return ctx

    # ── Per-epoch logging ──

    def log_epoch(self, epoch: int, **metrics):
        """Log per-epoch metrics (visible as curves in MLflow UI)."""
        if self.backend == "mlflow" and MLFLOW_AVAILABLE:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value, step=epoch)

    # ── Summary metrics ──

    def log_test_metrics(self, metrics: dict):
        """Log final test metrics on the current run."""
        if self.backend == "mlflow" and MLFLOW_AVAILABLE:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"test_{key}", value)

    def log_stability(self, metrics: dict):
        """Log stability metrics on the current (nested) run."""
        if self.backend == "mlflow" and MLFLOW_AVAILABLE:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"stab_{key}", value)
                elif isinstance(value, str):
                    mlflow.set_tag(f"stab_{key}", value[:500])

    def log_model_artifact(self, checkpoint_path: str):
        """Log a model checkpoint as an MLflow artifact."""
        if self.backend == "mlflow" and MLFLOW_AVAILABLE:
            mlflow.log_artifact(str(checkpoint_path), artifact_path="models")

    # ── Legacy flat logging (CSV fallback + backward compat) ──

    def log_run(
        self,
        scenario: str,
        architecture: str,
        balancing: str,
        explainer: str,
        seed: int,
        predictive_metrics: dict,
        stability_metrics: dict = None,
        hyperparams: dict = None,
        tags: dict = None,
    ) -> None:
        """Legacy: log a complete run as a single row (CSV backend)."""
        run_data = {
            "timestamp": datetime.now().isoformat(),
            "scenario": scenario,
            "architecture": architecture,
            "balancing": balancing,
            "explainer": explainer,
            "seed": seed,
            **{f"model_test_{k}": v for k, v in predictive_metrics.items()},
        }

        if stability_metrics:
            for k, v in stability_metrics.items():
                if isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        if not isinstance(sub_v, (list, dict)):
                            run_data[f"stab_{k}_{sub_k}"] = sub_v
                else:
                    run_data[f"stab_{k}"] = v

        self._log_csv(run_data)

    def _log_csv(self, run_data: dict):
        """Log to CSV file atomically (tmp → rename) with periodic backup.

        Uses a fixed schema (CSV_SCHEMA_FIELDS) so every row has the same
        columns regardless of whether it's a normal run or an error row.
        Missing fields are written as empty (None → '').
        """
        tmp_path = self.csv_path.with_suffix(".tmp")
        bak_path = Path(str(self.csv_path) + ".bak")

        # Map run_data onto the fixed schema; missing fields become None
        row = {field: run_data.get(field, None) for field in CSV_SCHEMA_FIELDS}

        # Copy existing CSV into tmp so we can append to it
        if self.csv_path.exists():
            shutil.copy2(self.csv_path, tmp_path)
        file_exists = tmp_path.exists()

        with open(tmp_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_SCHEMA_FIELDS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())

        # Atomic replace — on Linux/Windows this is a single-syscall rename
        os.replace(tmp_path, self.csv_path)

        # Backup every 5 writes
        self._csv_write_count += 1
        if self._csv_write_count % 5 == 0:
            shutil.copy2(self.csv_path, bak_path)

    # ── Utilities ──

    def mark_interrupted(self, run_id: str) -> None:
        """
        Persist run_id as interrupted so --resume skips it cleanly.

        Writes to {results_dir}/.interrupted_runs (one run_id per line).
        Also tags the active MLflow run if one is open.
        """
        interrupted_file = self.results_dir / ".interrupted_runs"
        try:
            with open(interrupted_file, "a") as f:
                f.write(run_id + "\n")
                f.flush()
                os.fsync(f.fileno())
        except OSError:
            pass
        if self.backend == "mlflow" and MLFLOW_AVAILABLE:
            try:
                mlflow.set_tag("status", "interrupted")
            except Exception:
                pass

    def get_completed_runs(self) -> set:
        """
        Return set of run_ids to skip on --resume.

        Checks three sources in order:
          1. MLflow COMPLETED parent runs (authoritative when DB exists)
          2. CSV backup rows without errors (fallback when MLflow DB is missing)
          3. .interrupted_runs file (always checked — partial runs should not retry)
        """
        completed: set = set()

        # Source 1: MLflow
        if self.backend == "mlflow" and MLFLOW_AVAILABLE:
            try:
                experiment = mlflow.get_experiment_by_name(self.experiment_name)
                if experiment is not None:
                    runs = mlflow.search_runs(
                        experiment_ids=[experiment.experiment_id],
                        filter_string="tags.status = 'COMPLETED' AND tags.`mlflow.parentRunId` = ''",
                        output_format="list",
                    )
                    completed = {r.info.run_name for r in runs}
            except Exception as e:
                print(f"  WARNING: MLflow query failed ({e}), falling back to CSV")

        # Source 2: CSV backup (handles missing/corrupt MLflow DB)
        if not completed and self.csv_path.exists():
            try:
                import csv as csv_mod
                with open(self.csv_path, "r") as f:
                    reader = csv_mod.DictReader(f)
                    for row in reader:
                        # Only count clean rows as completed
                        if row.get("stab_error") or row.get("explainer") == "SKIPPED":
                            continue
                        s = row.get("scenario", "")
                        a = row.get("architecture", "")
                        b = row.get("balancing", "")
                        if s and a and b:
                            completed.add(f"{s}_{a}_{b}")
            except Exception as e:
                print(f"  WARNING: CSV resume check failed ({e})")

        # Source 3: Interrupted runs file
        interrupted_file = self.results_dir / ".interrupted_runs"
        if interrupted_file.exists():
            try:
                with open(interrupted_file) as f:
                    for line in f:
                        run_id = line.strip()
                        if run_id:
                            completed.add(run_id)
            except OSError:
                pass

        return completed

    def get_results_df(self):
        """Load results as a pandas DataFrame."""
        import pandas as pd
        if self.backend == "csv":
            return pd.read_csv(self.csv_path)
        elif self.backend == "mlflow" and MLFLOW_AVAILABLE:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            return mlflow.search_runs(experiment_ids=[experiment.experiment_id])

    def clean_experiment(self):
        """Delete all runs in the current experiment (for --clean flag)."""
        if self.backend == "mlflow" and MLFLOW_AVAILABLE:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment:
                runs = mlflow.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    output_format="list",
                )
                for run in runs:
                    mlflow.delete_run(run.info.run_id)
                print(f"  Cleaned {len(runs)} MLflow runs")

        # Also clean CSV if it exists
        csv_path = self.results_dir / f"{self.experiment_name}.csv"
        if csv_path.exists():
            csv_path.unlink()
            print(f"  Cleaned CSV: {csv_path}")
