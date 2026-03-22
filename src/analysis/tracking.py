"""
Experiment tracking and logging.

Supports MLflow for rich experiment tracking or CSV fallback
for lightweight, portable logging.
"""

import json
import csv
import os
from pathlib import Path
from datetime import datetime
from typing import Optional

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class ExperimentTracker:
    """
    Unified experiment tracker supporting MLflow or CSV backends.

    Args:
        backend: "mlflow" or "csv".
        experiment_name: Name of the experiment.
        results_dir: Base directory for results.
    """

    def __init__(
        self,
        backend: str = "csv",
        experiment_name: str = "xai-gnn-stability",
        results_dir: str = "./results",
    ):
        self.backend = backend
        self.experiment_name = experiment_name
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        if backend == "mlflow" and MLFLOW_AVAILABLE:
            mlflow.set_experiment(experiment_name)
            print(f"  MLflow experiment: {experiment_name}")
        elif backend == "mlflow" and not MLFLOW_AVAILABLE:
            print("  MLflow not installed, falling back to CSV")
            self.backend = "csv"

        if self.backend == "csv":
            self.csv_path = self.results_dir / f"{experiment_name}.csv"
            self._csv_initialized = self.csv_path.exists()

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
        """
        Log a single experiment run.

        Args:
            scenario: Imbalance scenario (e.g., "1:10").
            architecture: GNN arch name.
            balancing: Balancing technique.
            explainer: XAI method.
            seed: Random seed.
            predictive_metrics: Dict with f1, mcc, pr_auc, etc.
            stability_metrics: Dict with jaccard, spearman, etc.
            hyperparams: Model hyperparameters.
            tags: Additional metadata tags.
        """
        run_data = {
            "timestamp": datetime.now().isoformat(),
            "scenario": scenario,
            "architecture": architecture,
            "balancing": balancing,
            "explainer": explainer,
            "seed": seed,
            **{f"pred_{k}": v for k, v in predictive_metrics.items()},
        }

        if stability_metrics:
            for k, v in stability_metrics.items():
                if isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        if not isinstance(sub_v, (list, dict)):
                            run_data[f"stab_{k}_{sub_k}"] = sub_v
                else:
                    run_data[f"stab_{k}"] = v

        if self.backend == "mlflow" and MLFLOW_AVAILABLE:
            self._log_mlflow(run_data, hyperparams, tags)
        else:
            self._log_csv(run_data)

    def _log_mlflow(self, run_data: dict, hyperparams: dict = None, tags: dict = None):
        """Log to MLflow."""
        with mlflow.start_run():
            # Log params
            for key in ["scenario", "architecture", "balancing", "explainer", "seed"]:
                mlflow.log_param(key, run_data[key])
            if hyperparams:
                mlflow.log_params(hyperparams)

            # Log metrics
            for k, v in run_data.items():
                if isinstance(v, (int, float)) and k not in ["seed"]:
                    mlflow.log_metric(k, v)

            # Log tags
            if tags:
                mlflow.set_tags(tags)

    def _log_csv(self, run_data: dict):
        """Log to CSV file."""
        file_exists = self.csv_path.exists()

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=run_data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(run_data)

    def get_results_df(self):
        """Load results as a pandas DataFrame."""
        import pandas as pd

        if self.backend == "csv":
            return pd.read_csv(self.csv_path)
        elif self.backend == "mlflow" and MLFLOW_AVAILABLE:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            return mlflow.search_runs(experiment_ids=[experiment.experiment_id])
