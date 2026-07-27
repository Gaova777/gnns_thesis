"""
Training matrix runner (Pipeline v3 — Script 1 of 2).

Responsibility: run the full experimental matrix (scenarios × architectures ×
balancing) with literature-backed warm-start Optuna + expanded search space,
then train a final model with the best hyperparameters and save its checkpoint.

Does NOT run any explainability methods — that is Script 2 (explain_matrix.py).

Artifacts produced per config (in `models_dir` from config):
  - {scenario}_{arch}_{balancing}_best.pt    — model.state_dict()
  - {scenario}_{arch}_{balancing}_meta.json  — best_params, test_metrics,
                                                quality_passed, mlflow_run_id, seed

Usage:
    uv run python scripts/train_matrix.py --config configs/experiment_machineB_v3.yaml
    uv run python scripts/train_matrix.py --config ... --resume
    uv run python scripts/train_matrix.py --config ... --arch GCN --scenario "1:1" --balancing class_weighting
    uv run python scripts/train_matrix.py --config ... --quick   # smoke
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

import torch
import yaml
from tqdm import tqdm

from src.analysis.tracking import ExperimentTracker
from src.balancing.losses import get_loss_function
from src.data.imbalance import create_imbalance_scenario
from src.data.loader import load_elliptic, print_dataset_stats
from src.data.preprocessing import preprocess
from src.training.hyperopt import run_hyperopt
from src.training.trainer import Trainer, build_model

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

_interrupted = False
_current_run_id: str | None = None


def _signal_handler(signum, frame):
    global _interrupted
    _interrupted = True
    sig_name = signal.Signals(signum).name
    tqdm.write(f"\n[SIGNAL] {sig_name} received — finishing current config then exiting.")
    tqdm.write("         Resume with: --resume")


def parse_args():
    p = argparse.ArgumentParser(description="Train matrix (no explainers)")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--resume", action="store_true",
                   help="Skip configs whose meta.json already exists")
    p.add_argument("--quick", action="store_true",
                   help="Smoke test: 5 epochs, 2 Optuna trials, 1 scenario/arch/balancing")
    p.add_argument("--arch", type=str, default=None, help="Filter to one arch")
    p.add_argument("--scenario", type=str, default=None, help="Filter to one scenario")
    p.add_argument("--balancing", type=str, default=None, help="Filter to one balancing")
    p.add_argument("--max-hours", type=float, default=10.0)
    p.add_argument("--no-warm-start", action="store_true",
                   help="Disable Optuna warm-start (random exploration only)")
    p.add_argument("--seed", type=int, default=None,
                   help="Train only this model seed (default: every seed in config training.seeds). "
                        "Useful to split the seed sweep across sessions or machines.")
    p.add_argument("--reuse-hp", action="store_true",
                   help="For non-canonical seeds, reuse the hyperparameters found for the canonical "
                        "seed instead of re-running Optuna. Isolates seed variance from search "
                        "variance and cuts most of the cost of a seed sweep.")
    return p.parse_args()


def _safe_name(s: str) -> str:
    return s.replace(":", "-").replace("/", "-")


def _meta_path(models_dir: Path, run_id: str) -> Path:
    return models_dir / f"{_safe_name(run_id)}_meta.json"


def _ckpt_path(models_dir: Path, run_id: str) -> Path:
    return models_dir / f"{_safe_name(run_id)}_best.pt"


def main():
    global _current_run_id
    args = parse_args()
    pipeline_start = time.time()
    deadline_sec = args.max_hours * 3600

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    # encoding is explicit: the YAML configs contain non-ASCII characters in comments, and on a
    # default-locale Windows Python (cp1252) an implicit open() raises UnicodeDecodeError.
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Quick-mode rewrite (smoke only)
    if args.quick:
        config["training"]["epochs"] = 5
        config["training"]["patience"] = 5
        config["models"]["hyperparameter_search"]["optuna_trials"] = 2
        config["models"]["architectures"] = config["models"]["architectures"][:1]
        config["balancing"]["techniques"] = config["balancing"]["techniques"][:1]
        config["scenarios"]["imbalance_ratios"] = config["scenarios"]["imbalance_ratios"][:1]

    tracking_cfg = config["tracking"]
    models_dir = Path(tracking_cfg.get("models_dir", "./results_models_v3"))
    models_dir.mkdir(parents=True, exist_ok=True)

    tracker = ExperimentTracker(
        backend=tracking_cfg["backend"],
        experiment_name=tracking_cfg["experiment_name"],
        results_dir=tracking_cfg["results_dir"],
    )

    # Dataset — load + preprocess once
    print("\n" + "=" * 70)
    print("LOADING ELLIPTIC DATASET")
    print("=" * 70)
    data_raw = load_elliptic(root=config["data"]["root"])
    print_dataset_stats(data_raw)
    preprocess(data_raw)

    # Experimental matrix (with CLI filters)
    scenarios = {s["name"]: s["illicit_to_licit"] for s in config["scenarios"]["imbalance_ratios"]}
    archs = [m["name"] for m in config["models"]["architectures"]]
    balances = [t["name"] for t in config["balancing"]["techniques"]]

    # Model seeds. The FIRST seed declared in the config is the CANONICAL one: its artifacts
    # keep the historical un-suffixed run_id, so the published run (checkpoints, meta.json and
    # the CSV rows backing the manuscript) keeps its exact identity and --resume still finds
    # it. Every additional seed gets its own namespace via a _s{seed} suffix. Seeds are the
    # outer loop so an interrupted sweep still leaves whole seeds finished, not fragments.
    config_seeds = config["training"].get("seeds", [42]) or [42]
    canonical_seed = config_seeds[0]
    seeds = [args.seed] if args.seed is not None else list(config_seeds)

    all_configs = []
    for seed in seeds:
        for scenario_name, ratio in scenarios.items():
            if args.scenario and scenario_name != args.scenario:
                continue
            for arch_name in archs:
                if args.arch and arch_name != args.arch:
                    continue
                for balance_name in balances:
                    if args.balancing and balance_name != args.balancing:
                        continue
                    run_id = f"{scenario_name}_{arch_name}_{balance_name}"
                    if seed != canonical_seed:
                        run_id = f"{run_id}_s{seed}"
                    all_configs.append(
                        (scenario_name, ratio, arch_name, balance_name, seed, run_id)
                    )

    if not all_configs:
        print("No configs matched the given filters. Nothing to do.")
        return

    total = len(all_configs)
    print(f"\nTraining matrix: {total} configs "
          f"({len(seeds)} seed(s): {', '.join(str(s) for s in seeds)})")

    hyp_cfg = config["models"]["hyperparameter_search"]
    opt_trials = hyp_cfg.get("optuna_trials", 50)
    opt_metric = hyp_cfg.get("optuna_metric", "pr_auc")
    warm_start = hyp_cfg.get("warm_start", True) and not args.no_warm_start

    train_cfg = config["training"]
    epochs = train_cfg.get("epochs", 600)
    patience = train_cfg.get("patience", 50)
    early_stop_metric = train_cfg.get("early_stop_metric", "f1")
    # NOTE: `seed` is per-config now (see the matrix build above), not a single global.

    gate_cfg = config.get("analysis", {}).get("quality_gate", {})
    gate_f1 = gate_cfg.get("f1_min", 0.70)
    gate_mcc = gate_cfg.get("mcc_min", 0.40)

    # Summary for final report
    summary = {"passed": [], "failed": [], "skipped_resume": []}

    pbar = tqdm(all_configs, desc="Train matrix", unit="config",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

    for scenario_name, ratio, arch_name, balance_name, seed, run_id in pbar:
        if _interrupted:
            tqdm.write("  Interrupt — stopping.")
            break

        elapsed = time.time() - pipeline_start
        if (deadline_sec - elapsed) < 1800:
            tqdm.write(f"  DEADLINE: {elapsed/3600:.1f}h elapsed, <30 min remaining — stopping.")
            break

        _current_run_id = run_id
        pbar.set_postfix_str(run_id, refresh=True)

        # --resume: skip if metadata already exists
        meta_file = _meta_path(models_dir, run_id)
        if args.resume and meta_file.exists():
            tqdm.write(f"  SKIP (resume — {meta_file.name} exists): {run_id}")
            summary["skipped_resume"].append(run_id)
            continue

        tqdm.write(f"\n{'='*70}")
        tqdm.write(f"CONFIG: {run_id}")
        tqdm.write(f"{'='*70}")

        data = create_imbalance_scenario(data_raw, ratio, seed=seed)

        # ── Hyperopt (Optuna with warm-start) ──────────────────────────────────
        # --reuse-hp: for a non-canonical seed, take the hyperparameters already found for the
        # canonical seed instead of re-running the search. This is the cheaper AND cleaner way
        # to run a seed sweep: holding the hyperparameters fixed isolates the variance due to
        # model initialisation and data subsampling, which is what the sweep is measuring, from
        # the variance of the search itself. (The search is near-invariant across seeds anyway:
        # hyperopt.py pins TPESampler(seed=42) regardless of the run seed.)
        reuse_hp_src = None
        if args.reuse_hp and seed != canonical_seed:
            base_run_id = f"{scenario_name}_{arch_name}_{balance_name}"
            base_meta = _meta_path(models_dir, base_run_id)
            if not base_meta.exists():
                tqdm.write(f"  SKIP (--reuse-hp: missing {base_meta.name} for seed "
                           f"{canonical_seed}): {run_id}")
                summary["failed"].append(f"{run_id} (no canonical HP)")
                continue
            with open(base_meta, encoding="utf-8") as f:
                reuse_hp_src = json.load(f)

        if reuse_hp_src is not None:
            best_hp = reuse_hp_src["best_params"]
            best_score = reuse_hp_src.get("optuna_best_score")
            tqdm.write(f"  Hyperparams: reused from seed {canonical_seed} "
                       f"({base_meta.name}): {best_hp}")
        elif args.quick or opt_trials == 0:
            best_hp = {
                "hidden_dim": 64, "num_layers": 2,
                "dropout": 0.3, "lr": 0.001, "weight_decay": 5e-4,
            }
            best_score = None
            tqdm.write("  Hyperparams: defaults (quick / trials=0)")
        else:
            tqdm.write(f"  Optuna: {opt_trials} trials, metric={opt_metric}, "
                       f"warm_start={warm_start}")
            # Use shorter trials for search (cheaper exploration).
            # 50 epochs + patience 10 is enough to see learning curve and reject
            # bad HPs without running forever (GAT/TAGCN observed ~10s/epoch).
            hyp_epochs = min(50, epochs)
            hyp_patience = min(10, patience)
            focal_cfg = next(
                (t for t in config["balancing"]["techniques"] if t["name"] == "focal_loss"),
                {"gamma": 2.0, "alpha": 0.75},
            )
            # Respect config's hidden_dim choices (prevents slow GAT at hidden=256)
            hidden_choices = hyp_cfg.get("hidden_dim", None)
            res = run_hyperopt(
                data, arch_name, balancing=balance_name,
                n_trials=opt_trials, device=device,
                epochs=hyp_epochs, patience=hyp_patience,
                metric=opt_metric,
                focal_gamma=focal_cfg.get("gamma", 2.0),
                focal_alpha=focal_cfg.get("alpha", 0.75),
                warm_start=warm_start,
                hidden_dim_choices=hidden_choices,
            )
            best_hp = res["best_params"]
            best_score = res["best_score"]
            tqdm.write(f"  Best Optuna val {opt_metric}={best_score:.4f}: {best_hp}")

        # ── Final training with best hyperparameters ───────────────────────────
        torch.manual_seed(seed)
        focal_cfg = next(
            (t for t in config["balancing"]["techniques"] if t["name"] == "focal_loss"),
            {"gamma": 2.0, "alpha": 0.75},
        )
        loss_fn = get_loss_function(
            balance_name, data.y, data.train_mask,
            gamma=focal_cfg.get("gamma", 2.0),
            alpha=focal_cfg.get("alpha", 0.75),
            device=device,
        )

        arch_kwargs = {}
        if arch_name == "GAT" and "heads" in best_hp:
            arch_kwargs["heads"] = best_hp["heads"]
        if arch_name == "TAGCN" and "K" in best_hp:
            arch_kwargs["K"] = best_hp["K"]

        model = build_model(
            arch_name,
            in_channels=data.num_node_features,
            hidden_channels=best_hp.get("hidden_dim", 128),
            num_layers=best_hp.get("num_layers", 2),
            dropout=best_hp.get("dropout", 0.3),
            **arch_kwargs,
        )

        lr = best_hp.get("lr", 0.001)
        wd = best_hp.get("weight_decay", 5e-4)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        train_params = {
            "scenario": scenario_name, "architecture": arch_name,
            "balancing": balance_name,
            "hidden_channels": best_hp.get("hidden_dim"),
            "num_layers": best_hp.get("num_layers"),
            "dropout": best_hp.get("dropout"),
            "lr": lr, "weight_decay": wd,
            "epochs": epochs, "patience": patience,
            "early_stop_metric": early_stop_metric,
            "optuna_metric": opt_metric,
        }
        if "heads" in best_hp:
            train_params["heads"] = best_hp["heads"]
        if "K" in best_hp:
            train_params["K"] = best_hp["K"]

        mlflow_run_id = None
        with tracker.training_run(run_id, params=train_params):
            try:
                import mlflow  # type: ignore
                run = mlflow.active_run()
                mlflow_run_id = run.info.run_id if run else None
            except Exception:
                pass

            trainer = Trainer(
                model, loss_fn, optimizer, device,
                patience=patience, tracker=tracker,
                checkpoint_dir=str(models_dir),
                early_stop_metric=early_stop_metric,
            )
            results = trainer.train(
                data, epochs=epochs, run_name=run_id,
            )
            test_metrics = results["test_metrics"]
            tqdm.write(
                f"  Test (argmax):     F1={test_metrics['f1']:.4f} "
                f"MCC={test_metrics['mcc']:.4f} PR-AUC={test_metrics['pr_auc']:.4f}"
            )

            # Threshold calibration with prevalence matching.
            # Calibrating F1 on raw val picks a threshold optimal for val's
            # class ratio (1:30). But test has 1:136, so that threshold
            # over-tightens on val and under-recalls on test. We resample val
            # to match test prevalence BEFORE sweeping — produces a threshold
            # tuned for the distribution we'll actually face in test.
            calib = trainer.calibrate_threshold(
                data, mask_name="val_mask", match_ratio_mask="test_mask",
            )
            calibrated_threshold = calib["threshold"]
            val_f1_calibrated = calib["f1"]
            resample_info = calib.get("resample") or {}
            test_metrics_calibrated = trainer.evaluate(
                data, mask_name="test_mask", threshold=calibrated_threshold,
            )
            resample_str = ""
            if resample_info.get("applied"):
                a = resample_info["after"]
                resample_str = (f" [val resampled to {a['illicit']}+{a['licit']} "
                                f"matching test ratio]")
            tqdm.write(
                f"  Calibrated (t={calibrated_threshold:.2f}, "
                f"val F1={val_f1_calibrated:.4f}{resample_str}): "
                f"Test F1={test_metrics_calibrated['f1']:.4f} "
                f"MCC={test_metrics_calibrated['mcc']:.4f} "
                f"PR-AUC={test_metrics_calibrated['pr_auc']:.4f}"
            )

            # MLflow sees the calibrated metrics (the ones we actually report).
            tracker.log_test_metrics(test_metrics_calibrated)

        # Quality gate — evaluate on VAL metrics, not test.
        #
        # Rationale: Elliptic has severe temporal covariate shift between train
        # (timesteps 1-34) and test (43-49) due to the "dark market shutdown"
        # event, well documented in literature. Simple GCN/GraphSAGE cannot
        # bridge that shift without specialized temporal architectures. Test F1
        # is therefore affected by shift rather than by learning quality.
        #
        # For XAI stability analysis, the property we need is "the model
        # learned fraud patterns" — validated by val F1/MCC, which reflect
        # generalization to timesteps the model was optimized against via early
        # stopping. Test metrics are reported + stored, but not used as filter.
        val_f1 = results["history"]["val_f1"][results["best_epoch"] - 1] \
            if results["history"]["val_f1"] else 0.0
        val_mcc = results["best_val_mcc"]
        quality_passed = (val_f1 >= gate_f1 and val_mcc >= gate_mcc)
        status = "PASSED" if quality_passed else "FAILED"
        tqdm.write(
            f"  Quality gate (val F1≥{gate_f1}, val MCC≥{gate_mcc}): {status} "
            f"(val F1={val_f1:.4f}, val MCC={val_mcc:.4f})"
        )

        if quality_passed:
            summary["passed"].append(run_id)
        else:
            summary["failed"].append(run_id)

        # ── Persist metadata JSON (alongside .pt checkpoint already written
        #     by Trainer at best epoch) ──────────────────────────────────────
        meta = {
            "run_id": run_id,
            "scenario": scenario_name,
            "imbalance_ratio": ratio,
            "architecture": arch_name,
            "balancing": balance_name,
            "seed": seed,
            "best_params": best_hp,
            "optuna_best_score": best_score,
            "optuna_metric": opt_metric,
            # provenance of best_params: "optuna" (search ran here), "reused:<seed>" (inherited
            # from the canonical seed via --reuse-hp), or "defaults" (quick / trials=0).
            "hp_source": (f"reused:{canonical_seed}" if reuse_hp_src is not None
                          else ("defaults" if (args.quick or opt_trials == 0) else "optuna")),
            "early_stop_metric": early_stop_metric,
            "best_epoch": results["best_epoch"],
            "best_val_score": results["best_val_score"],
            "best_val_mcc": results["best_val_mcc"],
            # test_metrics = canonical (what we report, i.e. calibrated)
            "test_metrics": test_metrics_calibrated,
            # Preserved for analysis: what argmax gave before calibration
            "test_metrics_argmax": test_metrics,
            "calibrated_threshold": calibrated_threshold,
            "val_f1_at_threshold": val_f1_calibrated,
            "calibration_resample": resample_info,
            "quality_passed": quality_passed,
            "quality_gate": {"f1_min": gate_f1, "mcc_min": gate_mcc, "evaluated_on": "val"},
            "val_f1_best_epoch": float(val_f1),
            "val_mcc_best_epoch": float(val_mcc),
            "mlflow_run_id": mlflow_run_id,
            "checkpoint": str(_ckpt_path(models_dir, run_id).name),
            "config_path": args.config,
        }
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        tqdm.write(f"  Saved metadata: {meta_file}")

    pbar.close()

    # ── Final summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TRAIN MATRIX SUMMARY")
    print("=" * 70)
    print(f"  Passed (F1>={gate_f1}, MCC>={gate_mcc}): {len(summary['passed'])}")
    for r in summary["passed"]:
        print(f"    ✓ {r}")
    print(f"  Failed quality gate:                    {len(summary['failed'])}")
    for r in summary["failed"]:
        print(f"    ✗ {r}")
    if summary["skipped_resume"]:
        print(f"  Skipped (resume):                       {len(summary['skipped_resume'])}")

    total_elapsed_h = (time.time() - pipeline_start) / 3600
    print(f"\n  Total time: {total_elapsed_h:.2f} h")
    print(f"  Checkpoints + metadata in: {models_dir}")
    print("\n  Next step: run scripts/explain_matrix.py on the passing models.")


if __name__ == "__main__":
    main()
