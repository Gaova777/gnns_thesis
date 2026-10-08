"""
Training matrix runner (Script 1 of 2: train + quality gate; explainers live in explain_matrix.py).

Runs scenarios × architectures × balancing × seeds with warm-started Optuna, trains a final
model with the best hyperparameters and saves its checkpoint + metadata.

Pipeline v4 (configs/experiment_v4.yaml, ``scenarios.mode: v4``):
  * Correct Elliptic labels (0 licit, 1 illicit, -1 unknown) with hard count checks
    (src/data/loader.py); the run stops on any mismatch.
  * Label mode (data.label_mode or --label-mode): C "licit_unknown" (main: licit + unknown
    are the negative class), B "licit" or A "unknown". Every model is also scored on the
    common yardsticks of meta["cross_label_eval"] (illicit vs licit, vs unknown, vs all),
    which is what scripts/v4/compare_label_modes.py compares across A/B/C.
  * v4 scenarios (src/data/imbalance.py::create_v4_scenario): only the train mask changes,
    subsampled with a FIXED data seed (scenarios.subsample_seed, default 2026) so every
    model seed sees the same subset. The scenario report goes into meta.json.
  * Decision threshold calibrated on validation only (no test-prevalence resampling).
  * Progress events in {logs_dir}/progress.jsonl via src.monitoring.RunLog (stage "train").
  * A CUDA OOM, a per-config timeout (--max-minutes-per-config) or any other per-config
    exception is logged and the matrix CONTINUES with the next config. Only data-integrity
    errors (wrong labels / scenario counts) stop the run.
Legacy v3 configs (``scenarios.imbalance_ratios``) still work with the old scenario code.

Artifacts per config (in tracking.models_dir):
  - {safe_run_id}_best.pt    model.state_dict()      (safe_run_id = run_id with ':' -> '-')
  - {safe_run_id}_meta.json  metadata (written atomically; --resume keys on it)

Usage:
    uv run --frozen python scripts/train_matrix.py --config configs/experiment_v4.yaml --seed 42
    uv run --frozen python scripts/train_matrix.py --config configs/experiment_v4.yaml --seed 43 --reuse-hp --resume
    uv run --frozen python scripts/train_matrix.py --config ... --arch GAT --resume
    # CPU smoke: 1 scenario / GCN / none, 1 Optuna trial, 2 epochs, separate output dirs
    uv run --frozen python scripts/train_matrix.py --config configs/experiment_v4.yaml \
        --scenario native --arch GCN --balancing none --seed 42 --trials 1 --epochs 2 \
        --models-dir results_models_v4_smoke --results-dir results_v4_smoke --no-mlflow
"""

import argparse
import gc
import hashlib
import json
import os
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
from src.balancing.losses import get_loss_function
from src.data.imbalance import (
    V4_SUBSAMPLE_SEED,
    create_imbalance_scenario,
    create_v4_scenario,
)
from src.data.loader import (
    DataIntegrityError,
    EXPECTED_COUNTS,
    ILLICIT,
    LICIT,
    UNKNOWN,
    apply_label_mode,
    load_elliptic,
    print_dataset_stats,
    resolve_label_mode,
)
from src.data.preprocessing import preprocess
from src.monitoring.progress import RunLog
from src.training.hyperopt import run_hyperopt
from src.training.trainer import (
    ConfigTimeoutError,
    Trainer,
    build_model,
    make_deadline_callback,
)

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

_interrupted = False

# Consecutive non-OOM/non-timeout failures after which the matrix aborts: one bad config
# should not kill a night-long chain, but the same bug on every config should.
MAX_CONSECUTIVE_ERRORS = 3


def _signal_handler(signum, frame):
    global _interrupted
    _interrupted = True
    sig_name = signal.Signals(signum).name
    tqdm.write(f"\n[SIGNAL] {sig_name} received — finishing current config then exiting.")
    tqdm.write("         Resume with: --resume")


def parse_args():
    p = argparse.ArgumentParser(description="Train matrix (no explainers)")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda")
    p.add_argument("--resume", action="store_true",
                   help="Skip configs whose meta.json already exists")
    p.add_argument("--quick", action="store_true",
                   help="Smoke test: 5 epochs, default HPs (no Optuna), 1 scenario/arch/balancing")
    p.add_argument("--arch", type=str, default=None, help="Filter to one arch")
    p.add_argument("--scenario", type=str, default=None, help="Filter to one scenario")
    p.add_argument("--balancing", type=str, default=None, help="Filter to one balancing")
    p.add_argument("--max-hours", type=float, default=10.0)
    p.add_argument("--max-minutes-per-config", type=float, default=90.0,
                   help="Wall-clock budget per config (Optuna + final training), checked "
                        "between epochs. On overrun the config is logged as an error and the "
                        "matrix continues. <=0 disables it.")
    p.add_argument("--no-warm-start", action="store_true",
                   help="Disable Optuna warm-start (random exploration only)")
    p.add_argument("--seed", type=int, default=None,
                   help="Train only this model seed (default: every seed in training.seeds).")
    p.add_argument("--reuse-hp", action="store_true",
                   help="For non-canonical seeds, reuse the hyperparameters found for the "
                        "canonical (first) seed instead of re-running Optuna.")
    # Overrides (smoke tests / debugging)
    p.add_argument("--epochs", type=int, default=None, help="Override training.epochs "
                   "(also caps Optuna trial epochs)")
    p.add_argument("--hp-from", type=str, default=None,
                   help="Reuse, for EVERY seed, the hyperparameters of the canonical-seed meta.json "
                        "of the same configuration found in this models dir (no search).")
    p.add_argument("--early-stop-metric", type=str, default=None,
                   help="Override training.early_stop_metric (f1 | mcc | pr_auc)")
    p.add_argument("--trials", type=int, default=None,
                   help="Override optuna_trials (0 = default HPs, no search)")
    p.add_argument("--models-dir", type=str, default=None, help="Override tracking.models_dir")
    p.add_argument("--results-dir", type=str, default=None, help="Override tracking.results_dir")
    p.add_argument("--logs-dir", type=str, default=None, help="Override tracking.logs_dir")
    p.add_argument("--no-mlflow", action="store_true", help="Use the CSV tracker backend")
    p.add_argument("--label-mode", type=str, default=None,
                   help="Override data.label_mode: licit_unknown (C) | licit (B) | unknown "
                        "(A); aliases C/B/A")
    return p.parse_args()


def _safe_name(s: str) -> str:
    return s.replace(":", "-").replace("/", "-")


def _meta_path(models_dir: Path, run_id: str) -> Path:
    return models_dir / f"{_safe_name(run_id)}_meta.json"


def _ckpt_path(models_dir: Path, run_id: str) -> Path:
    return models_dir / f"{_safe_name(run_id)}_best.pt"


def _write_json_atomic(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def _free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _legacy_report(data, name, ratio) -> dict:
    m = data.train_mask
    n_il = int((data.y[m] == 1).sum())
    n_lic = int((data.y[m] == 0).sum())
    return {"scenario": name, "kind": "legacy_v3", "illicit_to_licit_target": ratio,
            "n_illicit": n_il, "n_licit": n_lic, "total": n_il + n_lic,
            "ratio": n_il / n_lic if n_lic else float("inf")}


def main():
    args = parse_args()
    pipeline_start = time.time()
    deadline_sec = args.max_hours * 3600

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    # encoding explicit: YAML comments are non-ASCII and Windows defaults to cp1252.
    config_bytes = Path(args.config).read_bytes()
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    config = yaml.safe_load(config_bytes.decode("utf-8"))
    config_version = str(config.get("version", "v3"))

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device} | config {args.config} (version {config_version}, "
          f"sha256 {config_sha256[:12]})")

    scen_cfg = config["scenarios"]
    v4_mode = scen_cfg.get("mode") == "v4"

    # ── CLI overrides ────────────────────────────────────────────────────────
    hyp_cfg = config["models"]["hyperparameter_search"]
    train_cfg = config["training"]
    if args.quick:
        train_cfg["epochs"] = 5
        train_cfg["patience"] = 5
        hyp_cfg["optuna_trials"] = 0
        config["models"]["architectures"] = config["models"]["architectures"][:1]
        config["balancing"]["techniques"] = config["balancing"]["techniques"][:1]
        if v4_mode:
            scen_cfg["names"] = scen_cfg["names"][:1]
        else:
            scen_cfg["imbalance_ratios"] = scen_cfg["imbalance_ratios"][:1]
    if args.epochs is not None:
        train_cfg["epochs"] = args.epochs
    if args.trials is not None:
        hyp_cfg["optuna_trials"] = args.trials
    tracking_cfg = config["tracking"]
    if args.models_dir:
        tracking_cfg["models_dir"] = args.models_dir
    if args.results_dir:
        tracking_cfg["results_dir"] = args.results_dir
    if args.logs_dir:
        tracking_cfg["logs_dir"] = args.logs_dir
    if args.no_mlflow:
        tracking_cfg["backend"] = "csv"
    if args.label_mode:
        config["data"]["label_mode"] = resolve_label_mode(args.label_mode)
    effective_sha256 = hashlib.sha256(
        json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()

    models_dir = Path(tracking_cfg.get("models_dir", "./results_models_v3"))
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = Path(tracking_cfg.get("logs_dir", "./runs"))
    runlog = RunLog(logs_dir, stage="train")

    tracker = ExperimentTracker(
        backend=tracking_cfg["backend"],
        experiment_name=tracking_cfg["experiment_name"],
        results_dir=tracking_cfg["results_dir"],
    )

    # ── Dataset: load + validate + preprocess once ───────────────────────────
    print("\n" + "=" * 70)
    print("LOADING ELLIPTIC DATASET")
    print("=" * 70)
    t0 = time.time()
    data_raw = load_elliptic(root=config["data"]["root"])  # hard label-count checks
    print_dataset_stats(data_raw)
    dcfg = config["data"]
    # Legacy v3 configs have no label_mode: they keep the reviewed-licit negatives (B).
    label_mode = apply_label_mode(
        data_raw, dcfg.get("label_mode", "licit_unknown" if v4_mode else "licit"))
    print(f"  Label mode: {label_mode} (negatives = "
          f"{ {'licit_unknown': 'licit + unknown', 'licit': 'licit', 'unknown': 'unknown'}[label_mode]})")
    preprocess(
        data_raw,
        train_range=tuple(dcfg.get("train_timesteps", (1, 34))),
        val_range=tuple(dcfg.get("val_timesteps", (35, 42))),
        test_range=tuple(dcfg.get("test_timesteps", (43, 49))),
    )  # hard split-count checks
    split_counts = {
        n: {"illicit": int((data_raw.y[getattr(data_raw, f"{n}_mask")] == 1).sum()),
            "licit": int((data_raw.y[getattr(data_raw, f"{n}_mask")] == 0).sum())}
        for n in ("train", "val", "test")
    }
    print(f"  Data loaded + validated in {time.time() - t0:.1f}s: {split_counts}")
    split_ranges = {n: tuple(dcfg.get(f"{n}_timesteps", d)) for n, d in
                    (("train", (1, 34)), ("val", (35, 42)), ("test", (43, 49)))}

    # ── Experimental matrix (with CLI filters) ───────────────────────────────
    if v4_mode:
        subsample_seed = int(scen_cfg.get("subsample_seed", V4_SUBSAMPLE_SEED))
        scenarios = {n: None for n in scen_cfg["names"]}
    else:
        subsample_seed = None
        scenarios = {s["name"]: s["illicit_to_licit"] for s in scen_cfg["imbalance_ratios"]}
    archs = [m["name"] for m in config["models"]["architectures"]]
    balances = [t["name"] for t in config["balancing"]["techniques"]]

    # Model seeds. The FIRST seed in training.seeds is CANONICAL: its run_id has no suffix;
    # every other seed gets `_s{seed}`. Seeds are the outer loop so an interrupted sweep
    # leaves whole seeds finished.
    config_seeds = train_cfg.get("seeds", [42]) or [42]
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
                        (scenario_name, ratio, arch_name, balance_name, seed, run_id))

    if not all_configs:
        print("No configs matched the given filters. Nothing to do.")
        return

    total = len(all_configs)
    print(f"\nTraining matrix: {total} configs "
          f"({len(seeds)} seed(s): {', '.join(str(s) for s in seeds)})")
    runlog.plan(total, items=[c[-1] for c in all_configs])

    opt_trials = hyp_cfg.get("optuna_trials", 50)
    opt_metric = hyp_cfg.get("optuna_metric", "pr_auc")
    warm_start = hyp_cfg.get("warm_start", True) and not args.no_warm_start
    epochs = train_cfg.get("epochs", 600)
    patience = train_cfg.get("patience", 50)
    early_stop_metric = args.early_stop_metric or train_cfg.get("early_stop_metric", "f1")
    trial_epochs = min(int(hyp_cfg.get("trial_epochs", 50)), epochs)
    trial_patience = min(int(hyp_cfg.get("trial_patience", 10)), patience)
    search_space = {k: hyp_cfg[k] for k in ("num_layers", "dropout", "learning_rate",
                                            "weight_decay") if k in hyp_cfg}

    gate_cfg = config.get("analysis", {}).get("quality_gate", {})
    gate_f1 = gate_cfg.get("f1_min", 0.70)
    gate_mcc = gate_cfg.get("mcc_min", 0.40)

    focal_cfg = next(
        (t for t in config["balancing"]["techniques"] if t["name"] == "focal_loss"),
        {"gamma": 2.0, "alpha": 0.75},
    )

    run_ctx = dict(
        args=args, device=device, tracker=tracker, runlog=runlog, models_dir=models_dir,
        canonical_seed=canonical_seed, v4_mode=v4_mode, subsample_seed=subsample_seed,
        opt_trials=opt_trials, opt_metric=opt_metric, warm_start=warm_start,
        epochs=epochs, patience=patience, early_stop_metric=early_stop_metric,
        trial_epochs=trial_epochs, trial_patience=trial_patience,
        search_space=search_space, hidden_choices=hyp_cfg.get("hidden_dim"),
        gate_f1=gate_f1, gate_mcc=gate_mcc, focal_cfg=focal_cfg,
        config_sha256=config_sha256, effective_sha256=effective_sha256,
        config_version=config_version, split_counts=split_counts,
        label_mode=label_mode, split_ranges=split_ranges,
    )

    summary = {"passed": [], "failed": [], "skipped_resume": [], "errors": []}
    consecutive_errors = 0

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
        pbar.set_postfix_str(run_id, refresh=True)

        meta_file = _meta_path(models_dir, run_id)
        if args.resume and meta_file.exists():
            tqdm.write(f"  SKIP (resume — {meta_file.name} exists): {run_id}")
            runlog.skip(run_id, f"resume: {meta_file.name} exists")
            summary["skipped_resume"].append(run_id)
            continue

        reuse_hp_src = None
        if args.hp_from or (args.reuse_hp and seed != canonical_seed):
            hp_dir = Path(args.hp_from) if args.hp_from else models_dir
            base_meta = _meta_path(hp_dir, f"{scenario_name}_{arch_name}_{balance_name}")
            if not base_meta.exists():
                msg = f"--reuse-hp: missing {base_meta.name} (seed {canonical_seed})"
                tqdm.write(f"  SKIP ({msg}): {run_id}")
                runlog.skip(run_id, msg)
                summary["failed"].append(f"{run_id} (no canonical HP)")
                continue
            with open(base_meta, encoding="utf-8") as f:
                reuse_hp_src = json.load(f)
            reuse_hp_src["_path"] = str(base_meta) if args.hp_from else base_meta.name

        tqdm.write(f"\n{'='*70}\nCONFIG: {run_id}\n{'='*70}")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        budget_min = args.max_minutes_per_config
        cfg_deadline = (time.monotonic() + budget_min * 60) if budget_min and budget_min > 0 \
            else None

        error_kind = None
        try:
            with runlog.step(run_id) as rec:
                meta = run_one_config(
                    run_ctx, data_raw, scenario_name, ratio, arch_name, balance_name,
                    seed, run_id, reuse_hp_src, cfg_deadline,
                )
                rec["metrics"] = meta["_progress_metrics"]
            meta.pop("_progress_metrics")
            (summary["passed"] if meta["quality_passed"] else summary["failed"]).append(run_id)
            consecutive_errors = 0
        except DataIntegrityError:
            raise  # wrong data must stop everything
        except torch.cuda.OutOfMemoryError as exc:
            error_kind = f"OOM: {str(exc)[:200]}"
        except ConfigTimeoutError as exc:
            error_kind = f"TIMEOUT (> {budget_min} min): {exc}"
        except Exception as exc:  # noqa: BLE001 - logged by RunLog.step, matrix continues
            error_kind = f"{type(exc).__name__}: {str(exc)[:300]}"
            consecutive_errors += 1

        if error_kind is not None:
            _free_memory()
            tqdm.write(f"  ERROR in {run_id} — {error_kind}. Continuing with next config.")
            summary["errors"].append(f"{run_id} ({error_kind.split(':')[0]})")
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                tqdm.write(f"  {consecutive_errors} consecutive non-OOM errors — aborting "
                           "(likely a systematic bug, see progress.jsonl traces).")
                break
        _free_memory()

    pbar.close()

    print("\n" + "=" * 70)
    print("TRAIN MATRIX SUMMARY")
    print("=" * 70)
    print(f"  Passed (val F1>={gate_f1}, MCC>={gate_mcc}): {len(summary['passed'])}")
    for r in summary["passed"]:
        print(f"    + {r}")
    print(f"  Failed quality gate:                        {len(summary['failed'])}")
    for r in summary["failed"]:
        print(f"    - {r}")
    print(f"  Errors (OOM / timeout / exception):         {len(summary['errors'])}")
    for r in summary["errors"]:
        print(f"    ! {r}")
    if summary["skipped_resume"]:
        print(f"  Skipped (resume):                           {len(summary['skipped_resume'])}")
    print(f"\n  Total time: {(time.time() - pipeline_start) / 3600:.2f} h")
    print(f"  Checkpoints + metadata in: {models_dir}")
    print(f"  Progress log: {runlog.path}")


@torch.no_grad()
def cross_label_eval(model, data, threshold: float, split_ranges: dict, device) -> dict:
    """Score the model on yardsticks that do not depend on the label mode.

    For val and test (by timestep, real nodes only) and using the 3-class truth ``data.y3``:
      vs_licit    illicit vs reviewed licit     (the label-mode B problem)
      vs_unknown  illicit vs unknown            (the label-mode A problem, v3)
      vs_all      illicit vs licit + unknown    (the label-mode C problem)
    each with PR-AUC, and F1/MCC/recall/precision at ``threshold`` (calibrated on the
    model's own validation set), plus the fraction of each class flagged as illicit
    (flag_rate). If A/B/C models agree here, unknowns can be treated as licit.
    """
    from sklearn.metrics import (auc, f1_score, matthews_corrcoef, precision_recall_curve,
                                 precision_score, recall_score)
    import torch.nn.functional as F

    model.eval()
    out = model(data.x.to(device), data.edge_index.to(device))
    probs = F.softmax(out, dim=-1)[:, 1].cpu()
    y3 = data.y3.cpu()
    ts = data.timestep.cpu()
    real = ~data.is_synthetic.cpu() if hasattr(data, "is_synthetic") else torch.ones_like(y3, dtype=torch.bool)

    def _score(mask):
        lab = (y3[mask] == ILLICIT).numpy().astype(int)
        p = probs[mask].numpy()
        pred = (p >= threshold).astype(int)
        prec, rec, _ = precision_recall_curve(lab, p, pos_label=1)
        return {"n_illicit": int(lab.sum()), "n_negative": int(len(lab) - lab.sum()),
                "pr_auc": float(auc(rec, prec)),
                "f1": float(f1_score(lab, pred, zero_division=0)),
                "mcc": float(matthews_corrcoef(lab, pred)),
                "recall": float(recall_score(lab, pred, zero_division=0)),
                "precision": float(precision_score(lab, pred, zero_division=0))}

    res = {"threshold": float(threshold)}
    for split in ("val", "test"):
        lo, hi = split_ranges[split]
        in_split = real & (ts >= lo) & (ts <= hi)
        il, li, un = (in_split & (y3 == c) for c in (ILLICIT, LICIT, UNKNOWN))
        res[split] = {
            "vs_licit": _score(il | li),
            "vs_unknown": _score(il | un),
            "vs_all": _score(il | li | un),
            "flag_rate": {name: float((probs[m] >= threshold).float().mean())
                          for name, m in (("illicit", il), ("licit", li), ("unknown", un))},
            "mean_prob": {name: float(probs[m].mean())
                          for name, m in (("illicit", il), ("licit", li), ("unknown", un))},
        }
    return res


def run_one_config(ctx, data_raw, scenario_name, ratio, arch_name, balance_name, seed,
                   run_id, reuse_hp_src, cfg_deadline) -> dict:
    """Train one config end to end and write its meta.json. Returns the meta dict."""
    args, device, tracker = ctx["args"], ctx["device"], ctx["tracker"]
    runlog, models_dir = ctx["runlog"], ctx["models_dir"]
    t_start = time.monotonic()

    # Model seed: fixes Optuna trial init + final training init. The DATA subset does not
    # depend on it in v4 (fixed subsample_seed).
    torch.manual_seed(seed)
    np.random.seed(seed)

    if ctx["v4_mode"]:
        data, scenario_report = create_v4_scenario(
            data_raw, scenario_name, subsample_seed=ctx["subsample_seed"])
    else:
        data = create_imbalance_scenario(data_raw, ratio, seed=seed)
        scenario_report = _legacy_report(data, scenario_name, ratio)

    last_beat = [0.0]

    def _beat(phase):
        def _b(epoch):
            now = time.monotonic()
            if now - last_beat[0] >= 15:
                runlog.beat(run_id, f"{phase} epoch {epoch}")
                last_beat[0] = now
        return _b

    focal_cfg = ctx["focal_cfg"]
    opt_trials, opt_metric = ctx["opt_trials"], ctx["opt_metric"]

    # ── Hyperparameters ─────────────────────────────────────────────────────
    # --reuse-hp: non-canonical seeds reuse the canonical seed's HPs, isolating model-init
    # variance (what the seed sweep measures) from search variance.
    if reuse_hp_src is not None:
        best_hp = reuse_hp_src["best_params"]
        best_score = reuse_hp_src.get("optuna_best_score")
        hp_source = f"reused:{ctx['canonical_seed']}" + (f":{reuse_hp_src['_path']}" if '/' in reuse_hp_src['_path'] else '')
        tqdm.write(f"  Hyperparams: reused from seed {ctx['canonical_seed']} "
                   f"({reuse_hp_src['_path']}): {best_hp}")
    elif opt_trials == 0:
        best_hp = {"hidden_dim": 64, "num_layers": 2, "dropout": 0.3,
                   "lr": 0.001, "weight_decay": 5e-4}
        best_score = None
        hp_source = "defaults"
        tqdm.write("  Hyperparams: defaults (trials=0 / --quick)")
    else:
        tqdm.write(f"  Optuna: {opt_trials} trials x <= {ctx['trial_epochs']} epochs, "
                   f"metric={opt_metric}, warm_start={ctx['warm_start']}")
        remaining = (cfg_deadline - time.monotonic()) if cfg_deadline else None
        res = run_hyperopt(
            data, arch_name, balancing=balance_name,
            n_trials=opt_trials, device=device,
            epochs=ctx["trial_epochs"], patience=ctx["trial_patience"],
            metric=opt_metric,
            focal_gamma=focal_cfg.get("gamma", 2.0),
            focal_alpha=focal_cfg.get("alpha", 0.75),
            warm_start=ctx["warm_start"],
            hidden_dim_choices=ctx["hidden_choices"],
            search_space=ctx["search_space"],
            epoch_callback=make_deadline_callback(cfg_deadline, run_id, _beat("optuna")),
            timeout_s=max(remaining, 1.0) if remaining is not None else None,
            show_progress_bar=False,
        )
        best_hp = res["best_params"]
        best_score = res["best_score"]
        hp_source = "optuna"
        tqdm.write(f"  Best Optuna val {opt_metric}={best_score:.4f}: {best_hp}")

    # ── Final training ──────────────────────────────────────────────────────
    torch.manual_seed(seed)
    loss_fn = get_loss_function(
        balance_name, data.y, data.train_mask,
        gamma=focal_cfg.get("gamma", 2.0), alpha=focal_cfg.get("alpha", 0.75),
        device=device,
    )
    # Libera la VRAM acumulada por Optuna antes del reentrenamiento final: si no,
    # GAT (grafo completo) puede quedarse sin memoria para el modelo final. DECISIONES.md.
    import gc as _gc
    _gc.collect()
    if str(device) != "cpu":
        torch.cuda.empty_cache()
    arch_kwargs = {}
    if arch_name == "GAT" and "heads" in best_hp:
        arch_kwargs["heads"] = best_hp["heads"]
    if arch_name == "TAGCN" and "K" in best_hp:
        arch_kwargs["K"] = best_hp["K"]
    model = build_model(
        arch_name, in_channels=data.num_node_features,
        hidden_channels=best_hp.get("hidden_dim", 128),
        num_layers=best_hp.get("num_layers", 2),
        dropout=best_hp.get("dropout", 0.3),
        **arch_kwargs,
    )
    lr = best_hp.get("lr", 0.001)
    wd = best_hp.get("weight_decay", 5e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    epochs, patience = ctx["epochs"], ctx["patience"]
    train_params = {
        "scenario": scenario_name, "architecture": arch_name, "balancing": balance_name,
        "seed": seed, "hidden_channels": best_hp.get("hidden_dim"),
        "num_layers": best_hp.get("num_layers"), "dropout": best_hp.get("dropout"),
        "lr": lr, "weight_decay": wd, "epochs": epochs, "patience": patience,
        "early_stop_metric": ctx["early_stop_metric"], "optuna_metric": opt_metric,
        "n_train_illicit": scenario_report["n_illicit"],
        "n_train_licit": scenario_report["n_licit"],
    }
    for k in ("heads", "K"):
        if k in best_hp:
            train_params[k] = best_hp[k]

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
            early_stop_metric=ctx["early_stop_metric"],
            epoch_callback=make_deadline_callback(cfg_deadline, run_id, _beat("train")),
        )
        results = trainer.train(data, epochs=epochs, run_name=run_id)
        test_metrics = results["test_metrics"]

        # Validation metrics of the selected (best-epoch) model, argmax decision. Re-evaluated
        # after the best checkpoint is loaded, so it is correct even after a resumed run.
        val_argmax = trainer.evaluate(data, mask_name="val_mask")

        # Threshold calibration on VALIDATION ONLY (v4). v3 resampled val to the test
        # prevalence first, which leaks test-label information into model selection.
        calib = trainer.calibrate_threshold(data, mask_name="val_mask")
        thr = calib["threshold"]
        val_calibrated = trainer.evaluate(data, mask_name="val_mask", threshold=thr)
        test_calibrated = trainer.evaluate(data, mask_name="test_mask", threshold=thr)
        tqdm.write(
            f"  Val (argmax): F1={val_argmax['f1']:.4f} MCC={val_argmax['mcc']:.4f} "
            f"PR-AUC={val_argmax['pr_auc']:.4f} | Test PR-AUC={test_metrics['pr_auc']:.4f}\n"
            f"  Calibrated on val (t={thr:.2f}): val F1={calib['f1']:.4f} | "
            f"test F1={test_calibrated['f1']:.4f} MCC={test_calibrated['mcc']:.4f}"
        )
        tracker.log_test_metrics(test_calibrated)
        cross_eval = cross_label_eval(trainer.model, data, thr, ctx["split_ranges"], device)
        t_ce = cross_eval["test"]
        tqdm.write(
            f"  Cross-label test PR-AUC: vs licit={t_ce['vs_licit']['pr_auc']:.4f} "
            f"vs unknown={t_ce['vs_unknown']['pr_auc']:.4f} vs all={t_ce['vs_all']['pr_auc']:.4f}"
            f" | unknown flagged={t_ce['flag_rate']['unknown']:.3%}"
        )

    # Quality gate on VALIDATION (argmax at the best epoch), as in v3: test is hit by the
    # temporal covariate shift (dark-market shutdown), val shows whether the model learned.
    val_f1, val_mcc = float(val_argmax["f1"]), float(val_argmax["mcc"])
    quality_passed = bool(val_f1 >= ctx["gate_f1"] and val_mcc >= ctx["gate_mcc"])
    tqdm.write(f"  Quality gate (val F1>={ctx['gate_f1']}, val MCC>={ctx['gate_mcc']}): "
               f"{'PASSED' if quality_passed else 'FAILED'}")

    duration_s = round(time.monotonic() - t_start, 1)
    meta = {
        "run_id": run_id,
        "pipeline_version": ctx["config_version"],
        "scenario": scenario_name,
        "scenario_mode": "v4" if ctx["v4_mode"] else "legacy_v3",
        # v4: ACTUAL illicit/licit ratio of the train mask (informative only). To rebuild the
        # train mask use create_v4_scenario(data, scenario, subsample_seed) — NOT
        # create_imbalance_scenario(imbalance_ratio).
        "imbalance_ratio": scenario_report["ratio"] if ctx["v4_mode"] else ratio,
        "subsample_seed": ctx["subsample_seed"],
        "scenario_report": scenario_report,
        "n_train_illicit": scenario_report["n_illicit"],
        "n_train_licit": scenario_report["n_licit"],
        "label_mode": ctx["label_mode"],
        "label_encoding": {"licit": 0, "illicit": 1, "unknown": -1},
        "negative_class": {"licit_unknown": "licit+unknown", "licit": "licit",
                           "unknown": "unknown"}[ctx["label_mode"]],
        "cross_label_eval": cross_eval,
        "data_counts": {"total": EXPECTED_COUNTS, "splits": ctx["split_counts"]},
        "architecture": arch_name,
        "balancing": balance_name,
        "seed": seed,
        "best_params": best_hp,
        "optuna_best_score": best_score,
        "optuna_metric": opt_metric,
        "hp_source": hp_source,
        "early_stop_metric": ctx["early_stop_metric"],
        "best_epoch": results["best_epoch"],
        "epochs_run": results["epochs_run"],
        "best_val_score": results["best_val_score"],
        "best_val_mcc": results["best_val_mcc"],
        "val_metrics": val_argmax,
        "val_metrics_calibrated": val_calibrated,
        # test_metrics = reported (threshold calibrated on val only)
        "test_metrics": test_calibrated,
        "test_metrics_argmax": test_metrics,
        "calibrated_threshold": thr,
        "threshold_calibration": "val_only",
        "val_f1_at_threshold": calib["f1"],
        "calibration_resample": None,
        "quality_passed": quality_passed,
        "quality_gate": {"f1_min": ctx["gate_f1"], "mcc_min": ctx["gate_mcc"],
                         "evaluated_on": "val", "decision": "argmax"},
        "val_f1_best_epoch": val_f1,
        "val_mcc_best_epoch": val_mcc,
        "mlflow_run_id": mlflow_run_id,
        "checkpoint": _ckpt_path(models_dir, run_id).name,
        "config_path": args.config,
        "config_sha256": ctx["config_sha256"],
        "effective_config_sha256": ctx["effective_sha256"],
        "cli_overrides": {k: getattr(args, k) for k in
                          ("epochs", "trials", "quick", "max_minutes_per_config")},
        "device": device,
        "duration_s": duration_s,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _write_json_atomic(_meta_path(models_dir, run_id), meta)
    tqdm.write(f"  Saved metadata: {_meta_path(models_dir, run_id)} ({duration_s:.0f}s)")

    meta["_progress_metrics"] = {
        "val_pr_auc": round(float(val_argmax["pr_auc"]), 4),
        "val_f1": round(val_f1, 4),
        "val_mcc": round(val_mcc, 4),
        "test_pr_auc": round(float(test_calibrated["pr_auc"]), 4),
        "test_pr_auc_vs_licit": round(float(cross_eval["test"]["vs_licit"]["pr_auc"]), 4),
        "unknown_flag_rate_test": round(float(cross_eval["test"]["flag_rate"]["unknown"]), 4),
        "gate_passed": quality_passed,
        "label_mode": ctx["label_mode"],
        "n_train_illicit": scenario_report["n_illicit"],
        "n_train_licit": scenario_report["n_licit"],
        "epochs_run": results["epochs_run"],
    }
    return meta


if __name__ == "__main__":
    main()
