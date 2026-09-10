"""
Re-evaluation script (inference-only).

Loads the 60 trained checkpoints produced by train_matrix.py and computes
ranking metrics the pipeline did NOT persist — ROC-AUC, PR-AUC (average
precision AND trapezoidal auc(recall, precision)), and precision@k — over
BOTH validation and test splits.

NO re-training. Pure inference: build the model, load_state_dict, one forward
pass, compute metrics from the illicit-class (index 1) softmax scores.

The loading machinery is copied verbatim from scripts/explain_matrix.py so that
the results are directly comparable to what the pipeline produced. The scoring
(softmax[:, 1], precision_recall_curve(..., pos_label=1) -> auc(recall,
precision)) is copied from src/training/trainer.py::evaluate so that
`pr_auc_trap` on the test split reproduces meta.json test_metrics.pr_auc,
validating the inference is correct.

Usage:
    ~/.local/bin/uv run python reeval_rocauc.py --limit 1 --device cpu
    ~/.local/bin/uv run python reeval_rocauc.py                # all 60 on GPU (auto)
    # Also dump PR/ROC curve points (for plotting the real curves):
    ~/.local/bin/uv run python reeval_rocauc.py --dump-curves

Output:
    results_v3/reeval_metrics.csv                 (always)
    results_v3/reeval_curves.csv                  (only with --dump-curves)

The repo root is auto-detected (two levels up from this file); override it with
the GNN_REPO_ROOT environment variable if needed.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

# Project root importable (so `src.*` resolves), mirroring explain_matrix.py.
# The script now lives in scripts/consolidacion/, so the repo root is two levels
# up. Override with the GNN_REPO_ROOT env var if you run it from elsewhere.
import os
REPO_ROOT = Path(os.environ.get("GNN_REPO_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    auc,
)

# Same imports explain_matrix.py uses for the load path.
from src.data.imbalance import create_imbalance_scenario
from src.data.loader import load_elliptic, print_dataset_stats
from src.data.preprocessing import preprocess
from src.training.trainer import build_model


# Columns for the output CSV, in order.
CSV_FIELDS = [
    "run_id",
    "scenario",
    "architecture",
    "balancing",
    "quality_passed",
    "split",
    "n_pos",
    "roc_auc",
    "pr_auc_ap",
    "pr_auc_trap",
    "prec_at_50",
    "prec_at_100",
    "prec_at_npos",
]

# Columns for the OPTIONAL curve-points CSV (long format). One block of
# `grid_x` values per (run_id, split): for the PR curve grid_x is recall and
# the y is pr_precision; for the ROC curve grid_x is the false-positive rate and
# the y is roc_tpr. Both curves share the same 0..1 grid so they average cleanly.
CURVE_FIELDS = [
    "run_id",
    "scenario",
    "architecture",
    "balancing",
    "quality_passed",
    "split",
    "grid_x",
    "pr_precision",
    "roc_tpr",
]


def parse_args():
    p = argparse.ArgumentParser(description="Re-evaluate checkpoints (inference only)")
    p.add_argument("--models-dir", type=str, default=str(REPO_ROOT / "results_models_v3"))
    p.add_argument("--out", type=str, default=str(REPO_ROOT / "results_v3" / "reeval_metrics.csv"))
    p.add_argument("--config", type=str, default=None,
                   help="Override YAML config path. Default: read from each meta's config_path.")
    p.add_argument("--device", type=str, default="auto",
                   help="'auto' | 'cpu' | 'cuda'. Force 'cpu' to avoid a busy GPU.")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N checkpoints (validation runs).")
    p.add_argument("--dump-curves", action="store_true",
                   help="Also write PR/ROC curve points (interpolated on a 0..1 "
                        "grid) for every processed checkpoint and split.")
    p.add_argument("--curves-out", type=str,
                   default=str(REPO_ROOT / "results_v3" / "reeval_curves.csv"),
                   help="Where to write the curve points when --dump-curves is set.")
    p.add_argument("--curve-points", type=int, default=101,
                   help="Number of grid points (0..1) per curve. Default 101.")
    return p.parse_args()


def resolve_device(arg: str) -> str:
    if arg == "cpu":
        return "cpu"
    if arg == "cuda":
        return "cuda"
    # auto
    return "cuda" if torch.cuda.is_available() else "cpu"


def precision_at_k(probs: np.ndarray, labels: np.ndarray, k: int):
    """Top-k precision: order by prob desc, fraction of true illicit in top-k.

    Edge handling: k is clamped to the number of available nodes. Returns
    (precision, k_used). If no nodes are available, returns (None, 0).
    """
    n = len(probs)
    if n == 0 or k <= 0:
        return None, 0
    k_used = min(k, n)
    top_idx = np.argsort(-probs)[:k_used]
    hits = int(labels[top_idx].sum())
    return hits / k_used, k_used


def compute_split_metrics(probs: np.ndarray, labels: np.ndarray) -> dict:
    """All ranking metrics for one split. Robust to degenerate splits."""
    n_pos = int(labels.sum())
    n = len(labels)
    n_classes = len(np.unique(labels))

    # ROC-AUC / PR-AUC need at least 2 classes and >=1 positive.
    if n_classes < 2 or n_pos == 0:
        roc = None
        pr_ap = None
        pr_trap = None
    else:
        roc = float(roc_auc_score(labels, probs))
        pr_ap = float(average_precision_score(labels, probs))
        # Trapezoidal auc over the precision-recall curve — EXACTLY as in
        # trainer.evaluate(), so this matches meta.json test_metrics.pr_auc.
        precision, recall, _ = precision_recall_curve(labels, probs, pos_label=1)
        pr_trap = float(auc(recall, precision))

    p50, _ = precision_at_k(probs, labels, 50)
    p100, _ = precision_at_k(probs, labels, 100)
    # precision@(n_pos): if n_pos==0 there is nothing to retrieve -> None.
    if n_pos > 0:
        pnpos, _ = precision_at_k(probs, labels, n_pos)
    else:
        pnpos = None

    return {
        "n_pos": n_pos,
        "roc_auc": roc,
        "pr_auc_ap": pr_ap,
        "pr_auc_trap": pr_trap,
        "prec_at_50": p50,
        "prec_at_100": p100,
        "prec_at_npos": pnpos,
    }


def curve_points_on_grid(probs: np.ndarray, labels: np.ndarray, n: int = 101):
    """PR and ROC curves resampled onto a shared 0..1 grid of `n` points.

    Returns (grid, pr_precision, roc_tpr):
      - PR curve: precision interpolated at each recall = grid value.
      - ROC curve: true-positive rate interpolated at each fpr = grid value.
    Resampling onto a fixed grid keeps the file small AND lets several
    configurations be averaged point-by-point later. Degenerate splits (a
    single class or no positives) yield NaN arrays.
    """
    grid = np.linspace(0.0, 1.0, n)
    if len(np.unique(labels)) < 2 or int(labels.sum()) == 0:
        nan = np.full(n, np.nan)
        return grid, nan, nan
    # PR curve — sklearn returns recall in decreasing order; sort ascending so
    # np.interp gets a monotonically increasing x-axis.
    precision, recall, _ = precision_recall_curve(labels, probs, pos_label=1)
    pr_precision = np.interp(grid, *_dedupe_x(recall, precision))
    # ROC curve — fpr is already non-decreasing, but it also carries ties.
    fpr, tpr, _ = roc_curve(labels, probs, pos_label=1)
    roc_tpr = np.interp(grid, *_dedupe_x(fpr, tpr))
    return grid, pr_precision, roc_tpr


def _dedupe_x(x: np.ndarray, y: np.ndarray):
    """Sort by x and collapse repeated x values keeping the highest y.

    `np.interp` needs a strictly increasing x-axis: with duplicates it silently
    returns the y of the LAST occurrence, which on a PR curve is the wrong one.
    sklearn emits two points at recall = 0, the sentinel (0, precision=1) and a
    degenerate (0, precision=0); without this collapse the resampled curve
    starts at precision 0 instead of 1 and the left edge of the plot, which is
    exactly where the eye lands, comes out upside down. Keeping the maximum y
    per x is also the usual convention for both curves, since it is the
    attainable precision (or TPR) at that operating point.
    """
    order = np.argsort(x, kind="stable")
    xs, ys = np.asarray(x, float)[order], np.asarray(y, float)[order]
    starts = np.flatnonzero(np.r_[True, xs[1:] != xs[:-1]])
    return xs[starts], np.maximum.reduceat(ys, starts)


@torch.no_grad()
def scores_for_split(model, data, mask_name: str, device: str):
    """Replicate trainer.evaluate() scoring for one split.

    probs = softmax(logits)[:, 1] on the masked nodes; labels = data.y[mask].
    """
    mask = getattr(data, mask_name).to(device)
    out = model(data.x.to(device), data.edge_index.to(device))
    labels = data.y.to(device)[mask].cpu().numpy()
    probs = F.softmax(out[mask], dim=-1)[:, 1].cpu().numpy()
    return probs, labels


def main():
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Device: {device}")

    models_dir = Path(args.models_dir)
    if not models_dir.exists():
        raise FileNotFoundError(f"Models dir not found: {models_dir}")

    meta_files = sorted(models_dir.glob("*_meta.json"))
    if not meta_files:
        raise FileNotFoundError(f"No *_meta.json in {models_dir}")
    if args.limit is not None:
        meta_files = meta_files[: args.limit]
    print(f"Checkpoints to process: {len(meta_files)}")

    # Resolve dataset root from a config. All v3 metas point at ./data, but we
    # read it from the (first) meta's config_path to stay faithful to the run.
    if args.config is not None:
        cfg_path = Path(args.config)
    else:
        with open(meta_files[0]) as f:
            first_meta = json.load(f)
        cfg_path = REPO_ROOT / first_meta["config_path"]
    with open(cfg_path) as f:
        config = yaml.safe_load(f)
    data_root = config["data"]["root"]
    print(f"Config: {cfg_path}")
    print(f"Dataset root: {data_root}")

    # Load + preprocess ONCE (exactly as explain_matrix.py).
    print("\n" + "=" * 70)
    print("LOADING ELLIPTIC DATASET")
    print("=" * 70)
    data_raw = load_elliptic(root=data_root)
    print_dataset_stats(data_raw)
    preprocess(data_raw)

    rows = []
    curve_rows = []
    for i, meta_file in enumerate(meta_files, 1):
        with open(meta_file) as f:
            meta = json.load(f)

        run_id = meta["run_id"]
        arch = meta["architecture"]
        bp = meta["best_params"]
        print(f"\n[{i}/{len(meta_files)}] {run_id}  (arch={arch})")

        # Arch-specific kwargs — same guard as explain_matrix.py.
        arch_kwargs = {}
        if arch == "GAT" and "heads" in bp:
            arch_kwargs["heads"] = bp["heads"]
        if arch == "TAGCN" and "K" in bp:
            arch_kwargs["K"] = bp["K"]

        # Recreate the imbalance scenario (seed=42) — consistency with the run.
        # NOTE: only train_mask is subsampled; val/test masks are untouched, but
        # we recreate the scenario anyway to mirror explain_matrix.py exactly.
        data = create_imbalance_scenario(data_raw, meta["imbalance_ratio"], seed=42)

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
            print(f"  ERROR: checkpoint missing: {ckpt_file}. Skipping.")
            continue
        model.load_state_dict(
            torch.load(ckpt_file, map_location=device, weights_only=True)
        )
        model = model.to(device)
        model.eval()

        for split, mask_name in (("val", "val_mask"), ("test", "test_mask")):
            probs, labels = scores_for_split(model, data, mask_name, device)
            m = compute_split_metrics(probs, labels)
            row = {
                "run_id": run_id,
                "scenario": meta["scenario"],
                "architecture": arch,
                "balancing": meta["balancing"],
                "quality_passed": meta.get("quality_passed", False),
                "split": split,
                **m,
            }
            rows.append(row)

            if args.dump_curves:
                grid, pr_prec, roc_tpr = curve_points_on_grid(
                    probs, labels, n=args.curve_points
                )
                for gx, pp, rt in zip(grid, pr_prec, roc_tpr):
                    curve_rows.append({
                        "run_id": run_id,
                        "scenario": meta["scenario"],
                        "architecture": arch,
                        "balancing": meta["balancing"],
                        "quality_passed": meta.get("quality_passed", False),
                        "split": split,
                        "grid_x": f"{gx:.4f}",
                        "pr_precision": "" if np.isnan(pp) else f"{pp:.6f}",
                        "roc_tpr": "" if np.isnan(rt) else f"{rt:.6f}",
                    })

            def _fmt(x):
                return "None" if x is None else f"{x:.6f}"

            print(
                f"  {split:4s}: n_pos={m['n_pos']:5d}  "
                f"roc={_fmt(m['roc_auc'])}  ap={_fmt(m['pr_auc_ap'])}  "
                f"trap={_fmt(m['pr_auc_trap'])}  "
                f"p@50={_fmt(m['prec_at_50'])} p@100={_fmt(m['prec_at_100'])} "
                f"p@npos={_fmt(m['prec_at_npos'])}"
            )

        # Validation cross-check against the pipeline's stored PR-AUC.
        stored = meta.get("test_metrics", {}).get("pr_auc")
        test_row = next(r for r in reversed(rows) if r["split"] == "test")
        if stored is not None and test_row["pr_auc_trap"] is not None:
            diff = abs(stored - test_row["pr_auc_trap"])
            status = "MATCH" if diff < 1e-4 else "MISMATCH"
            print(
                f"  [CHECK] test pr_auc_trap={test_row['pr_auc_trap']:.8f} vs "
                f"meta.test_metrics.pr_auc={stored:.8f}  |diff|={diff:.2e}  -> {status}"
            )

    # Write CSV.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\nWrote {len(rows)} rows ({len(rows)//2} checkpoints × 2 splits) -> {out_path}")

    # Write curve points CSV (optional).
    if args.dump_curves:
        curves_path = Path(args.curves_out)
        curves_path.parent.mkdir(parents=True, exist_ok=True)
        with open(curves_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CURVE_FIELDS)
            writer.writeheader()
            for r in curve_rows:
                writer.writerow(r)
        print(f"Wrote {len(curve_rows)} curve points -> {curves_path}")


if __name__ == "__main__":
    main()
