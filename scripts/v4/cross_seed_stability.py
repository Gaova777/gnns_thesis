"""
Stability ACROSS MODEL SEEDS (pipeline v4, CPU only).

For every (scenario, arch, balancing, explainer) that has rankings for several model
seeds (42/43/44), compare node by node the MEAN RANKING of each seed: per node, the
replica score vectors are converted to average ranks (ties → mean rank) and averaged
over replicas; then Spearman between seeds. Features for GNNExplainer/ShapleyFeatures,
edge masks (on the edges both seeds share) for PGExplainer and GNNExplainer.

Inputs : {results_dir}/rankings/*.npz  (written by scripts/explain_matrix.py)
Output : {results_dir}/elliptic_v4_cross_seed.csv, one row per (config, seed pair):
  scenario, arch, balancing, explainer, seed_a, seed_b,
  cs_spearman_full       mean over common nodes of Spearman(mean feature rank a, b)
  cs_spearman_edges      same on edge masks (edges present in both seeds)
  cs_primary             edges for PGExplainer, features otherwise
  n_nodes, n_measurable_feat, n_measurable_edges, n_both_tp
  cs_spearman_full_tp    only nodes that are TP for both seeds
  gate_a, gate_b         gate of each model (from elliptic_v4_stability.csv, if present)

Usage:
  uv run --frozen python scripts/v4/cross_seed_stability.py [--results-dir ./results_v4]
                                                            [--gate-only]
"""

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.explainability.v4_explain import canonical_explainer  # noqa: E402


def mean_rank(scores: np.ndarray) -> np.ndarray | None:
    """[R, n] scores → mean over valid replicas of average ranks (1 = least important)."""
    if scores is None or scores.ndim != 2 or scores.shape[1] == 0:
        return None
    valid = ~np.all(np.isnan(scores), axis=1)
    if not valid.any():
        return None
    return np.mean([rankdata(np.nan_to_num(r)) for r in scores[valid]], axis=0)


def _rho(a, b) -> float:
    if a is None or b is None or len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def load_rankings(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k].item() if z[k].ndim == 0 else z[k] for k in z.files}
    d["explainer"] = canonical_explainer(str(d["explainer"]))
    nodes = {}
    for i, n in enumerate(d["nodes"]):
        ent = {"tp": bool(d["tp"][i]), "ok": not str(d["node_reason"][i]),
               "feat": None, "edge": None}
        if "feat" in d:
            ent["feat"] = mean_rank(d["feat"][i])
        if "edge" in d:
            a, b = int(d["edge_ptr"][i]), int(d["edge_ptr"][i + 1])
            ent["edge_ids"] = d["edge_ids"][a:b]
            ent["edge_scores"] = d["edge"][:, a:b]
        nodes[int(n)] = ent
    d["node_data"] = nodes
    return d


def _edge_rho(ea: dict, eb: dict) -> float:
    if "edge_ids" not in ea or "edge_ids" not in eb:
        return float("nan")
    common, ia, ib = np.intersect1d(ea["edge_ids"], eb["edge_ids"], return_indices=True)
    if len(common) < 2:
        return float("nan")
    return _rho(mean_rank(ea["edge_scores"][:, ia]), mean_rank(eb["edge_scores"][:, ib]))


def compare(a: dict, b: dict) -> dict:
    common = sorted(set(a["node_data"]) & set(b["node_data"]))
    rf, re, rtp = [], [], []
    for n in common:
        na, nb = a["node_data"][n], b["node_data"][n]
        if not (na["ok"] and nb["ok"]):
            continue
        f = _rho(na["feat"], nb["feat"])
        e = _edge_rho(na, nb)
        if f == f:
            rf.append(f)
            if na["tp"] and nb["tp"]:
                rtp.append(f)
        if e == e:
            re.append(e)
    m = lambda v: float(np.mean(v)) if v else float("nan")  # noqa: E731
    out = {"n_nodes": len(common), "cs_spearman_full": m(rf), "cs_spearman_edges": m(re),
           "cs_spearman_full_tp": m(rtp), "n_measurable_feat": len(rf),
           "n_measurable_edges": len(re),
           "n_both_tp": sum(a["node_data"][n]["tp"] and b["node_data"][n]["tp"]
                            for n in common)}
    out["cs_primary"] = (out["cs_spearman_edges"] if a["explainer"] == "PGExplainer"
                         else out["cs_spearman_full"])
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--results-dir", default="./results_v4")
    p.add_argument("--gate-only", action="store_true",
                   help="only models with gate_passed=True in elliptic_v4_stability.csv")
    args = p.parse_args(argv)
    res = Path(args.results_dir)
    files = sorted((res / "rankings").glob("*.npz"))
    if not files:
        sys.exit(f"no rankings in {res / 'rankings'}")

    gates = {}
    stab = res / "elliptic_v4_stability.csv"
    if stab.exists():
        s = pd.read_csv(stab)
        gates = {(str(r.run_id), canonical_explainer(r.explainer)):
                 str(r.gate_passed).strip().lower() in ("true", "1")
                 for r in s.itertuples() if r.status == "ok"}

    groups: dict = {}
    for f in files:
        d = load_rankings(f)
        g = gates.get((str(d["run_id"]), d["explainer"]))
        if args.gate_only and g is not True:
            continue
        d["gate"] = g
        key = (str(d["scenario"]), str(d["arch"]), str(d["balancing"]), d["explainer"])
        groups.setdefault(key, {})[int(d["seed"])] = d

    rows = []
    for key, by_seed in sorted(groups.items()):
        for sa, sb in combinations(sorted(by_seed), 2):
            r = compare(by_seed[sa], by_seed[sb])
            rows.append({"scenario": key[0], "arch": key[1], "balancing": key[2],
                         "explainer": key[3], "seed_a": sa, "seed_b": sb,
                         "gate_a": by_seed[sa]["gate"], "gate_b": by_seed[sb]["gate"], **r})
    cols = ["scenario", "arch", "balancing", "explainer", "seed_a", "seed_b", "cs_primary",
            "cs_spearman_full", "cs_spearman_full_tp", "cs_spearman_edges", "n_nodes",
            "n_measurable_feat", "n_measurable_edges", "n_both_tp", "gate_a", "gate_b"]
    out = pd.DataFrame(rows, columns=cols)
    out_path = res / "elliptic_v4_cross_seed.csv"
    out.to_csv(out_path, index=False)
    print(f"{len(groups)} configurations, {len(out)} seed pairs -> {out_path}")
    if len(out):
        print(out.groupby(["explainer", "arch"])["cs_primary"].mean().round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
