"""
Tests of the v4 explanation stage (CPU only).

Runs without pytest (not a project dependency):
    uv run --frozen python tests/test_explain_v4.py              # (a) unit tests, seconds
    uv run --frozen python tests/test_explain_v4.py --e2e        # + (b) end-to-end on Elliptic
    uv run --frozen python tests/test_explain_v4.py --all        # (a) + (b) + (c) analysis selftest
With pytest installed, ``pytest tests/test_explain_v4.py`` collects the unit tests; the
end-to-end test runs there only with the env var ``V4_E2E=1``.

(a) metrics: empty → NaN, ties, Jaccard with <2 edges → NaN, identical rankings → 1;
    batched ShapleyFeatures == legacy estimator; common node set persisted and reused;
    stratified PGExplainer sample.
(b) explain_matrix end to end: two tiny GCNs (2 CPU epochs, seeds 42/43) + one gated-out
    model, 3 common nodes, 2 replicas, the 3 explainers with the v4 parameters; checks
    CSV, npz, progress.jsonl, --resume, and cross_seed_stability.py. Prints timings.
(c) scripts/v4/analyze_elliptic_v4.py --selftest.
"""

import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.stability import metrics as M

nan = float("nan")


def isnan(v):
    return isinstance(v, float) and math.isnan(v)


# ─────────────────────────────── (a) metrics ────────────────────────────────

def test_empty_is_nan_not_zero():
    assert isnan(M.spearman_rank_agreement(np.array([]), np.array([1, 2])))
    r = M.spearman_full(None)
    assert isnan(r["mean"]) and r["reason"] == M.REASON_NA
    r = M.spearman_edges(None)
    assert isnan(r["mean"]) and r["reason"] == M.REASON_NA
    r = M.jaccard_edges_topk(None)
    assert isnan(r["mean"]) and r["reason"] == M.REASON_NA
    # one replica only
    r = M.spearman_full(np.random.rand(1, 10))
    assert isnan(r["mean"]) and r["reason"] == M.REASON_FEW_REPLICAS
    # constant vector (e.g. all-zero mask) → NaN, not 0
    r = M.spearman_full(np.zeros((3, 10)))
    assert isnan(r["mean"]) and r["reason"] == M.REASON_CONSTANT
    # PGExplainer-like node: no feature output → spearman_full NaN "no_aplica"
    n = M.node_stability_v4(None, np.random.rand(3, 5))
    assert isnan(n["spearman_full"]) and n["spearman_full_reason"] == M.REASON_NA
    assert not isnan(n["spearman_edges"])


def test_identical_rankings_are_one():
    v = np.random.RandomState(0).rand(165)
    s = np.stack([v, v, v])
    assert abs(M.spearman_full(s)["mean"] - 1) < 1e-12
    assert abs(M.spearman_topk(s, 20)["mean"] - 1) < 1e-12
    e = np.stack([v[:30]] * 3)
    assert abs(M.spearman_edges(e)["mean"] - 1) < 1e-12
    assert M.jaccard_edges_topk(e, 20)["mean"] == 1.0


def test_ties_use_average_rank():
    # 3 distinct values + 7 tied zeros. Average ranks → the two replicas agree exactly
    # even though argsort would order the tied zeros differently.
    a = np.array([3, 2, 1, 0, 0, 0, 0, 0, 0, 0], float)
    b = a.copy()
    assert abs(M.spearman_full(np.stack([a, b]))["mean"] - 1) < 1e-12
    # reversing the non-tied part lowers the correlation, ties stay tied
    c = np.array([1, 2, 3, 0, 0, 0, 0, 0, 0, 0], float)
    rho = M.spearman_full(np.stack([a, c]))["mean"]
    from scipy.stats import spearmanr, rankdata
    assert abs(rho - spearmanr(rankdata(a), rankdata(c)).statistic) < 1e-12
    assert 0 < rho < 1
    # the v3 top-k metric on 165 features with k=20: 145 tied at the worst rank
    rng = np.random.RandomState(1)
    x, y = rng.rand(165), rng.rand(165)
    top = M.spearman_topk(np.stack([x, y]), 20)["mean"]
    full = M.spearman_full(np.stack([x, y]))["mean"]
    assert not isnan(top) and not isnan(full)


def test_jaccard_needs_two_edges():
    r = M.jaccard_edges_topk(np.random.rand(3, 1), 20)
    assert isnan(r["mean"]) and r["reason"] == M.REASON_FEW_EDGES
    r = M.jaccard_edges_topk(np.zeros((3, 0)), 20)
    assert isnan(r["mean"]) and r["reason"] == M.REASON_FEW_EDGES
    r = M.spearman_edges(np.random.rand(3, 1))
    assert isnan(r["mean"]) and r["reason"] == M.REASON_FEW_EDGES
    # 2 edges: k_eff = 1, so replicas CAN disagree (v3's k=20 made this always 1)
    r = M.jaccard_edges_topk(np.array([[0.9, 0.1], [0.1, 0.9]]), 20)
    assert r["k_eff"] == 1 and r["mean"] == 0.0
    # the legacy function keeps its old contract
    assert M.jaccard_index(set(), set()) == 0.0


def test_aggregate_ignores_nan_and_reports_reason():
    rows = [{"m": nan, "m_reason": M.REASON_NA}, {"m": nan, "m_reason": M.REASON_NA}]
    a = M.aggregate_nodes_v4(rows, "m")
    assert isnan(a["mean"]) and a["reason"] == M.REASON_NA and a["n"] == 0
    rows.append({"m": 0.5, "m_reason": None})
    a = M.aggregate_nodes_v4(rows, "m")
    assert a["mean"] == 0.5 and a["n"] == 1
    a = M.aggregate_nodes_v4(rows, "m", mask=[True, False, False])
    assert isnan(a["mean"])


# ──────────────────────── (a) explainability helpers ─────────────────────────

def _toy_graph(n=40, f=6, seed=0):
    from torch_geometric.data import Data
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, f, generator=g)
    ei = torch.randint(0, n, (2, 3 * n), generator=g)
    y = torch.randint(0, 2, (n,), generator=g)
    d = Data(x=x, edge_index=ei, y=y)
    d.train_mask = torch.arange(n) < n // 2
    d.val_mask = torch.arange(n) >= n // 2
    return d


def test_shapley_batched_matches_legacy():
    from src.explainability.shap_runner import (
        compute_shap_values_permutation, shapley_features_subgraph)
    from src.training.trainer import build_model
    from torch_geometric.utils import k_hop_subgraph
    d = _toy_graph()
    torch.manual_seed(0)
    model = build_model("GCN", in_channels=6, hidden_channels=8, num_layers=2).eval()
    node = 3
    legacy = compute_shap_values_permutation(model, d, node, num_samples=4, seed=7,
                                             num_hops=3)
    subset, sei, mapping, _ = k_hop_subgraph(node, 3, d.edge_index, relabel_nodes=True,
                                             num_nodes=d.num_nodes)
    new = shapley_features_subgraph(model, d.x[subset], sei, int(mapping[0]),
                                    num_samples=4, seed=7, max_nodes_per_forward=50)
    assert np.allclose(legacy, new, atol=1e-6), (legacy, new)


def test_gradient_explainers():
    """IntegratedGradients satisfies completeness on the node's own features and is
    deterministic; ExpectedGradients depends only on its seed."""
    from src.explainability.gradients import (
        expected_gradients_subgraph, integrated_gradients_subgraph, reference_pool)
    from src.training.trainer import build_model
    from torch_geometric.utils import k_hop_subgraph
    d = _toy_graph()
    for arch in ("GCN", "GraphSAGE"):
        torch.manual_seed(0)
        model = build_model(arch, in_channels=6, hidden_channels=8, num_layers=2).eval()
        node = 3
        subset, sei, mapping, _ = k_hop_subgraph(node, 2, d.edge_index, relabel_nodes=True,
                                                 num_nodes=d.num_nodes)
        x, t = d.x[subset], int(mapping[0])
        phi, err = integrated_gradients_subgraph(model, x, sei, t, steps=200)
        phi2, _ = integrated_gradients_subgraph(model, x, sei, t, steps=200)
        with torch.no_grad():
            xb = x.clone()
            f1 = model(xb, sei)[t]
            xb[t] = 0.0
            f0 = model(xb, sei)[t]
        delta = float((f1[1] - f1[0]) - (f0[1] - f0[0]))
        assert abs(phi.sum() - delta) < 1e-2 and err < 1e-2, (arch, phi.sum(), delta, err)
        assert np.array_equal(phi, phi2)
        # chunked passes give the same gradients as one pass
        from src.explainability import gradients as G
        st = torch.randn(7, 6)
        m1, g1 = G._margin_and_grad(model, x, sei, t, st)
        m2, g2 = G._margin_and_grad(model, x, sei, t, st, max_nodes_per_pass=x.shape[0])
        assert torch.allclose(m1, m2, atol=1e-5) and torch.allclose(g1, g2, atol=1e-5)
        refs = reference_pool(d)
        assert refs.shape == (int((d.train_mask & (d.y == 0)).sum()), 6)
        a = expected_gradients_subgraph(model, x, sei, t, refs, num_samples=20, seed=5)
        b = expected_gradients_subgraph(model, x, sei, t, refs, num_samples=20, seed=5)
        c = expected_gradients_subgraph(model, x, sei, t, refs, num_samples=20, seed=6)
        assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_common_nodes_persist_and_verify():
    from src.explainability.v4_explain import select_common_explain_nodes
    d = _toy_graph(200)
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "nodes.json"
        a = select_common_explain_nodes(d, 5, 1234, p)
        assert p.exists() and len(a["nodes"]) == 5
        assert all(bool(d.val_mask[i]) and int(d.y[i]) == 1 for i in a["nodes"])
        b = select_common_explain_nodes(d, 5, 1234, p)
        assert a["nodes"] == b["nodes"]
        try:
            select_common_explain_nodes(d, 5, 999, p)
        except ValueError:
            pass
        else:
            raise AssertionError("a different seed must not silently reuse the file")


def test_pg_train_sample_is_stratified():
    from src.explainability.v4_explain import sample_pg_train_nodes
    d = _toy_graph(400)
    s1 = sample_pg_train_nodes(d, 10, 5, seed=1)
    assert len(s1) == 10 and int((d.y[s1] == 1).sum()) >= 5
    assert all(bool(d.train_mask[i]) for i in s1)
    assert s1 != sample_pg_train_nodes(d, 10, 5, seed=2)
    assert s1 == sample_pg_train_nodes(d, 10, 5, seed=1)


def test_explainer_alias():
    from src.explainability.v4_explain import canonical_explainer
    assert canonical_explainer("GNNShap") == "ShapleyFeatures"
    assert canonical_explainer("PGExplainer") == "PGExplainer"


UNIT_TESTS = [v for k, v in dict(globals()).items() if k.startswith("test_")]


# ─────────────────────────────── (b) end to end ───────────────────────────────

def _train_tiny(data, seed, epochs=2):
    from src.training.trainer import build_model
    torch.manual_seed(seed)
    model = build_model("GCN", in_channels=data.num_node_features, hidden_channels=16,
                        num_layers=2, dropout=0.0)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    w = torch.tensor([1.0, 7.6])  # up-weight illicit so the tiny model predicts some
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(data.x, data.edge_index)
        loss = torch.nn.functional.cross_entropy(out[data.train_mask],
                                                 data.y[data.train_mask], weight=w)
        loss.backward()
        opt.step()
    return model.eval()


def run_e2e(keep: str | None = None) -> dict:
    import yaml
    from src.data.loader import load_elliptic
    from src.data.preprocessing import preprocess
    tmp = Path(keep) if keep else Path(tempfile.mkdtemp(prefix="v4_e2e_"))
    tmp.mkdir(parents=True, exist_ok=True)
    models, results, logs = tmp / "models", tmp / "results", tmp / "runs"
    models.mkdir(exist_ok=True)
    data = load_elliptic("./data" if (ROOT / "data").exists() else str(ROOT / "data"))
    preprocess(data)
    for seed, gate in ((42, True), (43, True), (44, False)):
        rid = "native_GCN_none" + ("" if seed == 42 else f"_s{seed}")
        model = _train_tiny(data, seed)
        torch.save(model.state_dict(), models / f"{rid}_best.pt")
        meta = {"run_id": rid, "scenario": "native", "architecture": "GCN",
                "balancing": "none", "seed": seed,
                "best_params": {"hidden_dim": 16, "num_layers": 2, "dropout": 0.0},
                "quality_passed": gate, "val_metrics": {"pr_auc": 0.3, "f1": 0.4, "mcc": 0.3},
                "test_metrics": {"pr_auc": 0.1, "f1": 0.2}, "calibrated_threshold": 0.5,
                "checkpoint": f"{rid}_best.pt"}
        (models / f"{rid}_meta.json").write_text(json.dumps(meta))
    cfg = yaml.safe_load((ROOT / "configs/experiment_v4.yaml").read_text(encoding="utf-8"))
    cfg["explainability"]["explain_nodes"] = 3
    cfg["stability"]["num_replicas"] = 2
    cfg["tracking"].update(results_dir=str(results), models_dir=str(models),
                           logs_dir=str(logs))
    cfg["data"]["root"] = str(ROOT / "data")
    cfg_path = tmp / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")

    cmd = [sys.executable, str(ROOT / "scripts/explain_matrix.py"), "--config", str(cfg_path),
           "--device", "cpu"]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    wall = time.time() - t0
    print(r.stdout[-3000:], r.stderr[-3000:])
    assert r.returncode == 0, "explain_matrix failed"

    import pandas as pd
    df = pd.read_csv(results / "elliptic_v4_stability.csv")
    ok = df[df.status == "ok"]
    assert len(ok) == 6, df[["run_id", "explainer", "status", "reason"]]
    assert set(df[df.status == "gated_out"].run_id) == {"native_GCN_none_s44"}
    assert set(ok.explainer) == {"GNNExplainer", "PGExplainer", "ShapleyFeatures"}
    assert "GNNShap" not in set(df.explainer)
    pg = ok[ok.explainer == "PGExplainer"]
    assert pg.spearman_full.isna().all() and (pg.spearman_full_reason == "no_aplica").all()
    sh = ok[ok.explainer == "ShapleyFeatures"]
    assert sh.spearman_edges.isna().all() and (sh.spearman_edges_reason == "no_aplica").all()
    assert sh.spearman_full.notna().all()
    assert (ok.n_nodes == 3).all()
    nodes = json.loads((results / "explain_nodes_v4.json").read_text())["nodes"]
    for _, row in ok.iterrows():
        z = np.load(results / row.rankings_file)
        assert list(z["nodes"]) == nodes  # same nodes for every model
        if row.explainer != "PGExplainer":
            assert z["feat"].shape == (3, 2, data.num_node_features)
        if row.explainer != "ShapleyFeatures":
            assert z["edge"].shape[0] == 2 and len(z["edge_ptr"]) == 4
    ev = [json.loads(line) for line in (logs / "progress.jsonl").read_text().splitlines()]
    assert any(e["event"] == "plan" and e["total"] == 6 for e in ev)
    ends = [e for e in ev if e["event"] == "end" and e["stage"] == "explain"]
    assert len(ends) == 6 and all("spearman_full" in e["metrics"] for e in ends)
    assert any(e["event"] == "skip" for e in ev)
    per = pd.read_csv(results / "elliptic_v4_stability_pernode.csv")
    assert len(per) == 18

    # --resume: nothing re-explained
    r2 = subprocess.run(cmd + ["--resume"], cwd=ROOT, capture_output=True, text=True)
    assert r2.returncode == 0 and "(6 already complete)" in r2.stdout, r2.stdout[-2000:]
    ev2 = [json.loads(line) for line in (logs / "progress.jsonl").read_text().splitlines()]
    assert ev2[len(ev)]["event"] == "plan" and ev2[len(ev)]["total"] == 0

    # cross-seed
    r3 = subprocess.run([sys.executable, str(ROOT / "scripts/v4/cross_seed_stability.py"),
                         "--results-dir", str(results)], cwd=ROOT, capture_output=True,
                        text=True)
    print(r3.stdout[-1500:], r3.stderr[-1500:])
    assert r3.returncode == 0
    cs = pd.read_csv(results / "elliptic_v4_cross_seed.csv")
    assert set(cs.explainer) == {"GNNExplainer", "PGExplainer", "ShapleyFeatures"}
    assert (cs.seed_a == 42).any() and (cs.seed_b == 43).any()

    timing = ok[["run_id", "explainer", "seconds", "sec_per_node"]].copy()
    timing["sec_per_node_replica"] = timing.sec_per_node / 2
    print("\nTIMINGS (CPU, 3 nodes × 2 replicas, v4 explainer parameters):")
    print(timing.to_string(index=False))
    print(f"explain_matrix wall time (incl. data load): {wall:.1f}s; outputs in {tmp}")
    return {"dir": str(tmp), "timing": timing}


def run_analysis_selftest():
    r = subprocess.run([sys.executable, str(ROOT / "scripts/v4/analyze_elliptic_v4.py"),
                        "--selftest"], cwd=ROOT, capture_output=True, text=True)
    print(r.stdout[-4000:], r.stderr[-2000:])
    assert r.returncode == 0, "analysis selftest failed"


def test_e2e_if_enabled():
    if os.environ.get("V4_E2E") == "1":
        run_e2e()


if __name__ == "__main__":
    fails = 0
    for fn in UNIT_TESTS:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            fails += 1
            import traceback
            traceback.print_exc()
            print(f"FAIL {fn.__name__}: {exc}")
    if "--e2e" in sys.argv or "--all" in sys.argv:
        keep = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--keep=")), None)
        run_e2e(keep)
        print("PASS e2e")
    if "--analysis" in sys.argv or "--all" in sys.argv:
        run_analysis_selftest()
        print("PASS analysis selftest")
    sys.exit(1 if fails else 0)
