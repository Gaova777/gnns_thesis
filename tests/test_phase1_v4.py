"""Tests del eje sintético v4 y del monitor (CPU, < 1 min).

uv run --frozen python -m pytest tests/test_phase1_v4.py -q
"""
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "phase1")); sys.path.insert(0, str(ROOT / "scripts" / "v4"))

import synthetic_aml_generator as G  # noqa: E402
import v4_common as C  # noqa: E402


@pytest.fixture(scope="module")
def g_v4():
    return G.generate_aml_graph(seed=42, **G.V4_DEFAULTS)


@pytest.fixture(scope="module")
def g_legacy():
    return G.generate_aml_graph(seed=42)


def test_legacy_keeps_shortcuts(g_legacy):
    rep = G.shortcut_report(g_legacy)
    assert rep["out_deg"] > 0.9            # el atajo que motivó v4 sigue reproducible
    assert g_legacy.generator_params["shortcut_free"] is False
    assert not g_legacy.hard_negative.any()


def test_v4_single_variable_auc_below_065(g_v4):
    rep = G.shortcut_report(g_v4)
    assert max(rep.values()) <= 0.65, rep


def test_v4_hard_negatives_and_instances(g_v4):
    assert g_v4.hard_negative.sum() > 0
    assert (g_v4.y[g_v4.hard_negative] == 0).all()
    # toda arista de tipología tiene instancia, y ambos extremos son de esa instancia
    typ = g_v4.typology_edge > 0
    assert (g_v4.pattern_instance_edge[typ] >= 0).all()
    assert (g_v4.pattern_instance_edge[~typ] == -1).all()
    ei = g_v4.edge_index[:, typ]
    inst = g_v4.pattern_instance_node
    assert torch.equal(inst[ei[0]], inst[ei[1]])


def test_scenarios_are_distinct_and_1to10_no_longer_collapses(g_v4):
    d = C.scale(g_v4.clone())
    nat = C.native_ratio(d)
    # el submuestreo legado (solo lícitos para r ≥ 0,1) caía al nativo; el de v4 recorta ilícitos
    r10 = C.make_scenario(d, "1:10").train_mask
    got = float((d.y[r10] == 1).sum()) / float((d.y[r10] == 0).sum())
    assert abs(got - 0.1) < 0.005 and abs(got - nat) > 0.03, (got, nat)
    assert C.collapsing_scenarios(d, ["1:10"], tol=2.0) == ["1:10"]   # ...pero está a < 2× del nativo
    ratios = []
    for s in ["1:1", "natural", "1:50", "1:100"]:
        m = C.make_scenario(d, s).train_mask
        ratios.append(float((d.y[m] == 1).sum()) / float((d.y[m] == 0).sum()))
    assert len({round(r, 3) for r in ratios}) == 4, ratios
    assert abs(ratios[0] - 1.0) < 0.01 and abs(ratios[2] - 0.02) < 0.002 and abs(ratios[3] - 0.01) < 0.002


def test_receptive_hops():
    assert C.receptive_hops("TAGCN") == C.NUM_LAYERS * C.TAGCN_K == 6
    assert C.receptive_hops("GCN") == C.receptive_hops("GAT") == C.receptive_hops("GraphSAGE") == 2


def test_pg_sample_is_stratified(g_v4):
    import run_phase1_v4 as R
    d = C.scale(g_v4.clone())
    for scen in ["natural", "1:100"]:
        ds = C.make_scenario(d, scen)
        nodes, frac = R.stratified_train_nodes(ds, 60, seed=0)
        assert frac >= 0.5 and len(nodes) > 0
        assert all(bool(ds.train_mask[n]) for n in nodes)
        assert float(ds.y[torch.tensor(nodes)].float().mean()) >= 0.5


def test_drop_random_keeps_symmetry(g_v4):
    import alignment_check as A
    ei = g_v4.edge_index
    out = A.drop_random(ei, 100, np.random.default_rng(0))
    assert out.size(1) == ei.size(1) - 200
    assert A.undirected_pairs(out) is not None


# ------------------------------- monitor -------------------------------------------
def _write(tmp: Path, events, hb_age_min=1.0, stage="train"):
    tmp.mkdir(parents=True, exist_ok=True)
    with open(tmp / "progress.jsonl", "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    ts = (datetime.now(timezone.utc) - timedelta(minutes=hb_age_min)).isoformat(timespec="seconds")
    (tmp / f"HEARTBEAT_{stage}").write_text(json.dumps({"ts": ts, "run_id": "x", "note": ""}))


def _ev(event, rid="", stage="train", **kw):
    return {"ts": datetime.now(timezone.utc).isoformat(), "stage": stage, "run_id": rid, "event": event, **kw}


def _run(tmp):
    import monitor_run as M
    a = M.parse(["--once", "--log-dir", str(tmp)])
    return M.build(a)


def test_monitor_clean(tmp_path):
    ev = [_ev("plan", total=3)]
    for i in range(3):
        ev += [_ev("start", f"r{i}"), _ev("end", f"r{i}", duration_s=10, metrics={"val_pr_auc": 0.5, "gate_passed": True})]
    _write(tmp_path, ev)
    r = _run(tmp_path)
    assert r["exit_code"] == 0 and r["stages"]["train"]["finished"]


def test_monitor_critical_signals(tmp_path):
    ev = [_ev("plan", total=20)]
    for i in range(10):
        ev += [_ev("start", f"r{i}"), _ev("end", f"r{i}", duration_s=5,
                                           metrics={"val_pr_auc": 0.1, "gate_passed": False})]
    ev += [_ev("start", "o1"), _ev("error", "o1", msg="OutOfMemoryError: CUDA out of memory"),
           _ev("start", "o2"), _ev("error", "o2", msg="OOM: CUDA out of memory"), _ev("start", "r99")]
    _write(tmp_path, ev, hb_age_min=60)
    r = _run(tmp_path)
    msgs = " | ".join(a["msg"] for a in r["alerts"] if a["level"] == "critical")
    assert r["exit_code"] == 2
    for needle in ("OOM seguidos", "compuerta", "val_pr_auc", "latido"):
        assert needle in msgs, msgs


def test_monitor_nan_and_no_aplica(tmp_path):
    ev = [_ev("plan", total=4, stage="explain")]
    for i in range(4):
        ev += [_ev("end", f"m{i}__GNNExplainer", stage="explain", duration_s=3,
                   metrics={"spearman_full": float("nan"), "spearman_edges": 0.5, "n_measurable": 0}),
               _ev("end", f"m{i}__PGExplainer", stage="explain", duration_s=3,
                   metrics={"spearman_full": float("nan"), "spearman_edges": 0.6, "n_measurable": 5})]
    _write(tmp_path, ev, stage="explain")
    r = _run(tmp_path)
    crit = [a["msg"] for a in r["alerts"] if a["level"] == "critical"]
    assert any("GNNExplainer" in m and "no_aplica" in m for m in crit), crit
    assert not any("PGExplainer" in m for m in crit), crit   # su NaN de spearman_full es esperada
