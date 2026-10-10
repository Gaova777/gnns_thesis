"""Pipeline v4 checks: labels, split counts, scenarios and a CPU training smoke run.

Runs with pytest (if installed) or directly:
    uv run --frozen python tests/test_data_v4.py            # (a) + (b) + (c)
    uv run --frozen python tests/test_data_v4.py --no-train # (a) + (b) only

(c) writes to results_models_v4_smoke/ and results_v4_smoke/, and appends events to the
real monitoring log runs_v4/progress.jsonl (logs_dir of configs/experiment_v4.yaml).
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.imbalance import (  # noqa: E402
    V4_EXPECTED_TRAIN, V4_EXPECTED_TRAIN_BY_MODE, V4_MAIN_SCENARIOS, V4_STRESS_SCENARIOS,
    V4_SUBSAMPLE_SEED, create_v4_scenario,
)
from src.data.loader import (  # noqa: E402
    EXPECTED_COUNTS, EXPECTED_SPLIT_COUNTS, EXPECTED_SPLIT_COUNTS_BY_MODE, apply_label_mode,
    load_elliptic,
)
from src.data.preprocessing import preprocess  # noqa: E402

_DATA = None


def _data():
    global _DATA
    if _DATA is None:
        d = load_elliptic(root=str(ROOT / "data"))  # raises DataIntegrityError on mismatch
        preprocess(d, normalize=False)               # raises on split mismatch
        _DATA = d
    return _DATA


_MODE_DATA = {}


def _data_mode(mode):
    """Label-mode data, preprocessed with normalization (SMOTE interpolates real features)."""
    if mode not in _MODE_DATA:
        from copy import deepcopy
        d = load_elliptic(root=str(ROOT / "data"))
        apply_label_mode(d, mode)
        preprocess(d)
        _MODE_DATA[mode] = d
    return _MODE_DATA[mode]


def _counts(d, mask):
    return {"illicit": int((d.y[mask] == 1).sum()), "licit": int((d.y[mask] == 0).sum())}


def test_a_labels_and_splits():
    d = _data()
    got = {"illicit": int((d.y == 1).sum()), "licit": int((d.y == 0).sum()),
           "unknown": int((d.y == -1).sum())}
    assert got == EXPECTED_COUNTS == {"illicit": 4545, "licit": 42019, "unknown": 157205}
    assert set(d.y.unique().tolist()) == {-1, 0, 1}
    for name, exp in EXPECTED_SPLIT_COUNTS.items():
        mask = getattr(d, f"{name}_mask")
        assert _counts(d, mask) == exp, name
        assert int((d.y[mask] == -1).sum()) == 0, f"unknown in {name}_mask"
    assert EXPECTED_SPLIT_COUNTS["train"] == {"illicit": 3462, "licit": 26432}
    assert EXPECTED_SPLIT_COUNTS["val"] == {"illicit": 914, "licit": 9069}
    assert EXPECTED_SPLIT_COUNTS["test"] == {"illicit": 169, "licit": 6518}
    # unknown nodes remain in the graph (message passing)
    assert d.num_nodes == sum(EXPECTED_COUNTS.values())


def test_a2_old_v3_mapping_is_rejected():
    """The v3 remap (PyG 0 -> unknown, 2 -> licit) must trip the hard check."""
    from types import SimpleNamespace
    from src.data.loader import DataIntegrityError, validate_label_counts
    d = _data()
    pyg = torch.where(d.y == -1, torch.tensor(2), d.y)       # back to PyG encoding
    v3 = torch.full_like(pyg, -1)
    v3[pyg == 2] = 0
    v3[pyg == 1] = 1
    try:
        validate_label_counts(SimpleNamespace(y=v3))
    except DataIntegrityError:
        return
    raise AssertionError("v3 mapping was not rejected")


def test_a3_label_modes():
    """C = licit + unknown negatives, B = licit, A = unknown (v3); y3 keeps the truth."""
    exp_train = {"licit_unknown": (3462, 132803), "licit": (3462, 26432),
                 "unknown": (3462, 106371)}
    for mode, (e_il, e_neg) in exp_train.items():
        d = _data_mode(mode)
        assert d.label_mode == mode
        assert _counts(d, d.train_mask) == {"illicit": e_il, "licit": e_neg}, mode
        for name, exp in EXPECTED_SPLIT_COUNTS_BY_MODE[mode].items():
            assert _counts(d, getattr(d, f"{name}_mask")) == exp, (mode, name)
        # the 3-class truth is intact and illicit is always the positive class
        assert {"illicit": int((d.y3 == 1).sum()), "licit": int((d.y3 == 0).sum()),
                "unknown": int((d.y3 == -1).sum())} == EXPECTED_COUNTS
        assert torch.equal(d.y == 1, d.y3 == 1)
    assert EXPECTED_SPLIT_COUNTS_BY_MODE["licit_unknown"]["test"] == {"illicit": 169,
                                                                     "licit": 29515}
    # normalization does not depend on the label mode
    assert torch.equal(_data_mode("licit").x, _data_mode("licit_unknown").x)


def test_b_v4_scenarios():
    """Main scenarios (mode C): every real illicit node is kept in every scenario."""
    d = _data_mode("licit_unknown")
    assert V4_EXPECTED_TRAIN is V4_EXPECTED_TRAIN_BY_MODE["licit_unknown"]
    protected = {k: v for k, v in V4_EXPECTED_TRAIN.items() if k not in V4_STRESS_SCENARIOS}
    assert protected == {
        "native": (3462, 132803), "1:10": (3462, 34620), "1:1": (3462, 3462),
        "1:10_os": (6924, 69240), "1:20": (3462, 69240)}
    assert V4_MAIN_SCENARIOS == ["native", "1:10", "1:1", "1:10_os"]
    il_real = d.train_mask & (d.y == 1)
    for name, (e_il, e_neg) in protected.items():
        s, rep = create_v4_scenario(d, name, verbose=False)
        assert (rep["n_illicit"], rep["n_licit"]) == (e_il, e_neg), (name, rep)
        assert rep["expected_checked"] and rep["label_mode"] == "licit_unknown"
        n0 = d.num_nodes
        # no real illicit train node is ever dropped
        assert bool((s.train_mask[:n0] | ~il_real).all()), name
        # negatives keep the licit/unknown mix of the native train split (±1 node)
        o = rep["negative_origin"]
        assert abs(o["licit"] - round(e_neg * 26432 / 132803)) <= 1, (name, o)
        # real part of the graph and val/test untouched
        assert torch.equal(s.x[:n0], d.x) and torch.equal(s.y[:n0], d.y)
        assert torch.equal(s.val_mask[:n0], d.val_mask)
        assert torch.equal(s.test_mask[:n0], d.test_mask)
        assert bool((s.train_mask[:n0] & ~d.train_mask).sum() == 0)
        # fixed data seed -> same subset every call (i.e. for every model seed)
        s2, rep2 = create_v4_scenario(d, name, verbose=False)
        assert torch.equal(s.train_mask, s2.train_mask) and torch.equal(s.x, s2.x)
        assert rep["train_idx_sha1"] == rep2["train_idx_sha1"]
        assert rep["subsample_seed"] == V4_SUBSAMPLE_SEED == 2026
        if rep["n_synthetic_illicit"] == 0:
            assert torch.equal(s.edge_index, d.edge_index) and s.num_nodes == n0
    a, _ = create_v4_scenario(d, "1:1", verbose=False)
    b, _ = create_v4_scenario(d, "1:1", subsample_seed=7, verbose=False)
    assert not torch.equal(a.train_mask, b.train_mask)


def test_b2_smote():
    """SMOTE nodes: illicit, train only, interpolated (no duplicates), anchor's edges."""
    d = _data_mode("licit_unknown")
    s, rep = create_v4_scenario(d, "1:10_os", verbose=False)
    n0 = d.num_nodes
    syn = torch.arange(n0, s.num_nodes)
    assert len(syn) == rep["n_synthetic_illicit"] == 3462
    assert bool((s.y[syn] == 1).all() and (s.y3[syn] == 1).all())
    assert bool(s.train_mask[syn].all() and not s.val_mask[syn].any()
                and not s.test_mask[syn].any())
    assert bool(s.is_synthetic[syn].all()) and not bool(s.is_synthetic[:n0].any())
    # train timesteps only, and no edge crosses timesteps (so val/test never see them)
    assert int(s.timestep[syn].max()) <= 34
    e = s.edge_index
    assert bool((s.timestep[e[0]] == s.timestep[e[1]]).all())
    assert e[:, : d.edge_index.shape[1]].equal(d.edge_index)
    # interpolation: not an exact copy of any real illicit train node
    il = torch.where(d.train_mask & (d.y == 1))[0]
    dmin = torch.cdist(s.x[syn], d.x[il]).min(dim=1).values
    assert float((dmin == 0).float().mean()) < 0.01
    # features inside the convex range of the illicit train features
    lo, hi = d.x[il].min(0).values, d.x[il].max(0).values
    assert bool(((s.x[syn] >= lo - 1e-5) & (s.x[syn] <= hi + 1e-5)).all())


def test_b2b_stress_scenarios():
    """Stress scenarios (7-oct): every negative is kept, illicit nodes are dropped."""
    d = _data_mode("licit_unknown")
    assert V4_STRESS_SCENARIOS == ["1:100_subil", "1:200_subil"]
    exp = {"1:100_subil": (1328, 132803), "1:200_subil": (664, 132803)}
    neg_real = d.train_mask & (d.y == 0)
    for name in V4_STRESS_SCENARIOS:
        s, rep = create_v4_scenario(d, name, verbose=False)
        assert (rep["n_illicit"], rep["n_licit"]) == exp[name] == V4_EXPECTED_TRAIN[name]
        assert rep["expected_checked"] and rep["n_synthetic_illicit"] == 0
        assert s.num_nodes == d.num_nodes and torch.equal(s.edge_index, d.edge_index)
        # no negative is dropped, only illicit; val/test and features untouched
        assert bool((s.train_mask | ~neg_real).all()), name
        assert bool((s.train_mask & ~d.train_mask).sum() == 0)
        assert torch.equal(s.val_mask, d.val_mask) and torch.equal(s.test_mask, d.test_mask)
        assert torch.equal(s.x, d.x) and torch.equal(s.y, d.y)
        s2, rep2 = create_v4_scenario(d, name, verbose=False)
        assert rep["train_idx_sha1"] == rep2["train_idx_sha1"]


def test_b3_other_label_modes():
    for mode in ("licit", "unknown"):
        d = _data_mode(mode)
        for name, (e_il, e_neg) in V4_EXPECTED_TRAIN_BY_MODE[mode].items():
            _, rep = create_v4_scenario(d, name, verbose=False)
            assert (rep["n_illicit"], rep["n_licit"]) == (e_il, e_neg), (mode, name, rep)
            assert rep["expected_checked"]


def test_c_train_matrix_smoke():
    models = ROOT / "results_models_v4_smoke"
    log = ROOT / "runs_v4" / "progress.jsonl"
    n_before = len(log.read_text(encoding="utf-8").splitlines()) if log.exists() else 0
    meta_file = models / "native_GCN_none_meta.json"
    if meta_file.exists():
        meta_file.unlink()
    cmd = [sys.executable, "scripts/train_matrix.py", "--config", "configs/experiment_v4.yaml",
           "--device", "cpu", "--scenario", "native", "--arch", "GCN", "--balancing", "none",
           "--seed", "42", "--trials", "1", "--epochs", "2",
           "--models-dir", str(models), "--results-dir", str(ROOT / "results_v4_smoke"),
           "--no-mlflow"]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    print(f"  train_matrix smoke: {time.time() - t0:.1f}s, exit {r.returncode}")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    assert meta["scenario_mode"] == "v4" and meta["subsample_seed"] == 2026
    assert meta["label_mode"] == "licit_unknown"
    assert meta["scenario_report"]["n_illicit"] == 3462
    assert meta["scenario_report"]["n_licit"] == 132803
    ce = meta["cross_label_eval"]
    assert ce["test"]["vs_licit"]["n_negative"] == 6518
    assert ce["test"]["vs_unknown"]["n_negative"] == 22997
    assert ce["test"]["vs_all"]["n_negative"] == 29515
    assert ce["test"]["vs_all"]["n_illicit"] == 169
    assert meta["threshold_calibration"] == "val_only" and meta["hp_source"] == "optuna"
    assert meta["epochs_run"] == 2 and len(meta["config_sha256"]) == 64
    assert (models / meta["checkpoint"]).exists()
    events = [json.loads(line) for line in
              log.read_text(encoding="utf-8").splitlines()[n_before:]]
    kinds = [e["event"] for e in events if e["stage"] == "train"]
    assert kinds[:3] == ["plan", "start", "end"], kinds
    end = [e for e in events if e["event"] == "end"][-1]
    for k in ("val_pr_auc", "val_f1", "val_mcc", "test_pr_auc", "gate_passed",
              "n_train_illicit", "n_train_licit", "epochs_run"):
        assert k in end["metrics"], k
    print(f"  end event: {json.dumps(end['metrics'])}")


if __name__ == "__main__":
    tests = [test_a_labels_and_splits, test_a2_old_v3_mapping_is_rejected, test_a3_label_modes,
             test_b_v4_scenarios, test_b2_smote, test_b2b_stress_scenarios,
             test_b3_other_label_modes]
    if "--no-train" not in sys.argv:
        tests.append(test_c_train_matrix_smoke)
    for t in tests:
        t0 = time.time()
        t()
        print(f"PASS {t.__name__} ({time.time() - t0:.1f}s)")
    print("ALL PASS")
