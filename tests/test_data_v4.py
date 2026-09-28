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
    V4_EXPECTED_TRAIN, V4_SCENARIOS, V4_SUBSAMPLE_SEED, create_v4_scenario,
)
from src.data.loader import (  # noqa: E402
    EXPECTED_COUNTS, EXPECTED_SPLIT_COUNTS, load_elliptic,
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


def test_b_v4_scenarios():
    d = _data()
    table = {
        "native": (3462, 26432), "1:10": (2643, 26432), "1:50": (528, 26432),
        "1:100": (264, 26432), "1:1": (3462, 3462), "native_size_ctrl": (802, 6122),
    }
    assert V4_EXPECTED_TRAIN == table and set(V4_SCENARIOS) == set(table)
    for name, (e_il, e_lic) in table.items():
        s, rep = create_v4_scenario(d, name, verbose=False)
        assert (rep["n_illicit"], rep["n_licit"]) == (e_il, e_lic), (name, rep)
        assert _counts(s, s.train_mask) == {"illicit": e_il, "licit": e_lic}
        assert rep["total"] == e_il + e_lic and rep["expected_checked"]
        assert abs(rep["ratio"] - e_il / e_lic) < 1e-12
        # train mask is a subset of the original one; no unknown enters it
        assert bool((s.train_mask & ~d.train_mask).sum() == 0)
        assert int((s.y[s.train_mask] == -1).sum()) == 0
        # graph, features, labels, val/test untouched
        assert torch.equal(s.edge_index, d.edge_index) and torch.equal(s.y, d.y)
        assert torch.equal(s.val_mask, d.val_mask) and torch.equal(s.test_mask, d.test_mask)
        # fixed data seed -> same subset every call (i.e. for every model seed)
        s2, rep2 = create_v4_scenario(d, name, verbose=False)
        assert torch.equal(s.train_mask, s2.train_mask)
        assert rep["train_idx_sha1"] == rep2["train_idx_sha1"]
        assert rep["subsample_seed"] == V4_SUBSAMPLE_SEED == 2026
    # a different data seed gives a different subset (sampling actually happens)
    a, _ = create_v4_scenario(d, "1:1", verbose=False)
    b, _ = create_v4_scenario(d, "1:1", subsample_seed=7, verbose=False)
    assert not torch.equal(a.train_mask, b.train_mask)


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
    assert meta["scenario_report"]["n_illicit"] == 3462
    assert meta["scenario_report"]["n_licit"] == 26432
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
    tests = [test_a_labels_and_splits, test_a2_old_v3_mapping_is_rejected, test_b_v4_scenarios]
    if "--no-train" not in sys.argv:
        tests.append(test_c_train_matrix_smoke)
    for t in tests:
        t0 = time.time()
        t()
        print(f"PASS {t.__name__} ({time.time() - t0:.1f}s)")
    print("ALL PASS")
