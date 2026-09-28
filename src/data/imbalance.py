"""
Imbalance scenario generator for the Elliptic dataset.

Creates controlled imbalance scenarios by undersampling licit nodes
while preserving ALL illicit nodes. Maintains edge consistency.
"""

import torch
import numpy as np
from torch_geometric.data import Data
from typing import Optional
from copy import deepcopy


def create_imbalance_scenario(
    data: Data,
    target_ratio: Optional[float],
    mask_name: str = "train_mask",
    seed: int = 42,
) -> Data:
    """
    Create an imbalanced version of the dataset by undersampling licit nodes.

    Preserves ALL illicit nodes and reduces licit nodes to achieve the
    target illicit:licit ratio. Only modifies the specified mask (train by
    default); val/test masks remain unchanged.

    Args:
        data: PyG Data object with masks and labels.
        target_ratio: Desired illicit:licit ratio.
                      1.0 = 1:1, 0.1 = 1:10, 0.02 = 1:50, 0.01 = 1:100
                      None = preserve native distribution (no resampling).
                            Allows direct comparison with literature baselines
                            (e.g. Weber 2019 uses Elliptic's native ~1:30 ratio).
        mask_name: Which mask to apply undersampling to ("train_mask").
        seed: Random seed for reproducibility.

    Returns:
        New Data object with updated mask (does NOT remove nodes from graph,
        only modifies the mask to exclude sampled-out licit nodes).
    """
    # Native mode: preserve distribution as-is (no resampling).
    # Useful for literature replication where datasets have known imbalance.
    if target_ratio is None:
        data_new = deepcopy(data)
        mask = getattr(data_new, mask_name)
        illicit = (data_new.y[mask] == 1).sum().item()
        licit = (data_new.y[mask] == 0).sum().item()
        ratio = illicit / licit if licit > 0 else float("inf")
        print(f"  [mode=native] preserving natural distribution")
        print(f"  Scenario illicit:licit = 1:{1/ratio:.1f} (native)")
        print(f"    Illicit: {illicit:,} | Licit: {licit:,} | Total: {mask.sum().item():,}")
        print(f"    Actual ratio: {ratio:.4f}")
        return data_new

    rng = np.random.RandomState(seed)
    data_new = deepcopy(data)
    mask = getattr(data_new, mask_name)

    # Get indices of illicit and licit nodes within the mask
    masked_indices = torch.where(mask)[0]
    illicit_in_mask = masked_indices[data_new.y[masked_indices] == 1]
    licit_in_mask = masked_indices[data_new.y[masked_indices] == 0]

    n_illicit = len(illicit_in_mask)
    n_licit = len(licit_in_mask)
    new_mask = torch.zeros_like(mask)

    if target_ratio >= 0.1:
        # Mode A: subsample LICIT → more balanced scenarios (e.g. 1:1, 1:10)
        n_licit_target = int(n_illicit / target_ratio)
        if n_licit_target >= n_licit:
            print(f"  [mode=natural] ratio {target_ratio} needs {n_licit_target:,} licit "
                  f"but only {n_licit:,} available — using natural ratio")
            new_mask = mask.clone()
        else:
            licit_keep = torch.tensor(
                rng.choice(licit_in_mask.numpy(), size=n_licit_target, replace=False),
                dtype=torch.long,
            )
            new_mask[illicit_in_mask] = True
            new_mask[licit_keep] = True
            print(f"  [mode=subsample-licit] kept {n_licit_target:,}/{n_licit:,} licit nodes")
    else:
        # Mode B: subsample ILLICIT → more extreme imbalance (e.g. 1:50, 1:100)
        # Natural ratio (~1:9) cannot be made more imbalanced by dropping licit nodes.
        n_illicit_target = max(1, int(n_licit * target_ratio))
        if n_illicit_target >= n_illicit:
            print(f"  [mode=natural] ratio {target_ratio} needs {n_illicit_target:,} illicit "
                  f"but only {n_illicit:,} available — using natural ratio")
            new_mask = mask.clone()
        else:
            illicit_keep = torch.tensor(
                rng.choice(illicit_in_mask.numpy(), size=n_illicit_target, replace=False),
                dtype=torch.long,
            )
            new_mask[licit_in_mask] = True   # keep ALL licit
            new_mask[illicit_keep] = True    # keep only sampled illicit
            print(f"  [mode=subsample-illicit] kept {n_illicit_target:,}/{n_illicit:,} illicit nodes")

    setattr(data_new, mask_name, new_mask)

    # Stats
    total = new_mask.sum().item()
    illicit_count = (data_new.y[new_mask] == 1).sum().item()
    licit_count = (data_new.y[new_mask] == 0).sum().item()
    actual_ratio = illicit_count / licit_count if licit_count > 0 else float("inf")

    print(f"  Scenario illicit:licit = 1:{1/target_ratio:.0f}")
    print(f"    Illicit: {illicit_count:,} | Licit: {licit_count:,} | Total: {total:,}")
    print(f"    Actual ratio: {actual_ratio:.4f} (target: {target_ratio:.4f})")

    return data_new


def create_all_scenarios(
    data: Data,
    ratios: dict = None,
    seed: int = 42,
) -> dict:
    """
    Create all imbalance scenarios from the config.

    Args:
        data: Preprocessed PyG Data object.
        ratios: Dict mapping scenario names to illicit:licit ratios.
                Default: {"1:1": 1.0, "1:10": 0.1, "1:50": 0.02, "1:100": 0.01}
        seed: Random seed.

    Returns:
        Dict mapping scenario names to Data objects.
    """
    if ratios is None:
        ratios = {
            "1:1": 1.0,
            "1:10": 0.1,
            "1:50": 0.02,
            "1:100": 0.01,
        }

    print("Creating imbalance scenarios:")
    scenarios = {}
    for name, ratio in ratios.items():
        print(f"\n  --- Scenario {name} ---")
        scenarios[name] = create_imbalance_scenario(data, ratio, seed=seed)

    return scenarios


def verify_scenario_integrity(data: Data, scenario_name: str = "") -> bool:
    """
    Verify that a scenario has no isolated illicit node communities.

    Checks that every illicit node in the train mask has at least one
    edge connecting it to another node in the mask.

    Args:
        data: PyG Data with masks.
        scenario_name: Name for logging.

    Returns:
        True if integrity check passes.
    """
    mask = data.train_mask
    masked_nodes = set(torch.where(mask)[0].numpy())
    illicit_nodes = set(torch.where(mask & (data.y == 1))[0].numpy())

    edge_index = data.edge_index.numpy()
    isolated = []

    for node in illicit_nodes:
        # Check if node has any edge to/from another masked node
        out_edges = edge_index[1, edge_index[0] == node]
        in_edges = edge_index[0, edge_index[1] == node]
        neighbors = set(out_edges) | set(in_edges)
        connected = neighbors & masked_nodes

        if len(connected) == 0:
            isolated.append(node)

    if isolated:
        print(f"  ⚠ Scenario {scenario_name}: {len(isolated)} isolated illicit nodes found")
        return False
    else:
        print(f"  ✓ Scenario {scenario_name}: all illicit nodes connected")
        return True


# ════════════════════════════════════════════════════════════════════════════
# Pipeline v4 scenarios
# ════════════════════════════════════════════════════════════════════════════
#
# Design (v4):
#   * The graph is NEVER modified: only the train mask changes. Nodes dropped from the
#     train mask (and all unknown nodes) still take part in message passing.
#   * Subsampling uses a FIXED data seed (V4_SUBSAMPLE_SEED), independent of the model
#     seed, so the 3 model seeds (42/43/44) see exactly the same training subset and the
#     seed sweep measures model variance only.
#   * val/test are never subsampled.
#   * Every scenario returns a report and is checked against V4_EXPECTED_TRAIN (±1).
#
# Scenario semantics (default split: train = 3,462 illicit / 26,432 licit):
#   native            all labelled train nodes (≈ 1:7.6)
#   1:10, 1:50, 1:100 keep ALL licit (26,432), subsample illicit to floor(26,432 / k)
#   1:1               all illicit (3,462) + 3,462 subsampled licit
#   native_size_ctrl  size control for 1:1: same total (6,924) at the native ratio
#                     (illicit = round(6,924 × 3,462 / 29,894) = 802, licit = 6,122)

import hashlib
import json

V4_SUBSAMPLE_SEED = 2026

V4_SCENARIOS = {
    "native": {"kind": "native"},
    "1:1": {"kind": "balanced"},
    "native_size_ctrl": {"kind": "native_size_ctrl"},
    "1:10": {"kind": "illicit_per_licit", "k": 10},
    "1:50": {"kind": "illicit_per_licit", "k": 50},
    "1:100": {"kind": "illicit_per_licit", "k": 100},
}

# Expected (n_illicit, n_licit) in the train mask for the default temporal split.
V4_EXPECTED_TRAIN = {
    "native": (3462, 26432),
    "1:1": (3462, 3462),
    "native_size_ctrl": (802, 6122),
    "1:10": (2643, 26432),
    "1:50": (528, 26432),
    "1:100": (264, 26432),
}
V4_BASE_TRAIN = (3462, 26432)  # native train counts the expected table is derived from
V4_TOLERANCE = 1


def v4_target_counts(name: str, n_illicit: int, n_licit: int) -> tuple[int, int]:
    """Target (n_illicit, n_licit) for scenario ``name`` given the native train counts."""
    if name not in V4_SCENARIOS:
        raise ValueError(f"Unknown v4 scenario {name!r}. Choose from {list(V4_SCENARIOS)}")
    spec = V4_SCENARIOS[name]
    kind = spec["kind"]
    if kind == "native":
        return n_illicit, n_licit
    if kind == "balanced":
        m = min(n_illicit, n_licit)
        return m, m
    if kind == "illicit_per_licit":
        # integer division == floor(n_licit × 1/k) without float error
        return min(n_licit // spec["k"], n_illicit), n_licit
    if kind == "native_size_ctrl":
        total = 2 * min(n_illicit, n_licit)  # size of the 1:1 scenario
        il = int(round(total * n_illicit / (n_illicit + n_licit)))
        return il, total - il
    raise ValueError(f"Unhandled scenario kind {kind!r}")


def _sample(idx: torch.Tensor, size: int, rng: np.random.Generator) -> torch.Tensor:
    if size >= len(idx):
        return idx
    chosen = rng.choice(idx.numpy(), size=size, replace=False)
    return torch.as_tensor(np.sort(chosen), dtype=torch.long)


def create_v4_scenario(
    data: Data,
    name: str,
    subsample_seed: int = V4_SUBSAMPLE_SEED,
    mask_name: str = "train_mask",
    check_expected: bool = True,
    verbose: bool = True,
) -> tuple[Data, dict]:
    """Build a v4 imbalance scenario. Only ``mask_name`` changes; the graph is untouched.

    Returns ``(data_new, report)``. ``report`` has n_illicit, n_licit, total, ratio
    (illicit/licit), ratio_str ("1:x.x"), expected counts, subsample_seed and a SHA-1 of
    the selected train indices (identical across model seeds by construction).

    Raises ``DataIntegrityError`` if the result deviates from ``V4_EXPECTED_TRAIN`` by
    more than ±1 while the native train counts are the default-split ones. With a
    non-default split the expected table does not apply: a warning is printed instead.
    """
    from src.data.loader import DataIntegrityError, ILLICIT, LICIT

    data_new = deepcopy(data)
    mask = getattr(data_new, mask_name)
    idx = torch.where(mask)[0]
    il_idx = idx[data_new.y[idx] == ILLICIT]
    lic_idx = idx[data_new.y[idx] == LICIT]
    n_il0, n_lic0 = len(il_idx), len(lic_idx)

    t_il, t_lic = v4_target_counts(name, n_il0, n_lic0)
    # Independent streams per class so each class's draw does not depend on the other.
    rng_il = np.random.default_rng([subsample_seed, 1])
    rng_lic = np.random.default_rng([subsample_seed, 0])
    keep_il = _sample(il_idx, t_il, rng_il)
    keep_lic = _sample(lic_idx, t_lic, rng_lic)

    new_mask = torch.zeros_like(mask)
    new_mask[keep_il] = True
    new_mask[keep_lic] = True
    setattr(data_new, mask_name, new_mask)

    n_il = int((data_new.y[new_mask] == ILLICIT).sum())
    n_lic = int((data_new.y[new_mask] == LICIT).sum())
    sel = torch.where(new_mask)[0].numpy().astype(np.int64)
    report = {
        "scenario": name,
        "kind": V4_SCENARIOS[name]["kind"],
        "subsample_seed": subsample_seed,
        "n_illicit": n_il,
        "n_licit": n_lic,
        "total": n_il + n_lic,
        "ratio": (n_il / n_lic) if n_lic else float("inf"),
        "ratio_str": f"1:{n_lic / n_il:.1f}" if n_il else "no illicit",
        "native_train": {"n_illicit": n_il0, "n_licit": n_lic0},
        "target": {"n_illicit": t_il, "n_licit": t_lic},
        "expected": None,
        "expected_checked": False,
        "train_idx_sha1": hashlib.sha1(sel.tobytes()).hexdigest(),
        "val_test_subsampled": False,
    }

    if check_expected:
        if (n_il0, n_lic0) == V4_BASE_TRAIN:
            e_il, e_lic = V4_EXPECTED_TRAIN[name]
            report["expected"] = {"n_illicit": e_il, "n_licit": e_lic}
            report["expected_checked"] = True
            if abs(n_il - e_il) > V4_TOLERANCE or abs(n_lic - e_lic) > V4_TOLERANCE:
                raise DataIntegrityError(
                    f"Scenario {name}: got {n_il} illicit / {n_lic} licit, expected "
                    f"{e_il} / {e_lic} (±{V4_TOLERANCE})"
                )
        else:
            print(f"  WARNING: native train counts {(n_il0, n_lic0)} differ from the default "
                  f"split {V4_BASE_TRAIN}; scenario {name!r} not checked against the table.")

    if verbose:
        print(f"  [v4 scenario {name}] illicit={n_il:,} licit={n_lic:,} total={n_il + n_lic:,} "
              f"ratio={report['ratio_str']} (subsample_seed={subsample_seed}, "
              f"idx sha1={report['train_idx_sha1'][:10]})")
    return data_new, report


def build_v4_report_table(data: Data, names=None,
                          subsample_seed: int = V4_SUBSAMPLE_SEED) -> dict:
    """Build every v4 scenario and return {name: report} (used by tests and audits)."""
    names = list(names or V4_SCENARIOS)
    out = {}
    for n in names:
        _, rep = create_v4_scenario(data, n, subsample_seed=subsample_seed, verbose=False)
        out[n] = rep
    return out


if __name__ == "__main__":  # pragma: no cover - manual audit helper
    from src.data.loader import load_elliptic
    from src.data.preprocessing import preprocess

    d = load_elliptic()
    preprocess(d, normalize=False)
    print(json.dumps(build_v4_report_table(d), indent=2))
