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
# Pipeline v4 scenarios (revised 28-sep-2026 after the meeting with Cristian)
# ════════════════════════════════════════════════════════════════════════════
#
# Guiding rule: PROTECT THE FRAUD SIGNAL. Illicit nodes are the weakest, rarest signal, so
# the main scenarios never drop an illicit node. Balance is changed by
#   * undersampling the NEGATIVE class (licit, or licit + unknown in label mode C),
#     stratified by origin so the licit/unknown mix of the negatives is preserved, and
#   * oversampling the illicit class with SMOTE (Chawla et al., 2002): synthetic illicit
#     nodes interpolate the features of an illicit anchor and one of its k nearest illicit
#     neighbours, and inherit the anchor's edges (the edge-copying SMOTE baseline used by
#     GraphSMOTE, Zhao et al., 2021). No exact duplicates are created.
#
# Other design points kept from the 23-sep design:
#   * Only the train split changes; val/test are never resampled. Real nodes are never
#     removed from the graph (dropped ones still pass messages). SMOTE nodes are appended
#     to the graph, live in the train timesteps only and never touch val/test (Elliptic has
#     no edges across timesteps; checked by the tests).
#   * Sampling uses a FIXED data seed (V4_SUBSAMPLE_SEED), independent of the model seed, so
#     the 3 model seeds see the same training set and the seed sweep measures model
#     variance only.
#   * Every scenario returns a report and is checked against V4_EXPECTED_TRAIN (±1).
#
# Main scenarios (label mode C, train = 3,462 illicit / 132,803 licit+unknown):
#   native    everything, the REAL imbalance                        3,462 / 132,803 (1:38.4)
#   1:10      all illicit, negatives undersampled                   3,462 /  34,620
#   1:1       all illicit, negatives undersampled                   3,462 /   3,462
#   1:10_os   illicit ×2 with SMOTE (3,462 real + 3,462 synthetic),
#             negatives undersampled to 10 per illicit              6,924 /  69,240
# Legacy scenarios of the 23-sep design (drop illicit nodes; label mode B only; not run):
#   1:10_subil, 1:50_subil, 1:100_subil, native_size_ctrl

import hashlib
import json

V4_SUBSAMPLE_SEED = 2026
SMOTE_K = 5

V4_SCENARIOS = {
    "native": {"kind": "native"},
    "1:10": {"kind": "neg_per_illicit", "k": 10},
    "1:1": {"kind": "neg_per_illicit", "k": 1},
    "1:10_os": {"kind": "oversample", "factor": 2, "k": 10},
    # legacy (23-sep design, label mode B): subsample ILLICIT — kills the fraud signal
    "native_size_ctrl": {"kind": "native_size_ctrl"},
    "1:10_subil": {"kind": "illicit_per_licit", "k": 10},
    "1:50_subil": {"kind": "illicit_per_licit", "k": 50},
    "1:100_subil": {"kind": "illicit_per_licit", "k": 100},
}
V4_MAIN_SCENARIOS = ["native", "1:10", "1:1", "1:10_os"]

# Expected (n_illicit, n_negative) in the train mask, per label mode, default temporal split.
V4_BASE_TRAIN_BY_MODE = {
    "licit_unknown": (3462, 132803),
    "licit": (3462, 26432),
    "unknown": (3462, 106371),
}
V4_EXPECTED_TRAIN_BY_MODE = {
    "licit_unknown": {
        "native": (3462, 132803),
        "1:10": (3462, 34620),
        "1:1": (3462, 3462),
        "1:10_os": (6924, 69240),
    },
    "licit": {
        "native": (3462, 26432),
        "1:1": (3462, 3462),
        "1:10_os": (6924, 26432),  # capped: only 26,432 negatives exist (ratio 1:3.8)
        "native_size_ctrl": (802, 6122),
        "1:10_subil": (2643, 26432),
        "1:50_subil": (528, 26432),
        "1:100_subil": (264, 26432),
    },
    "unknown": {
        "native": (3462, 106371),
        "1:10": (3462, 34620),
        "1:1": (3462, 3462),
    },
}
V4_TOLERANCE = 1

# Backwards-compatible aliases (label mode C is the default).
V4_EXPECTED_TRAIN = V4_EXPECTED_TRAIN_BY_MODE["licit_unknown"]
V4_BASE_TRAIN = V4_BASE_TRAIN_BY_MODE["licit_unknown"]


def v4_target_counts(name: str, n_illicit: int, n_licit: int) -> tuple[int, int]:
    """Target (n_illicit, n_negative) of scenario ``name`` given the native train counts.

    For "oversample" the illicit target includes the synthetic nodes.
    """
    if name not in V4_SCENARIOS:
        raise ValueError(f"Unknown v4 scenario {name!r}. Choose from {list(V4_SCENARIOS)}")
    spec = V4_SCENARIOS[name]
    kind = spec["kind"]
    if kind == "native":
        return n_illicit, n_licit
    if kind == "neg_per_illicit":
        return n_illicit, min(n_licit, spec["k"] * n_illicit)
    if kind == "oversample":
        il = spec["factor"] * n_illicit
        return il, min(n_licit, spec["k"] * il)
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


def _sample_negatives(data: Data, neg_idx: torch.Tensor, size: int,
                      rng: np.random.Generator) -> torch.Tensor:
    """Undersample negatives stratified by origin (real licit vs unknown, from ``data.y3``).

    Keeps the licit/unknown proportion of the negatives (largest-remainder rounding), so the
    undersampled scenarios of label mode C stay comparable with its native scenario.
    """
    if size >= len(neg_idx):
        return neg_idx
    from src.data.loader import UNKNOWN
    if not hasattr(data, "y3"):
        return _sample(neg_idx, size, rng)
    origin_unknown = data.y3[neg_idx] == UNKNOWN
    groups = [neg_idx[~origin_unknown], neg_idx[origin_unknown]]
    groups = [g for g in groups if len(g)]
    if len(groups) == 1:
        return _sample(neg_idx, size, rng)
    n = len(neg_idx)
    exact = [size * len(g) / n for g in groups]
    alloc = [int(np.floor(e)) for e in exact]
    for i in np.argsort([-(e - a) for e, a in zip(exact, alloc)])[: size - sum(alloc)]:
        alloc[i] += 1
    parts = [_sample(g, a, rng) for g, a in zip(groups, alloc)]
    return torch.sort(torch.cat(parts)).values


def smote_oversample(data: Data, anchors: torch.Tensor, n_new: int, rng: np.random.Generator,
                     k: int = SMOTE_K) -> tuple[Data, torch.Tensor]:
    """Append ``n_new`` SMOTE illicit nodes built from ``anchors`` (train illicit nodes).

    Anchors are used round-robin in a random order (factor 2 → every illicit node anchors
    exactly one synthetic node). Each synthetic node gets
      x = x_a + λ (x_b − x_a),  b ∈ k nearest illicit anchors of a (Euclidean, normalized
      features), λ ~ U(0, 1);
      the anchor's incoming and outgoing edges, re-pointed to the new node;
      y = y3 = illicit, the anchor's timestep, train_mask = True, val/test = False.
    Every other node-level tensor attribute is extended with the anchor's value.
    Returns ``(data, new_node_indices)``; ``data`` is modified in place.
    """
    from src.data.loader import ILLICIT

    n_old = data.num_nodes
    a_idx = anchors.numpy()
    xa = data.x[anchors]
    dist = torch.cdist(xa, xa)
    dist.fill_diagonal_(float("inf"))
    kk = min(k, len(a_idx) - 1)
    nn_pos = torch.topk(dist, kk, largest=False).indices.numpy()  # positions within anchors

    order = rng.permutation(len(a_idx))
    src_pos = np.resize(order, n_new)
    nb_pos = nn_pos[src_pos, rng.integers(0, kk, size=n_new)]
    lam = torch.as_tensor(rng.random(n_new), dtype=data.x.dtype).unsqueeze(1)
    x_new = xa[src_pos] + lam * (xa[nb_pos] - xa[src_pos])
    src_nodes = torch.as_tensor(a_idx[src_pos], dtype=torch.long)
    new_idx = torch.arange(n_old, n_old + n_new, dtype=torch.long)

    # Edges: copy every edge incident to the anchor, re-pointed to the synthetic node.
    ei = data.edge_index
    anchor_to_new = torch.full((n_old,), -1, dtype=torch.long)
    # an anchor may seed several synthetic nodes (factor > 2): handle each copy round
    new_edges = []
    for start in range(0, n_new, len(a_idx)):
        sl = slice(start, min(start + len(a_idx), n_new))
        anchor_to_new.fill_(-1)
        anchor_to_new[src_nodes[sl]] = new_idx[sl]
        out_m = anchor_to_new[ei[0]] >= 0
        in_m = anchor_to_new[ei[1]] >= 0
        new_edges.append(torch.stack([anchor_to_new[ei[0, out_m]], ei[1, out_m]]))
        new_edges.append(torch.stack([ei[0, in_m], anchor_to_new[ei[1, in_m]]]))
    n_edges_added = int(sum(e.shape[1] for e in new_edges))

    for key in list(data.keys()):
        val = data[key]
        if key in ("x", "edge_index") or not torch.is_tensor(val) or val.dim() == 0:
            continue
        if val.size(0) != n_old:
            continue
        if key in ("y", "y3"):
            ext = torch.full((n_new,) + tuple(val.shape[1:]), ILLICIT, dtype=val.dtype)
        elif key == "train_mask":
            ext = torch.ones(n_new, dtype=val.dtype)
        elif key.endswith("_mask"):
            ext = torch.zeros(n_new, dtype=val.dtype)
        else:
            ext = val[src_nodes]
        data[key] = torch.cat([val, ext])
    data.x = torch.cat([data.x, x_new])
    data.edge_index = torch.cat([ei] + new_edges, dim=1)
    data.num_nodes = n_old + n_new
    data.is_synthetic = torch.cat([getattr(data, "is_synthetic", torch.zeros(n_old, dtype=torch.bool)),
                                   torch.ones(n_new, dtype=torch.bool)])
    data.smote_info = {"n_new": n_new, "k": kk, "n_edges_added": n_edges_added}
    return data, new_idx


def create_v4_scenario(
    data: Data,
    name: str,
    subsample_seed: int = V4_SUBSAMPLE_SEED,
    mask_name: str = "train_mask",
    check_expected: bool = True,
    verbose: bool = True,
) -> tuple[Data, dict]:
    """Build a v4 imbalance scenario.

    Only ``mask_name`` changes, except for the oversampling scenarios, which also append
    SMOTE nodes (see ``smote_oversample``). Labels follow ``data.label_mode`` (set by
    ``src.data.loader.apply_label_mode``; "licit" if absent): "licit" in the report means
    the negative class of that mode.

    Returns ``(data_new, report)``. ``report`` has n_illicit (real + synthetic), n_synthetic,
    n_licit (negatives) and their origin, total, ratio (illicit/negative), ratio_str
    ("1:x.x"), expected counts, subsample_seed and a SHA-1 of the selected real train
    indices (identical across model seeds by construction).

    Raises ``DataIntegrityError`` if the result deviates from the expected table of the
    label mode by more than ±1 while the native train counts are the default-split ones.
    """
    from src.data.loader import DataIntegrityError, ILLICIT, LICIT, UNKNOWN

    label_mode = getattr(data, "label_mode", "licit")
    data_new = deepcopy(data)
    mask = getattr(data_new, mask_name)
    idx = torch.where(mask)[0]
    il_idx = idx[data_new.y[idx] == ILLICIT]
    lic_idx = idx[data_new.y[idx] == LICIT]
    n_il0, n_lic0 = len(il_idx), len(lic_idx)

    spec = V4_SCENARIOS[name] if name in V4_SCENARIOS else None
    t_il, t_lic = v4_target_counts(name, n_il0, n_lic0)
    # Independent streams per class so each class's draw does not depend on the other.
    rng_il = np.random.default_rng([subsample_seed, 1])
    rng_lic = np.random.default_rng([subsample_seed, 0])
    rng_os = np.random.default_rng([subsample_seed, 2])

    n_syn = 0
    if spec["kind"] == "oversample":
        keep_il = il_idx
        n_syn = t_il - n_il0
    else:
        keep_il = _sample(il_idx, t_il, rng_il)
    keep_lic = _sample_negatives(data_new, lic_idx, t_lic, rng_lic)

    new_mask = torch.zeros_like(mask)
    new_mask[keep_il] = True
    new_mask[keep_lic] = True
    setattr(data_new, mask_name, new_mask)
    sel = torch.where(new_mask)[0].numpy().astype(np.int64)  # real nodes only
    if n_syn > 0:
        smote_oversample(data_new, il_idx, n_syn, rng_os)
        new_mask = getattr(data_new, mask_name)

    n_il = int((data_new.y[new_mask] == ILLICIT).sum())
    n_lic = int((data_new.y[new_mask] == LICIT).sum())
    origin = None
    if hasattr(data_new, "y3"):
        neg = new_mask & (data_new.y == LICIT)
        origin = {"licit": int((data_new.y3[neg] == LICIT).sum()),
                  "unknown": int((data_new.y3[neg] == UNKNOWN).sum())}
    report = {
        "scenario": name,
        "kind": spec["kind"],
        "label_mode": label_mode,
        "subsample_seed": subsample_seed,
        "n_illicit": n_il,
        "n_synthetic_illicit": n_syn,
        "n_licit": n_lic,
        "negative_origin": origin,
        "total": n_il + n_lic,
        "ratio": (n_il / n_lic) if n_lic else float("inf"),
        "ratio_str": f"1:{n_lic / n_il:.1f}" if n_il else "no illicit",
        "native_train": {"n_illicit": n_il0, "n_licit": n_lic0},
        "target": {"n_illicit": t_il, "n_licit": t_lic},
        "expected": None,
        "expected_checked": False,
        "train_idx_sha1": hashlib.sha1(sel.tobytes()).hexdigest(),
        "val_test_subsampled": False,
        "smote": getattr(data_new, "smote_info", None) if n_syn else None,
    }

    if check_expected:
        table = V4_EXPECTED_TRAIN_BY_MODE.get(label_mode, {})
        if (n_il0, n_lic0) == V4_BASE_TRAIN_BY_MODE.get(label_mode) and name in table:
            e_il, e_lic = table[name]
            report["expected"] = {"n_illicit": e_il, "n_licit": e_lic}
            report["expected_checked"] = True
            if abs(n_il - e_il) > V4_TOLERANCE or abs(n_lic - e_lic) > V4_TOLERANCE:
                raise DataIntegrityError(
                    f"Scenario {name} ({label_mode}): got {n_il} illicit / {n_lic} negative, "
                    f"expected {e_il} / {e_lic} (±{V4_TOLERANCE})"
                )
        else:
            print(f"  WARNING: scenario {name!r} in label mode {label_mode!r} with native train "
                  f"counts {(n_il0, n_lic0)} has no expected table; not checked.")

    if verbose:
        extra = f" (+{n_syn:,} SMOTE)" if n_syn else ""
        print(f"  [v4 scenario {name} | {label_mode}] illicit={n_il:,}{extra} "
              f"negatives={n_lic:,} {origin or ''} ratio={report['ratio_str']} "
              f"(subsample_seed={subsample_seed}, idx sha1={report['train_idx_sha1'][:10]})")
    return data_new, report


def build_v4_report_table(data: Data, names=None,
                          subsample_seed: int = V4_SUBSAMPLE_SEED) -> dict:
    """Build every v4 scenario and return {name: report} (used by tests and audits)."""
    names = list(names or V4_MAIN_SCENARIOS)
    out = {}
    for n in names:
        _, rep = create_v4_scenario(data, n, subsample_seed=subsample_seed, verbose=False)
        out[n] = rep
    return out


if __name__ == "__main__":  # pragma: no cover - manual audit helper
    import sys
    from src.data.loader import apply_label_mode, load_elliptic
    from src.data.preprocessing import preprocess

    d = load_elliptic()
    apply_label_mode(d, sys.argv[1] if len(sys.argv) > 1 else None)
    preprocess(d, normalize=True)
    print(json.dumps(build_v4_report_table(d), indent=2))
