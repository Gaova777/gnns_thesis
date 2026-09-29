"""
Data loader for the Elliptic Bitcoin Transaction dataset.

Uses PyTorch Geometric's built-in EllipticBitcoinDataset for automatic
download, caching, and conversion to PyG Data objects.

LABEL ENCODING (fixed 2026-09-28, pipeline v4)
----------------------------------------------
PyG's ``EllipticBitcoinDataset.process`` maps the raw ``class`` column with
``{'unknown': 2, '1': 1, '2': 0}``, i.e.

    PyG y == 0  -> licit    (raw class '2', 42,019 nodes)
    PyG y == 1  -> illicit  (raw class '1',  4,545 nodes)
    PyG y == 2  -> unknown  (raw 'unknown', 157,205 nodes)

Up to v3 this loader assumed the opposite (0=unknown, 2=licit). The effect was
that the 157,205 UNLABELLED nodes were trained/evaluated as "licit" and the
42,019 real licit nodes were silently dropped. v4 remaps correctly to
0=licit, 1=illicit, -1=unknown and hard-fails if the counts do not match the
official dataset (``EXPECTED_COUNTS``). Unknown nodes stay in the graph (they
take part in message passing) but every mask built downstream requires
``y >= 0``, so they never enter the loss, the metrics, the quality gate or the
scenario subsampling.
"""

from pathlib import Path

import pandas as pd
import torch
from torch_geometric.datasets import EllipticBitcoinDataset

# Label values used everywhere downstream.
LICIT, ILLICIT, UNKNOWN = 0, 1, -1

# PyG raw label -> project label.
PYG_LABEL_MAP = {0: LICIT, 1: ILLICIT, 2: UNKNOWN}

# Official Elliptic totals (Weber et al., 2019). The run STOPS if these differ.
EXPECTED_COUNTS = {"illicit": 4545, "licit": 42019, "unknown": 157205}

# Default causal split (inclusive timestep ranges) and its labelled counts.
DEFAULT_SPLIT = {"train": (1, 34), "val": (35, 42), "test": (43, 49)}
EXPECTED_SPLIT_COUNTS = {
    "train": {"illicit": 3462, "licit": 26432},
    "val": {"illicit": 914, "licit": 9069},
    "test": {"illicit": 169, "licit": 6518},
}


# ── Label modes (reunión con Cristian, 28-sep-2026) ─────────────────────────
# The three ways of defining the NEGATIVE class. Illicit is always the positive class.
#   licit_unknown  (C, "completo", main analysis): licit + unknown are negatives. Closest to
#                  the real world (undetected fraud cannot be labelled) and keeps the REAL
#                  imbalance (train ≈ 1:38, test ≈ 1:175).
#   licit          (B): only the reviewed licit nodes are negatives; unknowns are excluded
#                  from loss/metrics (they still pass messages).
#   unknown        (A): only unknown nodes are negatives and real licit ones are excluded.
#                  Reproduces exactly what v3 trained on by mistake; kept to compare.
# A and B are only trained (native scenario) to argue that unknowns can be treated as licit;
# the explainers and the whole stability analysis run on C.
LABEL_MODES = ("licit_unknown", "licit", "unknown")
DEFAULT_LABEL_MODE = "licit_unknown"
LABEL_MODE_ALIASES = {"C": "licit_unknown", "B": "licit", "A": "unknown"}

# Labelled (positive / negative) counts per split and label mode, default temporal split.
_UNK_SPLIT = {"train": 106371, "val": 27837, "test": 22997}
EXPECTED_SPLIT_COUNTS_BY_MODE = {
    "licit": EXPECTED_SPLIT_COUNTS,
    "unknown": {k: {"illicit": v["illicit"], "licit": _UNK_SPLIT[k]}
                for k, v in EXPECTED_SPLIT_COUNTS.items()},
    "licit_unknown": {k: {"illicit": v["illicit"], "licit": v["licit"] + _UNK_SPLIT[k]}
                      for k, v in EXPECTED_SPLIT_COUNTS.items()},
}


def resolve_label_mode(mode: str | None) -> str:
    mode = LABEL_MODE_ALIASES.get(mode, mode) if mode else DEFAULT_LABEL_MODE
    if mode not in LABEL_MODES:
        raise ValueError(f"Unknown label mode {mode!r}. Choose from {LABEL_MODES} "
                         f"(or aliases {LABEL_MODE_ALIASES})")
    return mode


def apply_label_mode(data, mode: str | None) -> str:
    """Set ``data.y`` for the given label mode, keeping the 3-class truth in ``data.y3``.

    Must run on the output of ``load_elliptic`` (y in {0 licit, 1 illicit, -1 unknown}) and
    BEFORE ``preprocess`` (the split masks take every node with y >= 0). Returns the
    resolved mode name and stores it in ``data.label_mode``.
    """
    mode = resolve_label_mode(mode)
    y3 = data.y3 if hasattr(data, "y3") else data.y.clone()
    y = y3.clone()
    if mode == "licit_unknown":
        y[y3 == UNKNOWN] = LICIT
    elif mode == "unknown":
        y[y3 == LICIT] = UNKNOWN
        y[y3 == UNKNOWN] = LICIT
    data.y3 = y3
    data.y = y
    data.label_mode = mode
    return mode


class DataIntegrityError(RuntimeError):
    """Raised when the loaded data does not match the official Elliptic counts.

    Deliberately NOT an ``assert``: asserts vanish under ``python -O`` and this check
    must never be skipped (it is what would have caught the v3 label bug).
    """


def _label_counts(y: torch.Tensor) -> dict:
    return {
        "illicit": int((y == ILLICIT).sum()),
        "licit": int((y == LICIT).sum()),
        "unknown": int((y == UNKNOWN).sum()),
    }


def validate_label_counts(data) -> dict:
    """Hard check of the global label totals against ``EXPECTED_COUNTS``."""
    counts = _label_counts(data.y)
    if counts != EXPECTED_COUNTS:
        raise DataIntegrityError(
            f"Elliptic label counts {counts} != expected {EXPECTED_COUNTS}. "
            "The label remapping is wrong or the raw data is not the official release."
        )
    return counts


def validate_split_counts(data, split_ranges: dict | None = None,
                          label_mode: str | None = None) -> dict:
    """Check train/val/test labelled counts.

    With the default split (train 1-34, val 35-42, test 43-49) the counts must match
    ``EXPECTED_SPLIT_COUNTS_BY_MODE[label_mode]`` exactly, otherwise
    ``DataIntegrityError``. With any other split only the global totals are validated and
    a warning is printed. ``label_mode`` defaults to ``data.label_mode`` (or "licit" if the
    data never went through ``apply_label_mode``). "licit" in the returned dict means the
    negative class of that mode.
    """
    split_ranges = split_ranges or DEFAULT_SPLIT
    label_mode = resolve_label_mode(label_mode or getattr(data, "label_mode", "licit"))
    expected = EXPECTED_SPLIT_COUNTS_BY_MODE[label_mode]
    got = {}
    for name in ("train", "val", "test"):
        mask = getattr(data, f"{name}_mask")
        got[name] = {"illicit": int((data.y[mask] == ILLICIT).sum()),
                     "licit": int((data.y[mask] == LICIT).sum())}
        if int((data.y[mask] == UNKNOWN).sum()) != 0:
            raise DataIntegrityError(f"{name}_mask contains unknown (-1) nodes")

    normalized = {k: tuple(v) for k, v in split_ranges.items()}
    if normalized == DEFAULT_SPLIT:
        if got != expected:
            raise DataIntegrityError(
                f"Split counts {got} != expected {expected} for label mode {label_mode!r} "
                "and the default temporal split (train 1-34, val 35-42, test 43-49)."
            )
    else:
        if hasattr(data, "y3"):
            if _label_counts(data.y3) != EXPECTED_COUNTS:
                raise DataIntegrityError(f"3-class totals {_label_counts(data.y3)} != "
                                         f"{EXPECTED_COUNTS}")
        else:
            validate_label_counts(data)
        print(f"  WARNING: non-default split {normalized}; only global label totals were "
              f"validated. Split counts: {got}")
    return got


def load_elliptic(root: str = "./data", validate: bool = True):
    """
    Load the Elliptic Bitcoin dataset via PyG.

    Returns a single Data object with:
        - data.x: Node features [N, 165] (PyG drops txId and time_step)
        - data.edge_index: Directed edge index [2, E]
        - data.y: Labels (0=licit, 1=illicit, -1=unknown)
        - data.timestep: [N] timestep 1..49, read from the raw CSV
        - data.train_mask/val_mask/test_mask are (re)set by preprocessing.preprocess

    Args:
        root: Directory to download/cache the dataset (raw CSVs in root/raw).
        validate: If True (default), hard-fail unless the label totals match
            ``EXPECTED_COUNTS`` and the raw classes CSV agrees node by node.
    """
    dataset = EllipticBitcoinDataset(root=root)
    data = dataset[0]

    # PyG: {'unknown': 2, '1': 1, '2': 0}  ->  0=licit, 1=illicit, 2=unknown.
    y_remapped = torch.full_like(data.y, UNKNOWN)
    y_remapped[data.y == 0] = LICIT      # PyG 0 = raw '2' = licit
    y_remapped[data.y == 1] = ILLICIT    # PyG 1 = raw '1' = illicit
    # PyG 2 = raw 'unknown' stays UNKNOWN (-1)
    data.y = y_remapped

    # PyG's feature frame and ours share the raw CSV row order (PyG maps txId -> row index
    # in feat_df order), so the timestep column can be read positionally.
    raw_dir = Path(root) / "raw"
    feat_path = raw_dir / "elliptic_txs_features.csv"
    if feat_path.exists():
        df = pd.read_csv(feat_path, header=None, usecols=[1])
        data.timestep = torch.tensor(df[1].values, dtype=torch.long)
    else:
        if validate:
            raise DataIntegrityError(f"Raw features CSV not found at {feat_path}")
        print(f"  WARNING: Could not find raw CSV at {feat_path}")
        data.timestep = torch.ones(data.num_nodes, dtype=torch.long)

    if validate:
        validate_label_counts(data)
        # Node-by-node cross-check against the raw classes CSV: guarantees both the
        # mapping and the row alignment (y and timestep) are right.
        cls_path = raw_dir / "elliptic_txs_classes.csv"
        if cls_path.exists():
            cls = pd.read_csv(cls_path)["class"].astype(str)
            raw_y = torch.tensor(
                cls.map({"1": ILLICIT, "2": LICIT, "unknown": UNKNOWN}).values,
                dtype=data.y.dtype,
            )
            if raw_y.shape != data.y.shape or not torch.equal(raw_y, data.y):
                raise DataIntegrityError(
                    "Remapped labels disagree with elliptic_txs_classes.csv node by node."
                )

    return data


def get_labeled_mask(data) -> torch.Tensor:
    """Return a boolean mask for nodes with known labels (not unknown)."""
    return data.y >= 0


def print_dataset_stats(data) -> None:
    """Print basic statistics about the loaded dataset."""
    labeled = data.y >= 0
    illicit = data.y == ILLICIT
    licit = data.y == LICIT

    print("=" * 60)
    print("Elliptic Bitcoin Dataset Statistics")
    print("=" * 60)
    print(f"Total nodes:      {data.num_nodes:,}")
    print(f"Total edges:      {data.num_edges:,}")
    print(f"Node features:    {data.num_node_features}")
    print(f"Labeled nodes:    {labeled.sum().item():,}")
    print(f"  Licit (0):      {licit.sum().item():,}")
    print(f"  Illicit (1):    {illicit.sum().item():,}")
    print(f"  Unknown (-1):   {(data.y == UNKNOWN).sum().item():,}")
    print(f"Illicit ratio:    {illicit.sum().item() / labeled.sum().item():.4f}")
    print("=" * 60)
