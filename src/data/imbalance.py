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
    target_ratio: float,
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
        mask_name: Which mask to apply undersampling to ("train_mask").
        seed: Random seed for reproducibility.

    Returns:
        New Data object with updated mask (does NOT remove nodes from graph,
        only modifies the mask to exclude sampled-out licit nodes).
    """
    rng = np.random.RandomState(seed)
    data_new = deepcopy(data)
    mask = getattr(data_new, mask_name)

    # Get indices of illicit and licit nodes within the mask
    masked_indices = torch.where(mask)[0]
    illicit_in_mask = masked_indices[data_new.y[masked_indices] == 1]
    licit_in_mask = masked_indices[data_new.y[masked_indices] == 0]

    n_illicit = len(illicit_in_mask)
    n_licit_target = int(n_illicit / target_ratio)

    if n_licit_target >= len(licit_in_mask):
        # Requested ratio is less imbalanced than current data — keep all
        print(f"  Warning: target ratio {target_ratio} requires {n_licit_target} licit "
              f"but only {len(licit_in_mask)} available. Keeping all.")
        return data_new

    # Randomly select licit nodes to keep
    licit_keep_idx = rng.choice(
        licit_in_mask.numpy(), size=n_licit_target, replace=False
    )
    licit_keep_idx = torch.tensor(licit_keep_idx, dtype=torch.long)

    # Build new mask: all illicit + sampled licit
    new_mask = torch.zeros_like(mask)
    new_mask[illicit_in_mask] = True
    new_mask[licit_keep_idx] = True
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
