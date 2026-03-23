"""
Data loader for the Elliptic Bitcoin Transaction dataset.

Uses PyTorch Geometric's built-in EllipticBitcoinDataset for automatic
download, caching, and conversion to PyG Data objects.
"""

import torch
import pandas as pd
from pathlib import Path
from torch_geometric.datasets import EllipticBitcoinDataset


def load_elliptic(root: str = "./data") -> torch.Tensor:
    """
    Load the Elliptic Bitcoin dataset via PyG.

    Returns a single Data object with:
        - data.x: Node features [N, 166]
        - data.edge_index: Directed edge index [2, E]
        - data.y: Labels (0=licit, 1=illicit, 2=unknown)
        - data.train_mask, data.val_mask, data.test_mask (set later)

    Args:
        root: Directory to download/cache the dataset.

    Returns:
        PyG Data object.
    """
    dataset = EllipticBitcoinDataset(root=root)
    data = dataset[0]

    # Map labels: PyG's EllipticBitcoinDataset uses:
    #   0 = unknown, 1 = illicit (~4,545), 2 = licit (~157,205)
    # We remap to: 0=licit, 1=illicit, -1=unknown
    y_remapped = torch.full_like(data.y, -1)
    y_remapped[data.y == 2] = 0  # licit -> 0
    y_remapped[data.y == 1] = 1  # illicit -> 1
    # unknown (0) stays as -1
    data.y = y_remapped

    # Extract timestep from raw CSV (PyG strips it from the feature matrix)
    # The raw CSV has: col0=txID, col1=timestep, col2-166=features
    raw_path = Path(root) / "raw" / "elliptic_txs_features.csv"
    if raw_path.exists():
        df = pd.read_csv(raw_path, header=None, usecols=[1])
        data.timestep = torch.tensor(df[1].values, dtype=torch.long)
    else:
        # Fallback: assign all nodes timestep 1 (will log a warning)
        print(f"  WARNING: Could not find raw CSV at {raw_path}")
        data.timestep = torch.ones(data.num_nodes, dtype=torch.long)

    return data


def get_labeled_mask(data) -> torch.Tensor:
    """Return a boolean mask for nodes with known labels (not unknown)."""
    return data.y >= 0


def print_dataset_stats(data) -> None:
    """Print basic statistics about the loaded dataset."""
    labeled = data.y >= 0
    illicit = data.y == 1
    licit = data.y == 0

    print("=" * 60)
    print("Elliptic Bitcoin Dataset Statistics")
    print("=" * 60)
    print(f"Total nodes:      {data.num_nodes:,}")
    print(f"Total edges:      {data.num_edges:,}")
    print(f"Node features:    {data.num_node_features}")
    print(f"Labeled nodes:    {labeled.sum().item():,}")
    print(f"  Licit (0):      {licit.sum().item():,}")
    print(f"  Illicit (1):    {illicit.sum().item():,}")
    print(f"  Unknown (-1):   {(data.y == -1).sum().item():,}")
    print(f"Illicit ratio:    {illicit.sum().item() / labeled.sum().item():.4f}")
    print("=" * 60)
