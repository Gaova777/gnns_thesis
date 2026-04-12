"""
GraphSMOTE integration for imbalanced node classification on graphs.

Adapts the approach from Zhao, Zhang & Wang (2021, WSDM):
  1. GNN encoder generates node embeddings
  2. Synthetic minority nodes are generated via interpolation in latent space
  3. An edge generator creates connections for synthetic nodes
  4. The augmented graph is used for classification

This module provides a simplified, self-contained implementation
compatible with the thesis pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np


class GraphSMOTEEncoder(nn.Module):
    """GNN encoder to generate node embeddings for SMOTE interpolation."""

    def __init__(self, in_channels: int, hidden_channels: int = 128):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = self.conv2(h, edge_index)
        return h


class EdgeGenerator(nn.Module):
    """
    Predicts edges for synthetic nodes based on embedding similarity.

    Given two node embeddings, outputs a probability that an edge
    exists between them.
    """

    def __init__(self, hidden_channels: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(2 * hidden_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, 1)

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """Predict edge probability between node embeddings z_i and z_j."""
        z_cat = torch.cat([z_i, z_j], dim=-1)
        h = F.relu(self.fc1(z_cat))
        return torch.sigmoid(self.fc2(h)).squeeze(-1)


class GraphSMOTE:
    """
    GraphSMOTE for generating synthetic minority nodes on graphs.

    Args:
        in_channels: Input feature dimension.
        hidden_channels: Embedding dimension.
        up_scale: Target ratio of minority to majority (1.0 = balanced).
        device: Target device.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        up_scale: float = 1.0,
        device: str = "cpu",
    ):
        self.hidden_channels = hidden_channels
        self.up_scale = up_scale
        self.device = device

        self.encoder = GraphSMOTEEncoder(in_channels, hidden_channels).to(device)
        self.edge_gen = EdgeGenerator(hidden_channels).to(device)

    def generate_synthetic_nodes(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
        seed: int = 42,
    ) -> tuple:
        """
        Generate synthetic nodes for the minority class via SMOTE interpolation.

        Args:
            embeddings: Node embeddings [N, D] from the encoder.
            labels: Node labels [N].
            mask: Training mask [N].
            seed: Random seed.

        Returns:
            Tuple of (synthetic_embeddings, synthetic_labels).
        """
        rng = np.random.RandomState(seed)

        masked_labels = labels[mask]
        minority_idx = torch.where(mask & (labels == 1))[0]
        majority_count = (mask & (labels == 0)).sum().item()
        minority_count = minority_idx.shape[0]

        # Calculate how many synthetic nodes to generate
        n_synthetic = int(majority_count * self.up_scale) - minority_count
        if n_synthetic <= 0:
            print(f"  WARNING: GraphSMOTE n_synthetic={n_synthetic} <= 0 "
                  f"(majority={majority_count}, minority={minority_count}, "
                  f"up_scale={self.up_scale}) — skipping augmentation.")
            return None, None

        # SMOTE interpolation in embedding space
        minority_embeds = embeddings[minority_idx]
        synthetic_embeds = []

        for _ in range(n_synthetic):
            # Pick two random minority nodes
            idx_pair = rng.choice(len(minority_idx), size=2, replace=True)
            z1 = minority_embeds[idx_pair[0]]
            z2 = minority_embeds[idx_pair[1]]
            # Interpolate
            lam = rng.uniform(0, 1)
            z_new = lam * z1 + (1 - lam) * z2
            synthetic_embeds.append(z_new)

        synthetic_embeddings = torch.stack(synthetic_embeds)
        synthetic_labels = torch.ones(n_synthetic, dtype=torch.long, device=self.device)

        return synthetic_embeddings, synthetic_labels

    def generate_edges(
        self,
        synthetic_embeds: torch.Tensor,
        all_embeds: torch.Tensor,
        threshold: float = 0.5,
        top_k: int = 5,
    ) -> torch.Tensor:
        """
        Generate edges for synthetic nodes using the edge generator.

        Connects each synthetic node to its top-k most probable neighbors.

        Args:
            synthetic_embeds: Embeddings of synthetic nodes [M, D].
            all_embeds: Embeddings of all existing nodes [N, D].
            threshold: Minimum probability for edge creation.
            top_k: Maximum number of edges per synthetic node.

        Returns:
            Edge index tensor [2, num_edges] for synthetic connections.
        """
        n_existing = all_embeds.shape[0]
        n_synthetic = synthetic_embeds.shape[0]
        edges_src, edges_dst = [], []

        with torch.no_grad():
            for i in range(n_synthetic):
                z_i = synthetic_embeds[i].unsqueeze(0).expand(n_existing, -1)
                probs = self.edge_gen(z_i, all_embeds)
                # Get top-k connections above threshold
                values, indices = torch.topk(probs, min(top_k, n_existing))
                valid = values >= threshold
                for j in indices[valid]:
                    # Synthetic node index = n_existing + i
                    edges_src.append(n_existing + i)
                    edges_dst.append(j.item())
                    # Bidirectional
                    edges_src.append(j.item())
                    edges_dst.append(n_existing + i)

        if len(edges_src) == 0:
            return torch.zeros((2, 0), dtype=torch.long, device=self.device)

        return torch.tensor([edges_src, edges_dst], dtype=torch.long, device=self.device)
