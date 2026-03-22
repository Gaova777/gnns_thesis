"""
Graph Convolutional Network (GCN) for node classification.

Based on Kipf & Welling (2017). Uses spectral convolution with
symmetric normalization of the adjacency matrix.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    """
    GCN for binary node classification.

    Architecture: GCNConv -> ReLU -> Dropout -> ... -> GCNConv -> output

    Args:
        in_channels: Number of input features (166 for Elliptic).
        hidden_channels: Hidden dimension size.
        num_layers: Number of GCN layers (default: 2).
        dropout: Dropout rate (default: 0.3).
        num_classes: Number of output classes (default: 2).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 2,
    ):
        super().__init__()
        self.dropout = dropout

        self.convs = nn.ModuleList()
        # First layer: input -> hidden
        self.convs.append(GCNConv(in_channels, hidden_channels))
        # Middle layers: hidden -> hidden
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        # Last layer: hidden -> output
        self.convs.append(GCNConv(hidden_channels, num_classes))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x
