"""
Topology Adaptive Graph Convolutional Network (TAGCN) for node classification.

Based on Du et al. (2017), arXiv:1710.10370. Uses learnable polynomial filters
of order K to capture multi-scale topological patterns in a single layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TAGConv


class TAGCN(nn.Module):
    """
    TAGCN for binary node classification.

    Architecture: TAGConv(K) -> ReLU -> Dropout -> ... -> TAGConv(K) -> output

    Args:
        in_channels: Number of input features (166 for Elliptic).
        hidden_channels: Hidden dimension size.
        num_layers: Number of TAGCN layers (default: 2).
        K: Order of the polynomial filter (default: 3, the standard TAGCN default).
        dropout: Dropout rate (default: 0.3).
        num_classes: Number of output classes (default: 2).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_layers: int = 2,
        K: int = 3,
        dropout: float = 0.3,
        num_classes: int = 2,
    ):
        super().__init__()
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(TAGConv(in_channels, hidden_channels, K=K))
        for _ in range(num_layers - 2):
            self.convs.append(TAGConv(hidden_channels, hidden_channels, K=K))
        self.convs.append(TAGConv(hidden_channels, num_classes, K=K))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x
