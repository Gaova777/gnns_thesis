"""
Graph Attention Network (GAT) for node classification.

Based on Veličković et al. (2018). Uses multi-head attention
mechanisms to learn importance weights for neighbor aggregation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    """
    GAT for binary node classification.

    Architecture:
        GATConv (multi-head, concat) -> ELU -> Dropout -> ...
        -> GATConv (multi-head, mean) -> output

    Args:
        in_channels: Number of input features (166 for Elliptic).
        hidden_channels: Hidden dimension PER HEAD.
        num_layers: Number of GAT layers (default: 2).
        heads: Number of attention heads (default: 4).
        dropout: Dropout rate (default: 0.3).
        num_classes: Number of output classes (default: 2).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.3,
        num_classes: int = 2,
    ):
        super().__init__()
        self.dropout = dropout

        self.convs = nn.ModuleList()
        # First layer: in_channels -> hidden_channels * heads (concat)
        self.convs.append(
            GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        )
        # Middle layers: hidden_channels * heads -> hidden_channels * heads
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    hidden_channels * heads,
                    hidden_channels,
                    heads=heads,
                    dropout=dropout,
                )
            )
        # Last layer: hidden_channels * heads -> num_classes (mean over heads)
        self.convs.append(
            GATConv(
                hidden_channels * heads,
                num_classes,
                heads=1,
                concat=False,
                dropout=dropout,
            )
        )

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x
