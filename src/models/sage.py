"""
GraphSAGE (Sample and Aggregate) for node classification.

Based on Hamilton, Ying & Leskovec (2017). Uses mean aggregation
with neighbor sampling and concatenation of self/neighbor embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGE(nn.Module):
    """
    GraphSAGE for binary node classification.

    Architecture: SAGEConv -> ReLU -> Dropout -> ... -> SAGEConv -> output

    Args:
        in_channels: Number of input features (166 for Elliptic).
        hidden_channels: Hidden dimension size.
        num_layers: Number of SAGE layers (default: 2).
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
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, num_classes))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x
