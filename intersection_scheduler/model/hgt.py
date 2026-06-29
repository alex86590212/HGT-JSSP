"""HGT policy backbone for intersection scheduling."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, Linear


class IntersectionHGT(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, num_layers: int) -> None:
        super().__init__()

        self.node_types = ["operation", "vehicle", "zone"]
        self.edge_types = [
            ("operation", "seq",      "operation"),
            ("operation", "lane",     "operation"),
            ("operation", "conflict", "operation"),
            ("vehicle",   "owns",     "operation"),
            ("zone",      "hosts",    "operation"),
        ]
        metadata = (self.node_types, self.edge_types)

        # Lazy input projection: -1 lets PyG infer input dim on first forward
        self.input_proj = nn.ModuleDict({
            ntype: Linear(-1, hidden_dim)
            for ntype in self.node_types
        })

        self.convs = nn.ModuleList([
            HGTConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                metadata=metadata,
                heads=num_heads,
            )
            for _ in range(num_layers)
        ])

        self.norms = nn.ModuleList([
            nn.ModuleDict({
                ntype: nn.LayerNorm(hidden_dim)
                for ntype in self.node_types
            })
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x_dict: dict,
        edge_index_dict: dict,
    ) -> torch.Tensor:
        h = {
            ntype: F.relu(self.input_proj[ntype](x))
            for ntype, x in x_dict.items()
        }

        for conv, norm_dict in zip(self.convs, self.norms):
            h_new = conv(h, edge_index_dict)
            # HGTConv only returns destination node types (operation here).
            # Carry forward source-only types (vehicle, zone) unchanged so
            # the next layer can still look them up as message sources.
            h_next = dict(h)  # preserve vehicle, zone embeddings
            for ntype, emb in h_new.items():
                if emb is not None and ntype in norm_dict:
                    h_next[ntype] = norm_dict[ntype](F.relu(emb) + h[ntype])
            h = h_next

        return h["operation"]   # [num_ops, hidden_dim]
