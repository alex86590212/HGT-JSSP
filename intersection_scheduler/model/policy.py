"""Actor-critic policy built on IntersectionHGT."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from intersection_scheduler.model.hgt import IntersectionHGT


class SchedulingPolicy(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, num_layers: int) -> None:
        super().__init__()
        self.gnn = IntersectionHGT(hidden_dim, num_heads, num_layers)

        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        data: HeteroData,
        feasible_mask: torch.BoolTensor,
    ):
        op_emb = self.gnn(data.x_dict, data.edge_index_dict)   # [num_ops, hidden]

        # Actor: mask infeasible ops to -inf
        logits = self.actor(op_emb).squeeze(-1)                 # [num_ops]
        logits = logits.masked_fill(~feasible_mask, float("-inf"))
        dist = torch.distributions.Categorical(logits=logits)

        # Critic: mean-pool over unscheduled ops only
        unscheduled_mask = ~data["operation"].x[:, 0].bool()    # d(i,j) == 0
        pool_emb = op_emb[unscheduled_mask]
        if pool_emb.shape[0] == 0:
            pool_emb = op_emb
        value = self.critic(pool_emb.mean(dim=0))               # [1]

        return dist, value
