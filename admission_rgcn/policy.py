"""Actor-critic R-GCN policy for admission decisions.

Action space per decision: one of the feasible pending vehicles, or (if
allowed) DEFER -- admit nobody now and look again shortly. The defer logit
comes from a graph-level embedding; vehicle logits from per-node scores.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, HeteroData
from torch_geometric.nn import RGCNConv, global_mean_pool
from torch_geometric.utils import to_dense_batch

from admission_rgcn.graph import EDGE_TYPES, NODE_DIMS, NODE_TYPES

DEFER = -1  # stored action value for the defer choice


class AdmissionRGCN(nn.Module):
    """R-GCN encoder: one weight matrix per edge type (mean aggregation) plus
    a self-loop weight. Node types are flattened into one index space so a
    single RGCNConv handles all relations; per-type input projections and
    per-type LayerNorm, residuals and readout match the HGT policy in
    admission_rl, so only the message-passing layer differs."""

    def __init__(self, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        self.input_proj = nn.ModuleDict({
            nt: nn.Linear(NODE_DIMS[nt], hidden_dim) for nt in NODE_TYPES
        })
        self.convs = nn.ModuleList([
            RGCNConv(hidden_dim, hidden_dim, num_relations=len(EDGE_TYPES), aggr="mean")
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.ModuleDict({nt: nn.LayerNorm(hidden_dim) for nt in NODE_TYPES})
            for _ in range(num_layers)
        ])

    def forward(self, x_dict, edge_index_dict):
        h = {nt: F.relu(self.input_proj[nt](x)) for nt, x in x_dict.items()}
        offsets, total = {}, 0
        for nt in NODE_TYPES:
            offsets[nt] = total
            total = total + h[nt].shape[0]
        edge_index, edge_type = [], []
        for r, (src, rel, dst) in enumerate(EDGE_TYPES):
            ei = edge_index_dict[(src, rel, dst)]
            edge_index.append(torch.stack([ei[0] + offsets[src], ei[1] + offsets[dst]]))
            edge_type.append(torch.full((ei.shape[1],), r, dtype=torch.long, device=ei.device))
        edge_index = torch.cat(edge_index, dim=1)
        edge_type = torch.cat(edge_type)
        for conv, norm_dict in zip(self.convs, self.norms):
            flat = conv(torch.cat([h[nt] for nt in NODE_TYPES]), edge_index, edge_type)
            h = {
                nt: norm_dict[nt](F.relu(flat[offsets[nt]:offsets[nt] + h[nt].shape[0]]) + h[nt])
                for nt in NODE_TYPES
            }
        return h


def _mlp(in_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))


class AdmissionPolicy(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        defer_init_bias: float = -2.0,
    ) -> None:
        super().__init__()
        self.gnn = AdmissionRGCN(hidden_dim, num_layers)
        pooled = hidden_dim * len(NODE_TYPES)
        self.actor = _mlp(hidden_dim + pooled, hidden_dim // 2)
        self.defer_head = _mlp(pooled, hidden_dim // 2)
        self.critic = _mlp(pooled, hidden_dim)
        # Start close to the order-only behaviour of the baselines: a
        # randomly initialised defer logit would hold vehicles back ~1/(n+1)
        # of the time, which is far worse than iGreedy and slows learning.
        nn.init.constant_(self.defer_head[-1].bias, defer_init_bias)

    def forward(
        self,
        batch: Batch,
        feasible_mask: torch.BoolTensor,
        defer_allowed: torch.BoolTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits [B, max_pending + 1], value [B]). Column -1 of
        logits is DEFER; padding and infeasible vehicles are -inf."""
        h = self.gnn(batch.x_dict, batch.edge_index_dict)
        n_graphs = batch.num_graphs
        pooled = torch.cat([
            global_mean_pool(h[nt], batch[nt].batch, size=n_graphs) for nt in NODE_TYPES
        ], dim=-1)

        pending_batch = batch["pending"].batch
        node_in = torch.cat([h["pending"], pooled[pending_batch]], dim=-1)
        scores = self.actor(node_in).squeeze(-1)
        scores = scores.masked_fill(~feasible_mask, float("-inf"))
        dense, _ = to_dense_batch(scores, pending_batch, fill_value=float("-inf"), batch_size=n_graphs)

        defer = self.defer_head(pooled).squeeze(-1)
        defer = defer.masked_fill(~defer_allowed, float("-inf"))
        logits = torch.cat([dense, defer.unsqueeze(-1)], dim=-1)
        value = self.critic(pooled).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def act(
        self,
        data: HeteroData,
        feasible_mask: torch.BoolTensor,
        defer_allowed: bool,
        deterministic: bool = False,
    ) -> Tuple[int, float, float]:
        """Single decision. Returns (action, log_prob, value) where action is
        the index into the graph's pending nodes, or DEFER."""
        batch = Batch.from_data_list([data])
        logits, value = self(batch, feasible_mask, torch.tensor([defer_allowed]))
        dist = torch.distributions.Categorical(logits=logits[0])
        idx = int(dist.probs.argmax()) if deterministic else int(dist.sample())
        log_prob = float(dist.log_prob(torch.tensor(idx)))
        n_pending = data["pending"].x.shape[0]
        action = DEFER if idx == logits.shape[1] - 1 else idx
        assert action == DEFER or idx < n_pending
        return action, log_prob, float(value[0])


def action_columns(actions: torch.Tensor, n_columns: int) -> torch.Tensor:
    """Map stored actions (pending index or DEFER) to logits columns."""
    return torch.where(actions == DEFER, torch.full_like(actions, n_columns - 1), actions)
