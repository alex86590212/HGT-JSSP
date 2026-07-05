"""PPO update logic with batched graph forward passes."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch_geometric.data import Batch, HeteroData


@dataclass
class Transition:
    data: HeteroData
    feasible_mask: torch.BoolTensor
    action: int
    log_prob: torch.Tensor
    value: torch.Tensor
    reward: float
    done: bool
    # Set after GAE computation, before PPO update
    advantage: float = 0.0
    ret: float = 0.0


def compute_gae(
    transitions: List[Transition],
    gamma: float,
    gae_lambda: float,
) -> None:
    """Compute GAE advantages and returns in-place on a single episode's transitions.

    Must be called on transitions in temporal order, before any shuffling.
    """
    gae = 0.0
    next_value = 0.0

    for t in reversed(transitions):
        mask = 0.0 if t.done else 1.0
        delta = t.reward + gamma * next_value * mask - t.value.item()
        gae = delta + gamma * gae_lambda * mask * gae
        t.advantage = gae
        t.ret = gae + t.value.item()
        next_value = t.value.item()


def ppo_update(
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    transitions: List[Transition],
    *,
    clip_epsilon: float = 0.5,
    epochs: int = 4,
    value_loss_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    mini_batch_size: int = 16,
) -> Dict[str, float]:
    """PPO update over a buffer of transitions.

    GAE must already be computed (advantage/ret set on each transition).
    Transitions are shuffled and processed in mini-batches for GPU efficiency.
    """
    device = next(policy.parameters()).device

    # Normalise advantages across the full buffer
    advs = torch.tensor([t.advantage for t in transitions], dtype=torch.float)
    advs = (advs - advs.mean()) / (advs.std() + 1e-8)
    for i, t in enumerate(transitions):
        t.advantage = advs[i].item()

    old_log_probs = torch.stack([t.log_prob.detach() for t in transitions]).to(device)
    actions = torch.tensor([t.action for t in transitions], dtype=torch.long, device=device)
    ret_tensor = torch.tensor([t.ret for t in transitions], dtype=torch.float, device=device)
    adv_tensor = torch.tensor([t.advantage for t in transitions], dtype=torch.float, device=device)

    total_actor_loss = 0.0
    total_critic_loss = 0.0
    total_entropy = 0.0
    n_updates = 0

    indices = list(range(len(transitions)))

    for _ in range(epochs):
        random.shuffle(indices)
        for start in range(0, len(indices), mini_batch_size):
            batch_idx = indices[start: start + mini_batch_size]

            # Build batched PyG graph
            batch_data = Batch.from_data_list(
                [transitions[i].data for i in batch_idx]
            ).to(device)

            # Stack feasible masks — each graph has its own op count, so we
            # can't simply cat; instead forward per-graph and accumulate.
            # For full GPU efficiency we forward the batched graph then split.
            batch_masks = torch.cat(
                [transitions[i].feasible_mask for i in batch_idx]
            ).to(device)

            batch_actions = actions[batch_idx]
            batch_old_lp = old_log_probs[batch_idx]
            batch_adv = adv_tensor[batch_idx]
            batch_ret = ret_tensor[batch_idx]

            # Forward: batched GNN pass, then split outputs per graph
            op_emb = policy.gnn(batch_data.x_dict, batch_data.edge_index_dict)

            # Split op embeddings back per graph using batch assignment vector
            op_batch = batch_data["operation"].batch  # [total_ops] graph index per op
            n_graphs = len(batch_idx)

            new_log_probs = []
            entropies = []
            values = []

            op_offset = 0
            for g in range(n_graphs):
                g_mask = op_batch == g
                g_op_emb = op_emb[g_mask]                          # [n_ops_g, hidden]
                g_feasible = batch_masks[op_offset: op_offset + g_op_emb.shape[0]]
                op_offset += g_op_emb.shape[0]

                # Actor
                logits = policy.actor(g_op_emb).squeeze(-1)
                logits = logits.masked_fill(~g_feasible, float("-inf"))
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs.append(dist.log_prob(batch_actions[g]))
                entropies.append(dist.entropy())

                # Critic: pool over unscheduled ops
                unscheduled = ~transitions[batch_idx[g]].data["operation"].x[:, 0].bool().to(device)
                pool = g_op_emb[unscheduled] if unscheduled.any() else g_op_emb
                values.append(policy.critic(pool.mean(dim=0)))

            new_log_probs = torch.stack(new_log_probs)             # [B]
            entropies = torch.stack(entropies)                     # [B]
            values = torch.stack(values).squeeze(-1)               # [B]

            ratio = torch.exp(new_log_probs - batch_old_lp)
            surr1 = ratio * batch_adv
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * batch_adv
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.functional.mse_loss(values, batch_ret)
            entropy = entropies.mean()
            loss = actor_loss + value_loss_coef * critic_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += entropy.item()
            n_updates += 1

    n_updates = max(n_updates, 1)
    return {
        "actor_loss": total_actor_loss / n_updates,
        "critic_loss": total_critic_loss / n_updates,
        "entropy": total_entropy / n_updates,
    }
