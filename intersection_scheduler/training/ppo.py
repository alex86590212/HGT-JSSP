"""PPO update logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData


@dataclass
class Transition:
    data: HeteroData
    feasible_mask: torch.BoolTensor
    action: int
    log_prob: torch.Tensor
    value: torch.Tensor
    reward: float
    done: bool


def compute_gae(
    rewards: List[float],
    values: List[torch.Tensor],
    dones: List[bool],
    gamma: float,
    gae_lambda: float,
) -> Tuple[List[float], List[float]]:
    """Compute GAE advantages and discounted returns."""
    n = len(rewards)
    advantages = [0.0] * n
    returns = [0.0] * n
    gae = 0.0
    next_value = 0.0

    for t in reversed(range(n)):
        mask = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value * mask - values[t].item()
        gae = delta + gamma * gae_lambda * mask * gae
        advantages[t] = gae
        returns[t] = gae + values[t].item()
        next_value = values[t].item()

    return advantages, returns


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
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> dict:
    rewards = [t.reward for t in transitions]
    values = [t.value.detach() for t in transitions]
    dones = [t.done for t in transitions]

    advantages, returns = compute_gae(rewards, values, dones, gamma, gae_lambda)

    adv_tensor = torch.tensor(advantages, dtype=torch.float)
    ret_tensor = torch.tensor(returns, dtype=torch.float)
    # Normalise advantages
    adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)

    old_log_probs = torch.stack([t.log_prob.detach() for t in transitions])
    actions = torch.tensor([t.action for t in transitions], dtype=torch.long)

    total_actor_loss = 0.0
    total_critic_loss = 0.0
    total_entropy = 0.0

    for _ in range(epochs):
        for idx, t in enumerate(transitions):
            dist, value = policy(t.data, t.feasible_mask)
            new_log_prob = dist.log_prob(actions[idx])
            entropy = dist.entropy()

            ratio = torch.exp(new_log_prob - old_log_probs[idx])
            adv = adv_tensor[idx]
            ret = ret_tensor[idx]

            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * adv
            actor_loss = -torch.min(surr1, surr2)
            critic_loss = nn.functional.mse_loss(value.squeeze(), ret)
            loss = actor_loss + value_loss_coef * critic_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += entropy.item()

    n = len(transitions) * epochs
    return {
        "actor_loss": total_actor_loss / n,
        "critic_loss": total_critic_loss / n,
        "entropy": total_entropy / n,
    }
