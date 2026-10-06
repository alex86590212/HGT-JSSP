"""GAE + clipped PPO update for AdmissionPolicy (batched graph forwards)."""

from __future__ import annotations

import random
from typing import Dict, List

import torch
import torch.nn as nn
from torch_geometric.data import Batch

from admission_rl.policy import AdmissionPolicy, action_columns
from admission_rl.rollout import Transition


def compute_gae(transitions: List[Transition], gamma: float, gae_lambda: float) -> None:
    """In place, on one episode's transitions in temporal order."""
    gae = 0.0
    next_value = 0.0
    for tr in reversed(transitions):
        mask = 0.0 if tr.done else 1.0
        delta = tr.reward + gamma * next_value * mask - tr.value
        gae = delta + gamma * gae_lambda * mask * gae
        tr.advantage = gae
        tr.ret = gae + tr.value
        next_value = tr.value


def ppo_update(
    policy: AdmissionPolicy,
    optimizer: torch.optim.Optimizer,
    transitions: List[Transition],
    *,
    clip_epsilon: float,
    epochs: int,
    mini_batch_size: int,
    value_loss_coef: float,
    entropy_coef: float,
    max_grad_norm: float,
) -> Dict[str, float]:
    advs = torch.tensor([tr.advantage for tr in transitions], dtype=torch.float)
    advs = (advs - advs.mean()) / (advs.std() + 1e-8)
    rets = torch.tensor([tr.ret for tr in transitions], dtype=torch.float)
    old_lp = torch.tensor([tr.log_prob for tr in transitions], dtype=torch.float)
    actions = torch.tensor([tr.action for tr in transitions], dtype=torch.long)

    totals = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clip_frac": 0.0}
    n_updates = 0
    indices = list(range(len(transitions)))
    for _ in range(epochs):
        random.shuffle(indices)
        for start in range(0, len(indices), mini_batch_size):
            idx = indices[start:start + mini_batch_size]
            batch = Batch.from_data_list([transitions[i].data for i in idx])
            mask = torch.cat([transitions[i].feasible_mask for i in idx])
            defer_allowed = torch.tensor([transitions[i].defer_allowed for i in idx])
            logits, values = policy(batch, mask, defer_allowed)
            dist = torch.distributions.Categorical(logits=logits)
            new_lp = dist.log_prob(action_columns(actions[idx], logits.shape[1]))
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_lp - old_lp[idx])
            adv = advs[idx]
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * adv
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.functional.mse_loss(values, rets[idx])
            loss = actor_loss + value_loss_coef * critic_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                totals["approx_kl"] += float((old_lp[idx] - new_lp).mean())
                totals["clip_frac"] += float(((ratio - 1.0).abs() > clip_epsilon).float().mean())
            totals["actor_loss"] += actor_loss.item()
            totals["critic_loss"] += critic_loss.item()
            totals["entropy"] += entropy.item()
            n_updates += 1

    return {k: v / max(n_updates, 1) for k, v in totals.items()}
