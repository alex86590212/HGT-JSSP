"""Episode loop and training orchestration for the 4x4 two-lane intersection.

Mirrors trainer.py exactly (same PPO update, GAE, curriculum mechanics, reward
normalisation, HGT architecture) — the only difference is the scenario
generator and zone geometry, threaded through via IntersectionEnv.reset's
zone_positions parameter.
"""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.tensorboard import SummaryWriter

from intersection_scheduler.data.scenario_generator_4x4 import (
    ScenarioGenerator,
    ZONE_POSITIONS,
    get_curriculum_scenario,
)
from intersection_scheduler.environment.intersection import IntersectionEnv
from intersection_scheduler.model.policy import SchedulingPolicy
from intersection_scheduler.training.ppo import Transition, compute_gae, ppo_update
from intersection_scheduler.training.trainer import run_episode


def train(cfg: Dict[str, Any], output_dir: str = "results", resume: Optional[str] = None) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(out / "tb"))

    ppo_cfg = cfg.get("ppo", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Training on {device}: {torch.cuda.get_device_name(device)}", flush=True)
    else:
        print("Training on CPU (CUDA not available)", flush=True)

    policy = SchedulingPolicy(
        hidden_dim=model_cfg.get("hidden_dim", 128),
        num_heads=model_cfg.get("num_heads", 4),
        num_layers=model_cfg.get("num_layers", 3),
    ).to(device)
    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=ppo_cfg.get("lr", 3e-4),
    )

    start_episode = 1
    if resume is not None:
        checkpoint = torch.load(resume, weights_only=True, map_location=device)
        if isinstance(checkpoint, dict) and "policy" in checkpoint:
            policy.load_state_dict(checkpoint["policy"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_episode = checkpoint.get("episode", 0) + 1
        else:
            policy.load_state_dict(checkpoint)
        print(f"Resumed from {resume}, starting at episode {start_episode}", flush=True)

    gen = ScenarioGenerator(seed=42)
    env = IntersectionEnv()

    num_episodes = train_cfg.get("num_episodes", 50_000)
    log_interval = train_cfg.get("log_interval", 100)
    eval_interval = train_cfg.get("eval_interval", 1_000)
    checkpoint_interval = train_cfg.get("checkpoint_interval", 5_000)

    best_eval_waiting = float("inf")
    best_state = None

    ROLLOUT_N = 8
    buffer: List[Transition] = []
    update_stats = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}

    print(f"Starting training: {num_episodes} episodes (log every {log_interval})", flush=True)

    for episode in range(start_episode, num_episodes + 1):
        scenario = get_curriculum_scenario(episode, gen)
        transitions, stats = run_episode(policy, env, scenario, zone_positions=ZONE_POSITIONS)
        # Compute GAE immediately, in temporal order, before buffer shuffling
        compute_gae(
            transitions,
            gamma=ppo_cfg.get("gamma", 0.99),
            gae_lambda=ppo_cfg.get("gae_lambda", 0.95),
        )
        buffer.extend(transitions)

        if episode == start_episode:
            print(
                f"[{episode:6d}] first episode done "
                f"({stats['steps']} steps, wt={stats['waiting_time']:.2f})",
                flush=True,
            )

        if episode % ROLLOUT_N == 0:
            random.shuffle(buffer)
            update_stats = ppo_update(
                policy,
                optimizer,
                buffer,
                clip_epsilon=ppo_cfg.get("clip_epsilon", 0.5),
                epochs=ppo_cfg.get("epochs_per_update", 4),
                value_loss_coef=ppo_cfg.get("value_loss_coef", 0.5),
                entropy_coef=ppo_cfg.get("entropy_coef", 0.01),
                max_grad_norm=ppo_cfg.get("max_grad_norm", 0.5),
                mini_batch_size=ppo_cfg.get("mini_batch_size", 16),
            )
            buffer = []

        if episode % log_interval == 0:
            writer.add_scalar("train/waiting_time", stats["waiting_time"], episode)
            writer.add_scalar("train/makespan", stats["makespan"], episode)
            writer.add_scalar("train/total_reward", stats["total_reward"], episode)
            writer.add_scalar("train/steps", stats["steps"], episode)
            writer.add_scalar("train/actor_loss", update_stats["actor_loss"], episode)
            writer.add_scalar("train/critic_loss", update_stats["critic_loss"], episode)
            writer.add_scalar("train/entropy", update_stats["entropy"], episode)
            print(
                f"[{episode:6d}] wt={stats['waiting_time']:.2f}  "
                f"mkspan={stats['makespan']:.2f}  "
                f"rew={stats['total_reward']:.2f}  "
                f"steps={stats['steps']}",
                flush=True,
            )

        if episode % eval_interval == 0:
            eval_wt = _evaluate(policy, env, gen, n_scenarios=50)
            writer.add_scalar("eval/waiting_time", eval_wt, episode)
            print(f"  >>> EVAL waiting_time={eval_wt:.3f}", flush=True)
            if eval_wt < best_eval_waiting:
                best_eval_waiting = eval_wt
                best_state = copy.deepcopy(policy.state_dict())
                torch.save(best_state, out / "checkpoint_best.pt")

        if episode % checkpoint_interval == 0:
            torch.save({
                "episode": episode,
                "policy": policy.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, out / f"checkpoint_{episode}.pt")

    writer.close()
    if best_state is not None:
        torch.save(best_state, out / "checkpoint_best.pt")
    print(f"Training complete. Best eval waiting time: {best_eval_waiting:.3f}", flush=True)


def _evaluate(
    policy: SchedulingPolicy,
    env: IntersectionEnv,
    gen: ScenarioGenerator,
    n_scenarios: int = 50,
) -> float:
    policy.eval()
    total_wt = 0.0
    for _ in range(n_scenarios):
        scenario = gen.hard(n_vehicles=12)
        _, stats = run_episode(policy, env, scenario, deterministic=True, zone_positions=ZONE_POSITIONS)
        total_wt += stats["waiting_time"]
    policy.train()
    return total_wt / n_scenarios
