"""Equivalence test: optimized run_episode must produce identical trajectories.

Runs the same scenarios through run_episode deterministically and asserts the
resulting trajectory (actions, rewards, waiting time, makespan, steps) is
identical to a from-scratch reference that mimics the original per-step
rebuild-everything behavior. Guards against the run_episode wiring (threaded
mask + cached static edges) changing any decision or metric.
"""

import torch

from intersection_scheduler.data.scenario_generator import ScenarioGenerator as Gen3x3
from intersection_scheduler.data.scenario_generator_4x4 import (
    ScenarioGenerator as Gen4x4,
    ZONE_POSITIONS as ZP_4x4,
)
from intersection_scheduler.environment.feasibility import (
    compute_feasible_set,
    next_feasible_time,
)
from intersection_scheduler.environment.graph_builder import build_hetero_graph
from intersection_scheduler.environment.intersection import IntersectionEnv
from intersection_scheduler.model.policy import SchedulingPolicy
from intersection_scheduler.training.trainer import run_episode


def _reference_episode(policy, env, scenario, zone_positions):
    """Original-style rollout: rebuild graph and recompute mask from scratch
    every step, no caching, no threaded mask. This is the behavior the
    optimized run_episode must match exactly."""
    device = next(policy.parameters()).device
    env.reset(scenario.vehicles, zone_positions=zone_positions)
    actions, rewards = [], []
    done = False
    while not done:
        data = build_hetero_graph(env).to(device)
        mask = compute_feasible_set(env).to(device)
        if not mask.any():
            nt = next_feasible_time(env)
            if nt is None:
                break
            env.current_time = nt
            continue
        with torch.no_grad():
            dist, _ = policy(data, mask)
        action = int(dist.probs.argmax().item())
        env, reward, done = env.step(action)
        actions.append(action)
        rewards.append(reward)
    from intersection_scheduler.utils.metrics import episode_waiting_time, episode_makespan
    return {
        "actions": actions,
        "rewards": rewards,
        "waiting_time": episode_waiting_time(env),
        "makespan": episode_makespan(env),
    }


def _check_tier(gen, zone_positions):
    policy = SchedulingPolicy(hidden_dim=32, num_heads=2, num_layers=2)
    policy.eval()
    env = IntersectionEnv()
    for tier in ("easy", "medium", "hard"):
        for _ in range(5):
            scenario = getattr(gen, tier)()

            ref = _reference_episode(policy, env, scenario, zone_positions)

            transitions, stats = run_episode(
                policy, env, scenario,
                deterministic=True, zone_positions=zone_positions,
            )
            opt_actions = [t.action for t in transitions]
            opt_rewards = [t.reward for t in transitions]

            assert opt_actions == ref["actions"], f"{tier}: action sequence differs"
            assert opt_rewards == ref["rewards"], f"{tier}: reward sequence differs"
            assert abs(stats["waiting_time"] - ref["waiting_time"]) < 1e-12, \
                f"{tier}: waiting_time differs"
            assert abs(stats["makespan"] - ref["makespan"]) < 1e-12, \
                f"{tier}: makespan differs"


def test_run_episode_equivalence_3x3():
    _check_tier(Gen3x3(seed=7), zone_positions=None)


def test_run_episode_equivalence_4x4():
    _check_tier(Gen4x4(seed=7), zone_positions=ZP_4x4)
