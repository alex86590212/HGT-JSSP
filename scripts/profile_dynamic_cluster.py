"""Standalone cluster profiling script for the dynamic scheduler slowdown.

Run directly on the cluster node (same environment as the real training job)
to see exactly where per-episode time goes there, rather than guessing from
local Mac numbers. No training loop, no PPO update — just isolates:
  1. Raw HGT forward-pass cost (the suspected PyG JIT/propagate overhead)
  2. compute_feasible_set cost
  3. build_hetero_graph cost
  4. A full run_episode, timed and broken into repeated single-call timings

Usage (from repo root, same env as training):
    PYTHONPATH=. python scripts/profile_dynamic_cluster.py
"""

from __future__ import annotations

import cProfile
import io
import pstats
import time

import torch

from dynamic_scheduler.data.traffic_generator import TrafficGenerator
from dynamic_scheduler.environment.dynamic_intersection import DynamicIntersectionEnv
from dynamic_scheduler.environment.feasibility import compute_feasible_set
from dynamic_scheduler.environment.graph_builder import build_hetero_graph
from dynamic_scheduler.training.trainer import run_episode
from intersection_scheduler.data.scenario_generator_4x4 import ZONE_POSITIONS
from intersection_scheduler.model.policy import SchedulingPolicy


def build_representative_graph():
    """Build one populated dynamic-env graph, matching the earlier local test."""
    env = DynamicIntersectionEnv(detection_window=10.0, commit_window=2.5, zone_positions=ZONE_POSITIONS)
    gen = TrafficGenerator(seed=1)
    env.reset(gen.hard(60.0), 60.0)
    for _ in range(20):
        m = compute_feasible_set(env)
        if m.any():
            idx = next(i for i in range(len(m)) if m[i].item())
            env.plan_operation(idx)
        env.advance_time()
    mask = compute_feasible_set(env)
    data = build_hetero_graph(env, feasible_mask=mask)
    return env, data, mask


def section(title: str) -> None:
    print(f"\n{'='*70}\n{title}\n{'='*70}", flush=True)


def main():
    print(f"torch version: {torch.__version__}", flush=True)
    print(f"torch num_threads (as configured): {torch.get_num_threads()}", flush=True)
    print(f"torch num_interop_threads: {torch.get_num_interop_threads()}", flush=True)

    section("1. Building representative graph")
    t0 = time.perf_counter()
    env, data, mask = build_representative_graph()
    print(f"graph: {data['operation'].x.shape[0]} ops, "
          f"{data['vehicle'].x.shape[0]} vehicles, build took {time.perf_counter()-t0:.3f}s", flush=True)

    section("2. Policy construction + FIRST forward pass (cold — includes any JIT/codegen)")
    policy = SchedulingPolicy(hidden_dim=128, num_heads=4, num_layers=3)
    policy.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        policy(data, mask)
    print(f"FIRST forward pass: {(time.perf_counter()-t0)*1000:.1f} ms  <-- if this is huge, "
          f"it's one-time JIT/codegen cost, not per-call", flush=True)

    section("3. Repeated forward passes (warm — steady-state cost)")
    N = 200
    times = []
    with torch.no_grad():
        for _ in range(N):
            t0 = time.perf_counter()
            policy(data, mask)
            times.append(time.perf_counter() - t0)
    times.sort()
    print(f"N={N} warm forward passes:", flush=True)
    print(f"  min={times[0]*1000:.2f}ms  median={times[N//2]*1000:.2f}ms  "
          f"max={times[-1]*1000:.2f}ms  mean={sum(times)/N*1000:.2f}ms", flush=True)
    print(f"  first 5 (ms): {[round(t*1000,2) for t in times[:5]]}", flush=True)
    print(f"  last 5  (ms): {[round(t*1000,2) for t in times[-5:]]}", flush=True)

    section("4. compute_feasible_set repeated timing")
    N = 200
    t0 = time.perf_counter()
    for _ in range(N):
        compute_feasible_set(env)
    dt = time.perf_counter() - t0
    print(f"N={N} calls, {dt/N*1000:.2f} ms/call avg", flush=True)

    section("5. build_hetero_graph repeated timing")
    N = 200
    t0 = time.perf_counter()
    for _ in range(N):
        build_hetero_graph(env, feasible_mask=mask)
    dt = time.perf_counter() - t0
    print(f"N={N} calls, {dt/N*1000:.2f} ms/call avg", flush=True)

    section("6. Full run_episode x3 (real workload, wall-clock per episode)")
    for i in range(3):
        policy2 = SchedulingPolicy(hidden_dim=128, num_heads=4, num_layers=3)
        gen = TrafficGenerator(seed=10 + i)
        env2 = DynamicIntersectionEnv(detection_window=10.0, commit_window=2.5, zone_positions=ZONE_POSITIONS)
        arrivals = gen.hard(duration=60.0)
        t0 = time.perf_counter()
        transitions, stats = run_episode(policy2, env2, arrivals, 60.0)
        dt = time.perf_counter() - t0
        print(f"  episode {i}: {stats['steps']} steps, {stats['n_vehicles_seen']} vehicles, "
              f"{dt:.2f}s wall  ({dt/max(stats['steps'],1)*1000:.2f} ms/step)", flush=True)

    section("7. cProfile of one full run_episode (top 20 by cumulative time)")
    gen = TrafficGenerator(seed=99)
    policy3 = SchedulingPolicy(hidden_dim=128, num_heads=4, num_layers=3)
    env3 = DynamicIntersectionEnv(detection_window=10.0, commit_window=2.5, zone_positions=ZONE_POSITIONS)
    arrivals = gen.hard(duration=60.0)
    pr = cProfile.Profile()
    pr.enable()
    run_episode(policy3, env3, arrivals, 60.0)
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(20)
    print(s.getvalue(), flush=True)

    print("\nDONE.", flush=True)


if __name__ == "__main__":
    main()
