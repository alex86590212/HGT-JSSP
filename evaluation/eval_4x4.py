"""Compare HGT checkpoint vs iGreedy on the 4x4 two-lane intersection.

Mirrors eval.py (same output format, memory/latency profiling, per-tier
comparison) but for the 4x4 topology: threads the 4x4 ZONE_POSITIONS through
env.reset so graph node features use the correct geometry.

Self-contained rollout (does not import from trainer.py) so it avoids pulling
in the TensorBoard/TensorFlow import stack that trainer.py triggers.
"""

from __future__ import annotations

import argparse
import platform
import resource
import time
from pathlib import Path
from typing import List

import torch
import yaml

from intersection_scheduler.data.scenario_generator_4x4 import (
    ScenarioGenerator,
    ZONE_POSITIONS,
)
from intersection_scheduler.environment.feasibility import (
    compute_feasible_set,
    next_feasible_time,
)
from intersection_scheduler.environment.graph_builder import (
    build_hetero_graph,
    build_static_edges,
)
from intersection_scheduler.environment.intersection import IntersectionEnv
from intersection_scheduler.model.policy import SchedulingPolicy
from intersection_scheduler.utils.metrics import (
    episode_makespan,
    episode_waiting_time,
    igreedy,
)


def run_hgt_episode(policy: SchedulingPolicy, env: IntersectionEnv, scenario) -> float:
    """Deterministic HGT rollout on the 4x4 topology; returns waiting time.

    Self-contained copy of run_episode's deterministic path, threading the
    4x4 zone positions and reusing the cached static edges + single feasible
    mask (same optimizations as trainer.run_episode).
    """
    device = next(policy.parameters()).device
    env.reset(scenario.vehicles, zone_positions=ZONE_POSITIONS)
    static_edges = build_static_edges(env)

    done = False
    while not done:
        cpu_mask = compute_feasible_set(env)
        if not cpu_mask.any():
            next_t = next_feasible_time(env)
            if next_t is None:
                break
            env.current_time = next_t
            continue

        data = build_hetero_graph(
            env, feasible_mask=cpu_mask, static_edges=static_edges
        ).to(device)
        mask = cpu_mask.to(device)
        with torch.no_grad():
            dist, _ = policy(data, mask)
        action = int(dist.probs.argmax().item())
        env, _, done = env.step(action)

    return episode_waiting_time(env)


def profile_inference_memory(policy: SchedulingPolicy, env: IntersectionEnv, scenario) -> None:
    """Report peak memory and latency for a single scheduling forward pass."""
    env.reset(scenario.vehicles, zone_positions=ZONE_POSITIONS)
    data = build_hetero_graph(env)
    mask = compute_feasible_set(env)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        policy(data, mask)
    if use_cuda:
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    print("\n[memory] Single inference forward pass:")
    print(f"  latency: {elapsed_ms:.2f} ms")
    if use_cuda:
        peak_mb = torch.cuda.max_memory_allocated() / 1e6
        print(f"  peak GPU memory: {peak_mb:.2f} MB")
    else:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        divisor = 1e6 if platform.system() == "Darwin" else 1024.0
        peak_mb = rss / divisor
        print(f"  peak process RSS (CPU): {peak_mb:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="Evaluate 4x4 HGT vs iGreedy")
    parser.add_argument(
        "--checkpoint",
        default="results_4x4/checkpoint_best.pt",
        help="Path to .pt checkpoint",
    )
    parser.add_argument(
        "--config",
        default="configs/default_4x4.yaml",
        help="Model config (must match checkpoint architecture)",
    )
    parser.add_argument(
        "--n-scenarios",
        type=int,
        default=100,
        help="Scenarios per tier",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional path to save per-scenario CSV results",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg.get("model", {})

    policy = SchedulingPolicy(
        hidden_dim=model_cfg.get("hidden_dim", 128),
        num_heads=model_cfg.get("num_heads", 4),
        num_layers=model_cfg.get("num_layers", 3),
    )

    ckpt = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    if isinstance(ckpt, dict) and "policy" in ckpt:
        policy.load_state_dict(ckpt["policy"])
        ep = ckpt.get("episode", "?")
        print(f"Loaded checkpoint: {args.checkpoint}  (episode {ep})")
    else:
        policy.load_state_dict(ckpt)
        print(f"Loaded weights: {args.checkpoint}")

    policy.eval()

    gen = ScenarioGenerator(seed=args.seed)
    n = args.n_scenarios

    profile_inference_memory(policy, IntersectionEnv(), gen.hard(n_vehicles=12))

    scenarios_by_tier = {
        "easy":   [gen.easy()   for _ in range(n)],
        "medium": [gen.medium() for _ in range(n)],
        "hard":   [gen.hard()   for _ in range(n)],
    }

    env = IntersectionEnv()
    rows: List[dict] = []
    results = {}

    print()
    for tier, scenarios in scenarios_by_tier.items():
        hgt_wts, ig_wts = [], []
        for i, scenario in enumerate(scenarios):
            w_hgt = run_hgt_episode(policy, env, scenario)
            w_ig = igreedy(env, scenario, zone_positions=ZONE_POSITIONS)
            hgt_wts.append(w_hgt)
            ig_wts.append(w_ig)
            rows.append({
                "tier": tier,
                "scenario_id": i,
                "n_vehicles": len(scenario.vehicles),
                "hgt_waiting_time": w_hgt,
                "igreedy_waiting_time": w_ig,
            })

        mean_hgt = sum(hgt_wts) / len(hgt_wts)
        mean_ig = sum(ig_wts) / len(ig_wts)
        improvement = (mean_ig - mean_hgt) / (mean_ig + 1e-9) * 100.0
        results[tier] = {"hgt_wt": mean_hgt, "igreedy_wt": mean_ig}
        print(f"[{tier:6s}]  iGreedy={mean_ig:.3f}  HGT={mean_hgt:.3f}  improvement={improvement:+.1f}%")

    all_hgt = sum(r["hgt_wt"] for r in results.values()) / len(results)
    all_ig = sum(r["igreedy_wt"] for r in results.values()) / len(results)
    overall = (all_ig - all_hgt) / (all_ig + 1e-9) * 100.0
    print(f"[{'overall':6s}]  iGreedy={all_ig:.3f}  HGT={all_hgt:.3f}  improvement={overall:+.1f}%")

    if args.output_csv:
        import csv as csv_module
        out_path = Path(args.output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as f:
            writer = csv_module.DictWriter(
                f, fieldnames=["tier", "scenario_id", "n_vehicles",
                               "hgt_waiting_time", "igreedy_waiting_time"]
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
