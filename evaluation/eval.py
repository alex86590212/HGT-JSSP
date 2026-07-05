"""Compare HGT checkpoint vs iGreedy across difficulty tiers."""

from __future__ import annotations

import argparse
import platform
import resource
import time
from pathlib import Path

import torch
import yaml

from intersection_scheduler.data.scenario_generator import ScenarioGenerator
from intersection_scheduler.environment.feasibility import compute_feasible_set
from intersection_scheduler.environment.graph_builder import build_hetero_graph
from intersection_scheduler.environment.intersection import IntersectionEnv
from intersection_scheduler.model.policy import SchedulingPolicy
from intersection_scheduler.utils.metrics import evaluate_hgt_vs_igreedy


def profile_inference_memory(policy: SchedulingPolicy, env: IntersectionEnv, scenario) -> None:
    """Report peak memory and latency for a single scheduling forward pass."""
    env.reset(scenario.vehicles)
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
    parser = argparse.ArgumentParser(description="Evaluate HGT vs iGreedy")
    parser.add_argument(
        "--checkpoint",
        default="results/checkpoint_best.pt",
        help="Path to .pt checkpoint",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
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

    profile_inference_memory(policy, IntersectionEnv(), gen.hard(n_vehicles=5))

    scenarios_by_tier = {
        "easy":   [gen.easy()   for _ in range(n)],
        "medium": [gen.medium() for _ in range(n)],
        "hard":   [gen.hard()   for _ in range(n)],
    }

    env = IntersectionEnv()
    csv_path = Path(args.output_csv) if args.output_csv else None

    print()
    results = evaluate_hgt_vs_igreedy(policy, env, scenarios_by_tier, output_csv=csv_path)

    all_hgt = sum(r["hgt_wt"] for r in results.values()) / len(results)
    all_ig  = sum(r["igreedy_wt"] for r in results.values()) / len(results)
    overall = (all_ig - all_hgt) / (all_ig + 1e-9) * 100.0
    print(f"[{'overall':6s}]  iGreedy={all_ig:.3f}  HGT={all_hgt:.3f}  improvement={overall:+.1f}%")


if __name__ == "__main__":
    main()
