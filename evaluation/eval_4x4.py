"""Compare HGT checkpoint vs iGreedy on the 4x4 two-lane intersection.

Threads the 4x4 ZONE_POSITIONS through env.reset so the graph node features
use the correct geometry. Reports mean waiting time per difficulty tier for
both HGT (deterministic) and the iGreedy baseline, plus the improvement.
"""

from __future__ import annotations

import argparse
import csv as csv_module
from pathlib import Path
from typing import List

import torch
import yaml

from intersection_scheduler.data.scenario_generator_4x4 import (
    ScenarioGenerator,
    ZONE_POSITIONS,
)
from intersection_scheduler.environment.intersection import IntersectionEnv
from intersection_scheduler.model.policy import SchedulingPolicy
from intersection_scheduler.training.trainer import run_episode
from intersection_scheduler.utils.metrics import igreedy


def load_policy(checkpoint: str, config: str) -> SchedulingPolicy:
    with open(config) as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg.get("model", {})
    policy = SchedulingPolicy(
        hidden_dim=model_cfg.get("hidden_dim", 128),
        num_heads=model_cfg.get("num_heads", 4),
        num_layers=model_cfg.get("num_layers", 3),
    )
    ckpt = torch.load(checkpoint, weights_only=True, map_location="cpu")
    if isinstance(ckpt, dict) and "policy" in ckpt:
        policy.load_state_dict(ckpt["policy"])
        ep = ckpt.get("episode", "?")
        print(f"Loaded checkpoint: {checkpoint}  (episode {ep})")
    else:
        policy.load_state_dict(ckpt)
        print(f"Loaded weights: {checkpoint}")
    policy.eval()
    return policy


def main():
    parser = argparse.ArgumentParser(description="Evaluate 4x4 HGT vs iGreedy")
    parser.add_argument("--checkpoint", default="results_4x4/checkpoint_best.pt")
    parser.add_argument("--config", default="configs/default_4x4.yaml")
    parser.add_argument("--n-scenarios", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    policy = load_policy(args.checkpoint, args.config)

    gen = ScenarioGenerator(seed=args.seed)
    n = args.n_scenarios
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
            _, stats = run_episode(
                policy, env, scenario,
                deterministic=True, zone_positions=ZONE_POSITIONS,
            )
            w_hgt = stats["waiting_time"]
            w_ig = igreedy(env, scenario, zone_positions=ZONE_POSITIONS)
            hgt_wts.append(w_hgt)
            ig_wts.append(w_ig)
            rows.append({
                "tier": tier,
                "scenario_id": i,
                "n_vehicles": len(scenario.vehicles),
                "hgt_wt": w_hgt,
                "igreedy_wt": w_ig,
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
        out_path = Path(args.output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as f:
            writer = csv_module.DictWriter(
                f, fieldnames=["tier", "scenario_id", "n_vehicles", "hgt_wt", "igreedy_wt"]
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
