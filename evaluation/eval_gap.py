"""Optimality gap evaluation: HGT and iGreedy vs OR-Tools exact solver."""

from __future__ import annotations

import argparse
import csv as csv_module
from pathlib import Path
from typing import Dict, List

import torch
import yaml

from intersection_scheduler.data.scenario_generator import Scenario, ScenarioGenerator
from intersection_scheduler.environment.intersection import IntersectionEnv
from intersection_scheduler.model.policy import SchedulingPolicy
from intersection_scheduler.training.trainer import run_episode
from intersection_scheduler.utils.metrics import igreedy

from evaluation.optimal_solver import compute_gap, solve_optimal

_LEFT_TURNS = {"N_E", "S_W", "E_S", "W_N"}


def analyse_conflict_density(scenarios: List[Scenario]) -> dict:
    """Compute Type-3 conflict density stats across a list of scenarios."""
    n_conflicts_list = []
    n_competing_zones_list = []
    vehicles_per_contested_zone: List[float] = []
    has_left_turn_flags = []

    for scenario in scenarios:
        zone_to_vehicles: Dict[int, List[int]] = {}
        for v in scenario.vehicles:
            for z in v.route:
                zone_to_vehicles.setdefault(z, []).append(v.id)

        contested = {z: vs for z, vs in zone_to_vehicles.items() if len(vs) > 1}
        n_competing_zones_list.append(len(contested))

        n_conflicts = 0
        for vs in contested.values():
            k = len(vs)
            n_conflicts += k * (k - 1) // 2
        n_conflicts_list.append(n_conflicts)

        if contested:
            vehicles_per_contested_zone.append(
                sum(len(vs) for vs in contested.values()) / len(contested)
            )

        has_left_turn_flags.append(any(m in _LEFT_TURNS for m in scenario.manoeuvres))

    n = len(scenarios) or 1
    return {
        "mean_n_type3_conflicts": sum(n_conflicts_list) / n,
        "pct_zero_conflicts": sum(1 for c in n_conflicts_list if c == 0) / n * 100.0,
        "pct_left_turns": sum(has_left_turn_flags) / n * 100.0,
        "mean_vehicles_per_contested_zone": (
            sum(vehicles_per_contested_zone) / len(vehicles_per_contested_zone)
            if vehicles_per_contested_zone else 0.0
        ),
        "n_conflicts_per_scenario": n_conflicts_list,
        "n_competing_zones_per_scenario": n_competing_zones_list,
    }


def _count_conflicts_and_zones(scenario: Scenario) -> tuple:
    zone_to_vehicles: Dict[int, List[int]] = {}
    for v in scenario.vehicles:
        for z in v.route:
            zone_to_vehicles.setdefault(z, []).append(v.id)
    contested = [vs for vs in zone_to_vehicles.values() if len(vs) > 1]
    n_conflicts = sum(len(vs) * (len(vs) - 1) // 2 for vs in contested)
    return n_conflicts, len(contested)


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
    else:
        policy.load_state_dict(ckpt)
    policy.eval()
    return policy


def run_tier(
    tier: str,
    scenarios: List[Scenario],
    policy: SchedulingPolicy,
    env: IntersectionEnv,
    time_limit: float,
    rows: List[dict],
) -> dict:
    gaps_hgt = []
    gaps_igreedy = []
    solved = 0

    for i, scenario in enumerate(scenarios):
        _, stats = run_episode(policy, env, scenario, deterministic=True)
        w_hgt = stats["waiting_time"]

        w_ig = igreedy(env, scenario)

        w_star = solve_optimal(
            scenario.vehicles, scenario.manoeuvres, time_limit_seconds=time_limit
        )

        n_conflicts, n_zones_contested = _count_conflicts_and_zones(scenario)

        row = {
            "tier": tier,
            "scenario_id": i,
            "n_vehicles": len(scenario.vehicles),
            "n_conflicts": n_conflicts,
            "n_zones_contested": n_zones_contested,
            "W_hgt": w_hgt,
            "W_igreedy": w_ig,
            "W_star": w_star if w_star is not None else "",
            "gap_hgt_pct": "",
            "gap_igreedy_pct": "",
            "solved": w_star is not None,
        }

        if w_star is not None:
            solved += 1
            gap_hgt = compute_gap(w_hgt, w_star)
            gap_ig = compute_gap(w_ig, w_star)
            if gap_hgt is not None:
                gaps_hgt.append(gap_hgt)
                row["gap_hgt_pct"] = gap_hgt
            if gap_ig is not None:
                gaps_igreedy.append(gap_ig)
                row["gap_igreedy_pct"] = gap_ig

        rows.append(row)

    mean_gap_hgt = sum(gaps_hgt) / len(gaps_hgt) if gaps_hgt else float("nan")
    mean_gap_ig = sum(gaps_igreedy) / len(gaps_igreedy) if gaps_igreedy else float("nan")

    print(
        f"[{tier:6s}]  scenarios={len(scenarios)}  solved={solved}  "
        f"HGT_gap={mean_gap_hgt:.1f}%  iGreedy_gap={mean_gap_ig:.1f}%",
        flush=True,
    )

    return {
        "n_scenarios": len(scenarios),
        "solved": solved,
        "mean_gap_hgt": mean_gap_hgt,
        "mean_gap_igreedy": mean_gap_ig,
    }


def main():
    parser = argparse.ArgumentParser(description="HGT/iGreedy optimality gap vs OR-Tools")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--n_scenarios", type=int, default=100)
    parser.add_argument("--time_limit", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="results/gap_eval.csv")
    args = parser.parse_args()

    policy = load_policy(args.checkpoint, args.config)
    print(f"Loaded checkpoint: {args.checkpoint}", flush=True)

    gen = ScenarioGenerator(seed=args.seed)
    n = args.n_scenarios

    scenarios_by_tier = {
        "easy":   [gen.easy()   for _ in range(n)],
        "medium": [gen.medium() for _ in range(n)],
        "hard":   [gen.hard()   for _ in range(n)],
    }

    env = IntersectionEnv()
    rows: List[dict] = []
    tier_summaries: Dict[str, dict] = {}

    for tier, scenarios in scenarios_by_tier.items():
        tier_summaries[tier] = run_tier(
            tier, scenarios, policy, env, args.time_limit, rows
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "tier", "scenario_id", "n_vehicles", "n_conflicts", "n_zones_contested",
        "W_hgt", "W_igreedy", "W_star", "gap_hgt_pct", "gap_igreedy_pct", "solved",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv_module.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {out_path}", flush=True)

    total_scenarios = sum(s["n_scenarios"] for s in tier_summaries.values())
    total_solved = sum(s["solved"] for s in tier_summaries.values())
    all_gaps_hgt = [r["gap_hgt_pct"] for r in rows if r["gap_hgt_pct"] != ""]
    all_gaps_ig = [r["gap_igreedy_pct"] for r in rows if r["gap_igreedy_pct"] != ""]
    overall_gap_hgt = sum(all_gaps_hgt) / len(all_gaps_hgt) if all_gaps_hgt else float("nan")
    overall_gap_ig = sum(all_gaps_ig) / len(all_gaps_ig) if all_gaps_ig else float("nan")

    print("\nFinal summary:")
    print(
        f"{'Tier':8s} | {'Scenarios':9s} | {'Solved':6s} | "
        f"{'HGT gap (mean)':>14s} | {'iGreedy gap (mean)':>18s} | {'HGT vs iGreedy':>14s}"
    )
    for tier, s in tier_summaries.items():
        hgt_vs_ig = s["mean_gap_igreedy"] - s["mean_gap_hgt"]
        print(
            f"{tier:8s} | {s['n_scenarios']:9d} | {s['solved']:6d} | "
            f"{s['mean_gap_hgt']:13.1f}% | {s['mean_gap_igreedy']:17.1f}% | "
            f"{hgt_vs_ig:+13.1f}%"
        )
    overall_hgt_vs_ig = overall_gap_ig - overall_gap_hgt
    print(
        f"{'overall':8s} | {total_scenarios:9d} | {total_solved:6d} | "
        f"{overall_gap_hgt:13.1f}% | {overall_gap_ig:17.1f}% | {overall_hgt_vs_ig:+13.1f}%"
    )
    print(
        f"\nSolved {total_solved}/{total_scenarios} scenarios within time limit. "
        f"Skipped {total_scenarios - total_solved}."
    )

    # Conflict density analysis (Component 3)
    print("\nEasy tier conflict analysis:")
    easy_density = analyse_conflict_density(scenarios_by_tier["easy"])
    n_easy = len(scenarios_by_tier["easy"])
    mean_vehicles_per_lane = (
        sum(len(s.vehicles) for s in scenarios_by_tier["easy"]) / n_easy / 4.0
    )  # 4 entry lanes (N/S/E/W)
    print(f"  Mean Type-3 conflicts per scenario: {easy_density['mean_n_type3_conflicts']:.1f}")
    print(f"  Scenarios with zero conflicts:      {easy_density['pct_zero_conflicts']:.0f}%")
    print(f"  Scenarios with left turns:          {easy_density['pct_left_turns']:.0f}%")
    print(f"  Mean vehicles per contested zone:   {easy_density['mean_vehicles_per_contested_zone']:.1f}")
    print("\n  Paper's low-density definition: ~0.3 vehicles per lane average")
    print(f"  Our easy tier average: {mean_vehicles_per_lane:.1f} vehicles per lane")


if __name__ == "__main__":
    main()
