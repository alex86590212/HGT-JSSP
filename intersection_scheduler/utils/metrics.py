"""Metrics, iGreedy baseline, and comparative evaluation."""

from __future__ import annotations

import copy
import csv
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from intersection_scheduler.environment.intersection import IntersectionEnv
    from intersection_scheduler.data.scenario_generator import Scenario


# ------------------------------------------------------------------
# Per-episode metrics helpers
# ------------------------------------------------------------------

def episode_waiting_time(env: "IntersectionEnv") -> float:
    """Mean waiting time across all vehicles.

    Waiting time for vehicle i = c(i, n_i) - r_i - sum(p(i,j))
    i.e. total idle time spent waiting.
    """
    total = 0.0
    for v in env.vehicles:
        ops = [o for o in env.operations if o.vehicle_id == v.id]
        last = max(ops, key=lambda o: o.route_position)
        min_finish = v.arrival_time + sum(o.processing_time for o in ops)
        total += max(0.0, last.earliest_finish - min_finish)
    return total / len(env.vehicles) if env.vehicles else 0.0


def episode_makespan(env: "IntersectionEnv") -> float:
    """Makespan = time at which the last operation completes."""
    if not env.operations:
        return 0.0
    return max(o.earliest_finish for o in env.operations)


# ------------------------------------------------------------------
# iGreedy baseline
# ------------------------------------------------------------------

def igreedy(env: "IntersectionEnv", scenario: "Scenario") -> float:
    """Run iGreedy on scenario and return mean waiting time.

    Rule: at each step pick the feasible op belonging to the vehicle
    with the earliest arrival time; break ties by route position.
    """
    from intersection_scheduler.environment.feasibility import compute_feasible_set, next_feasible_time

    env.reset(scenario.vehicles)
    done = False
    while not done:
        mask = compute_feasible_set(env)
        if not mask.any():
            next_t = next_feasible_time(env)
            if next_t is None:
                break
            env.current_time = next_t
            continue

        best_idx: Optional[int] = None
        best_key: Optional[Tuple] = None
        for idx, op in enumerate(env.operations):
            if not mask[idx].item():
                continue
            v = next((v for v in env.vehicles if v.id == op.vehicle_id), None)
            if v is None:
                continue
            key = (v.arrival_time, op.route_position, op.vehicle_id)
            if best_key is None or key < best_key:
                best_key = key
                best_idx = idx

        if best_idx is None:
            break

        env, _reward, done = env.step(best_idx)

    return episode_waiting_time(env)


# ------------------------------------------------------------------
# Comparative evaluation
# ------------------------------------------------------------------

def evaluate_hgt_vs_igreedy(
    policy,
    env: "IntersectionEnv",
    scenarios_by_tier: Dict[str, List["Scenario"]],
    output_csv: Optional[Path] = None,
) -> Dict[str, Dict[str, float]]:
    """Run both HGT and iGreedy on each tier and report statistics.

    Returns a dict: tier -> {hgt_wt, igreedy_wt, improvement_pct}
    """
    from intersection_scheduler.training.trainer import run_episode

    results: Dict[str, Dict[str, float]] = {}
    all_rows: List[Dict] = []

    for tier, scenarios in scenarios_by_tier.items():
        hgt_wts: List[float] = []
        ig_wts: List[float] = []

        for scenario in scenarios:
            # HGT
            _, stats = run_episode(policy, env, scenario, deterministic=True)
            hgt_wts.append(stats["waiting_time"])

            # iGreedy
            ig_wt = igreedy(env, scenario)
            ig_wts.append(ig_wt)

            all_rows.append({
                "tier": tier,
                "hgt_waiting_time": stats["waiting_time"],
                "igreedy_waiting_time": ig_wt,
                "hgt_makespan": stats["makespan"],
            })

        mean_hgt = sum(hgt_wts) / len(hgt_wts)
        mean_ig = sum(ig_wts) / len(ig_wts)
        improvement = (mean_ig - mean_hgt) / (mean_ig + 1e-9) * 100.0
        results[tier] = {
            "hgt_wt": mean_hgt,
            "igreedy_wt": mean_ig,
            "improvement_pct": improvement,
        }
        print(
            f"[{tier:6s}]  iGreedy={mean_ig:.3f}  HGT={mean_hgt:.3f}  "
            f"improvement={improvement:+.1f}%"
        )

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["tier", "hgt_waiting_time", "igreedy_waiting_time", "hgt_makespan"]
        with output_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

    return results
