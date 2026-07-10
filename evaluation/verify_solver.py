"""Verify the OR-Tools solver matches the environment's true achievable optimum.

Brute-forces every legal scheduling order the env can produce (DFS over the
feasible set at each step, mirroring exactly how the env/policy schedules) and
compares the minimum achievable waiting time to solve_optimal's W*.

If env_best == cpsat for every scenario, the CP-SAT model captures the env's
achievable set correctly. If cpsat < env_best, the solver is under-constrained
(finding schedules the env can never reach). If cpsat > env_best, the solver is
over-constrained or has a rounding/objective bug.
"""

from __future__ import annotations

import argparse
import copy

from intersection_scheduler.data.scenario_generator import ScenarioGenerator
from intersection_scheduler.environment.feasibility import (
    compute_feasible_set,
    next_feasible_time,
)
from intersection_scheduler.environment.intersection import IntersectionEnv
from intersection_scheduler.utils.metrics import episode_waiting_time

from evaluation.optimal_solver import solve_optimal


def brute_force_env_optimum(scenario, node_budget: int = 200_000) -> float:
    """Return the minimum waiting time reachable by any legal env schedule.

    DFS over the feasible set at every decision point. node_budget caps the
    search so pathological branching can't hang; returns the best found so far.
    """
    best = [float("inf")]
    nodes = [0]

    def dfs(env: IntersectionEnv) -> None:
        if nodes[0] > node_budget:
            return
        nodes[0] += 1

        if all(o.scheduled for o in env.operations):
            best[0] = min(best[0], episode_waiting_time(env))
            return

        mask = compute_feasible_set(env)
        if not mask.any():
            nt = next_feasible_time(env)
            if nt is None:
                return
            env.current_time = nt
            dfs(env)
            return

        for idx in range(len(env.operations)):
            if mask[idx].item():
                child = copy.deepcopy(env)
                child.step(idx)
                dfs(child)

    root = IntersectionEnv()
    root.reset(scenario.vehicles)
    dfs(root)
    return best[0]


def main():
    parser = argparse.ArgumentParser(description="Verify solver vs env brute-force optimum")
    parser.add_argument("--tiers", nargs="+", default=["easy", "medium"])
    parser.add_argument("--n_scenarios", type=int, default=20)
    parser.add_argument("--time_limit", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tol", type=float, default=1e-3,
                        help="Absolute tolerance (seconds) for env_best == cpsat")
    args = parser.parse_args()

    for tier in args.tiers:
        gen = ScenarioGenerator(seed=args.seed)
        mismatches = 0
        under = 0  # cpsat < env_best (solver too permissive)
        over = 0   # cpsat > env_best (solver too restrictive / bug)
        checked = 0

        print(f"\n=== {tier} ===", flush=True)
        for i in range(args.n_scenarios):
            scenario = getattr(gen, tier)()
            env_best = brute_force_env_optimum(scenario)
            w_star = solve_optimal(
                scenario.vehicles, scenario.manoeuvres, time_limit_seconds=args.time_limit
            )
            if w_star is None or env_best == float("inf"):
                print(f"  scenario {i}: SKIP (w_star={w_star}, env_best={env_best})", flush=True)
                continue

            checked += 1
            diff = w_star - env_best
            if abs(diff) > args.tol:
                mismatches += 1
                if diff < 0:
                    under += 1
                    verdict = "SOLVER TOO PERMISSIVE (cpsat < env_best)"
                else:
                    over += 1
                    verdict = "SOLVER TOO RESTRICTIVE (cpsat > env_best)"
                print(
                    f"  scenario {i}: env_best={env_best:.4f}  cpsat={w_star:.4f}  "
                    f"diff={diff:+.4f}  <-- {verdict}",
                    flush=True,
                )

        print(
            f"{tier}: {checked} checked, {mismatches} mismatches "
            f"({under} solver-too-permissive, {over} solver-too-restrictive)",
            flush=True,
        )


if __name__ == "__main__":
    main()
