"""iEDF paper: churn-cost case study, EDF/iGreedy/Backpressure/LIFO vs the
restricted-information optimum W* (CP-SAT re-solve, same online information
boundary as the online policies). No HGT.

Usage:
  PYTHONPATH=. python evaluation/eval_churn_gap.py --n-scenarios 10
"""

from __future__ import annotations

import argparse

import yaml

from dynamic_scheduler.data.churn_scenarios import churn_batch
from dynamic_scheduler.environment.dynamic_intersection import DynamicIntersectionEnv
from dynamic_scheduler.evaluation.optimal_solver_online import restricted_optimal_episode
from evaluation.eval_dynamic import (
    backpressure_select,
    edf_select,
    igreedy_select,
    lifo_select,
    run_episode_with,
)
from intersection_scheduler.data.scenario_generator_4x4 import ZONE_POSITIONS

METHODS = {
    "igreedy": igreedy_select,
    "lifo": lifo_select,
    "backpressure": backpressure_select,
    "edf": edf_select,
}


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def run_one(env, arrivals, episode_duration, selector):
    env.reset(list(arrivals), episode_duration)
    wt, comp, n_seen = run_episode_with(env, arrivals, episode_duration, selector)
    per_vehicle = {e["vehicle_id"]: e["waiting_time"] for e in env.completed_log}
    for vid, wtv in zip([v.id for v in env.vehicles.values()], env.inflight_waiting_times()):
        per_vehicle[vid] = wtv
    return wt, comp, per_vehicle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default_dynamic.yaml")
    ap.add_argument("--n-scenarios", type=int, default=30)
    ap.add_argument("--n-base", type=int, default=3)
    ap.add_argument("--episode-duration", type=float, default=40.0)
    ap.add_argument("--n-gap-scenarios", type=int, default=10,
                     help="How many of the n_scenarios also get a CP-SAT W* solve (slow)")
    ap.add_argument("--gap-time-limit", type=float, default=5.0)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    env_cfg = cfg.get("environment", {})
    detection_window = env_cfg.get("detection_window", 10.0)
    commit_window = env_cfg.get("commit_window", 2.5)

    def make_env():
        return DynamicIntersectionEnv(
            detection_window=detection_window,
            commit_window=commit_window,
            zone_positions=ZONE_POSITIONS,
            penalty_coef=env_cfg.get("penalty_coef", 0.1),
            max_proximity_weight=env_cfg.get("max_proximity_weight", 2.0),
        )

    scenarios = churn_batch(args.n_scenarios, n_base=args.n_base)

    results = {m: {"wt": [], "disruptor_wt": [], "base_wt": []} for m in METHODS}
    for arrivals in scenarios:
        disruptor_vid = arrivals[-1].id
        base_vids = [v.id for v in arrivals[:-1]]
        for name, selector in METHODS.items():
            wt, comp, per_vehicle = run_one(make_env(), arrivals, args.episode_duration, selector)
            results[name]["wt"].append(wt)
            if disruptor_vid in per_vehicle:
                results[name]["disruptor_wt"].append(per_vehicle[disruptor_vid])
            base_vals = [per_vehicle[v] for v in base_vids if v in per_vehicle]
            if base_vals:
                results[name]["base_wt"].append(mean(base_vals))

    print(f"churn-cost scenario: {args.n_base} settled base + 1 late-detected disruptor, "
          f"n_scenarios={args.n_scenarios}\n")
    print(f"{'method':14s} {'overall_wt':>12s} {'disruptor_wt':>14s} {'base_avg_wt':>14s}")
    for name in METHODS:
        r = results[name]
        print(f"{name:14s} {mean(r['wt']):12.4f} {mean(r['disruptor_wt']):14.4f} {mean(r['base_wt']):14.4f}")

    wstars = []
    n_unsolved = 0
    for arrivals in scenarios[: args.n_gap_scenarios]:
        wstar, n_solved, n_fail = restricted_optimal_episode(
            list(arrivals), args.episode_duration, detection_window, commit_window,
            time_limit_seconds=args.gap_time_limit,
        )
        if wstar is not None:
            wstars.append(wstar)
        else:
            n_unsolved += 1

    print()
    if wstars:
        w_mean = mean(wstars)
        print(f"W* (restricted-info optimum, n={len(wstars)} solved, {n_unsolved} unsolved): {w_mean:.4f}")
        for name in METHODS:
            sub = results[name]["wt"][: args.n_gap_scenarios]
            m = mean(sub)
            if w_mean > 0.05:
                gap = (m - w_mean) / w_mean * 100
                print(f"  {name:14s} avg_wt={m:.4f}  gap_to_W*={gap:+.1f}%")
            else:
                print(f"  {name:14s} avg_wt={m:.4f}  abs_gap={m - w_mean:+.4f}")
    else:
        print("W*: none solved within time limit")


if __name__ == "__main__":
    main()
