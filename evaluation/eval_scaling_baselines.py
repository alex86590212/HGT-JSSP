"""iEDF paper: EDF vs iGreedy/Backpressure/LIFO across arrival-rate, no HGT.

Sweeps arrival_rate over a finer grid than the fixed easy/medium/hard tiers
(0.20/0.35/1.00 veh/s) to show the EDF/iGreedy/Backpressure gap holds across
the full realistic operating range, not just at three tier boundaries.
Multiple seeds per rate for error bars. Same run_episode_with loop and
selectors as eval_dynamic.py -- only the traffic-generation axis differs.

Usage:
  PYTHONPATH=. python evaluation/eval_scaling_baselines.py \
      --rates 0.10 0.20 0.35 0.50 0.65 0.80 1.00 1.20 \
      --n-scenarios 15 --seeds 0 1 2 --output-csv scaling_results.csv
"""

from __future__ import annotations

import argparse
import csv

import yaml

from dynamic_scheduler.data.traffic_generator import TrafficGenerator
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


def std(xs):
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5 if len(xs) > 1 else 0.0


def _run_methods_for_rate(rate, seeds, n_scenarios, episode_duration, make_env):
    per_seed_wt = {m: [] for m in METHODS}
    per_seed_comp = {m: [] for m in METHODS}
    n_veh_all = []
    first_seed_scenarios = None
    first_seed_wt = None  # per-scenario wt for seeds[0] only, aligned with W* scenarios
    for seed in seeds:
        gen = TrafficGenerator(seed=seed * 100 + int(rate * 1000))
        scenarios = [
            gen.generate_episode_arrivals(episode_duration, arrival_rate=rate)
            for _ in range(n_scenarios)
        ]
        n_veh_all.extend(len(s) for s in scenarios)
        sums = {m: [] for m in METHODS}
        comps = {m: [] for m in METHODS}
        for arrivals in scenarios:
            for name, selector in METHODS.items():
                env = make_env()
                env.reset(list(arrivals), episode_duration)
                wt, comp, _ = run_episode_with(env, arrivals, episode_duration, selector)
                sums[name].append(wt)
                comps[name].append(comp)
        for name in METHODS:
            per_seed_wt[name].append(mean(sums[name]))
            per_seed_comp[name].append(mean(comps[name]))
        if first_seed_scenarios is None:
            first_seed_scenarios = scenarios
            first_seed_wt = sums

    row = {"rate": rate, "n_veh_avg": mean(n_veh_all)}
    for name in METHODS:
        row[f"{name}_wt_mean"] = mean(per_seed_wt[name])
        row[f"{name}_wt_std"] = std(per_seed_wt[name])
        row[f"{name}_comp_mean"] = mean(per_seed_comp[name])
    ig, ed, bp = row["igreedy_wt_mean"], row["edf_wt_mean"], row["backpressure_wt_mean"]
    row["edf_vs_igreedy_pct"] = (ig - ed) / ig * 100 if ig > 0.01 else float("nan")
    row["edf_vs_bp_pct"] = (bp - ed) / bp * 100 if bp > 0.01 else float("nan")
    return row, first_seed_scenarios, first_seed_wt


# Same convention as eval_dynamic.py's --optimality-gap: gap is computed
# PER SCENARIO (wt - w_star, using that scenario's own W*), then averaged --
# never pooling a multi-seed wt mean against a single-seed W* mean, which
# can produce a mathematically-impossible negative gap purely from sample
# mismatch (W* is a true per-scenario floor, but two different samples'
# means can cross even when every individual gap is >= 0).
_PCT_GAP_WSTAR_FLOOR = 0.05


def _wstar_gap_for_rate(row, scenarios, wt_by_method, episode_duration,
                         detection_window, commit_window, time_limit):
    abs_gaps = {name: [] for name in METHODS}
    pct_gaps = {name: [] for name in METHODS}
    n_unsolved = 0
    wstars = []
    for i, arrivals in enumerate(scenarios):
        wstar, _, _ = restricted_optimal_episode(
            list(arrivals), episode_duration, detection_window, commit_window,
            time_limit_seconds=time_limit,
        )
        if wstar is None:
            n_unsolved += 1
            continue
        wstars.append(wstar)
        for name in METHODS:
            # Clip at 0: W* and the online methods are solved by different
            # numerical paths (CP-SAT vs. the event-driven env loop), so a
            # true-zero gap can show as a ~1e-6 negative from solver/float
            # tolerance alone. W* is a hard floor -- a negative gap is never
            # a real result, only noise below both solvers' precision.
            abs_g = max(0.0, wt_by_method[name][i] - wstar)
            abs_gaps[name].append(abs_g)
            if wstar >= _PCT_GAP_WSTAR_FLOOR:
                pct_gaps[name].append(abs_g / wstar * 100.0)

    row["n_wstar_solved"] = len(wstars)
    row["n_wstar_unsolved"] = n_unsolved
    row["w_star"] = mean(wstars) if wstars else float("nan")
    for name, key in (("edf", "edf_gap_to_wstar_pct"),
                       ("igreedy", "igreedy_gap_to_wstar_pct"),
                       ("backpressure", "bp_gap_to_wstar_pct")):
        row[key] = mean(pct_gaps[name]) if pct_gaps[name] else mean(abs_gaps[name])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default_dynamic.yaml")
    ap.add_argument("--rates", type=float, nargs="+",
                     default=[0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 1.00, 1.20])
    ap.add_argument("--n-scenarios", type=int, default=15,
                     help="Scenarios per seed per rate")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--output-csv", default=None)
    ap.add_argument("--optimality-gap", action="store_true",
                     help="Also compute restricted-information W* per rate (CP-SAT "
                          "re-solve at every detection event, one seed's scenarios only)")
    ap.add_argument("--gap-time-limit", type=float, default=5.0)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    env_cfg = cfg.get("environment", {})
    episode_duration = env_cfg.get("episode_duration", 60.0)
    detection_window = env_cfg.get("detection_window", 10.0)
    commit_window = env_cfg.get("commit_window", 2.5)

    def make_env():
        return DynamicIntersectionEnv(
            detection_window=detection_window,
            commit_window=commit_window,
            zone_positions=ZONE_POSITIONS,
        )

    rows = []
    header = ["rate", "n_veh_avg"]
    for name in METHODS:
        header += [f"{name}_wt_mean", f"{name}_wt_std", f"{name}_comp_mean"]
    header += ["edf_vs_igreedy_pct", "edf_vs_bp_pct"]
    if args.optimality_gap:
        header += ["w_star", "n_wstar_solved", "n_wstar_unsolved",
                    "edf_gap_to_wstar_pct", "igreedy_gap_to_wstar_pct", "bp_gap_to_wstar_pct"]

    for rate in args.rates:
        row, first_seed_scenarios, first_seed_wt = _run_methods_for_rate(
            rate, args.seeds, args.n_scenarios, episode_duration, make_env,
        )
        ig, ed, bp = row["igreedy_wt_mean"], row["edf_wt_mean"], row["backpressure_wt_mean"]

        print(f"rate={rate:.2f}  n_veh={row['n_veh_avg']:.1f}  "
              f"igreedy={ig:.4f}±{row['igreedy_wt_std']:.4f}  "
              f"bp={bp:.4f}±{row['backpressure_wt_std']:.4f}  "
              f"edf={ed:.4f}±{row['edf_wt_std']:.4f}  "
              f"vs_ig={row['edf_vs_igreedy_pct']:+.1f}%  vs_bp={row['edf_vs_bp_pct']:+.1f}%")

        if args.optimality_gap:
            _wstar_gap_for_rate(
                row, first_seed_scenarios, first_seed_wt, episode_duration,
                detection_window, commit_window, args.gap_time_limit,
            )
            # Seed-0-only means, i.e. the SAME sample W* was solved on --
            # printed here (not the pooled 3-seed means above) so nothing
            # side-by-side on this line is ever a sample mismatch.
            seed0_ed = mean(first_seed_wt["edf"])
            seed0_ig = mean(first_seed_wt["igreedy"])
            seed0_bp = mean(first_seed_wt["backpressure"])
            print(f"    [seed0-only, n={len(first_seed_scenarios)}]  "
                  f"igreedy={seed0_ig:.4f}  bp={seed0_bp:.4f}  edf={seed0_ed:.4f}  "
                  f"W*={row['w_star']:.4f} (n_solved={row['n_wstar_solved']}, "
                  f"n_unsolved={row['n_wstar_unsolved']})")
            print(f"    edf_gap={row['edf_gap_to_wstar_pct']:+.1f}%  "
                  f"igreedy_gap={row['igreedy_gap_to_wstar_pct']:+.1f}%  "
                  f"bp_gap={row['bp_gap_to_wstar_pct']:+.1f}%")

        rows.append(row)

    if args.output_csv:
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved {args.output_csv}")


if __name__ == "__main__":
    main()
