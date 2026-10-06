"""Paper-scale comparison for the gap-free full-route admission model
(dynamic_scheduler.environment.gapfree_admission), with restricted-info W*
(dynamic_scheduler.evaluation.gapfree_admission_solver), mirroring
evaluation/eval_scaling_baselines.py's structure and conventions for the
blocking model.

The three commitment-exchange variants share proposal generation and run
until no profitable proposal remains; only profitable-exchange ordering differs.
Diagnostics report per-episode means, including proposal evaluations,
accepted exchanges, planned gain, and runtime in seconds.
Evaluated counts all examined targets; proposed counts feasible advancements
before the gain filter; profitable counts alternatives passing the full-set
gain filter; accepted counts applied exchanges. Realized evaluations can
differ after the orderings' trajectories diverge.

Usage:
  PYTHONPATH=. python evaluation/eval_gapfree_admission.py \
      --rates 0.10 0.20 0.35 0.50 0.65 0.80 1.00 1.20 \
      --n-scenarios 15 --seeds 0 1 2 --optimality-gap \
      --output-csv gapfree_admission_results.csv

  PYTHONPATH=. python evaluation/eval_gapfree_admission.py \
      --methods edf_commitment_exchange arrival_commitment_exchange gain_commitment_exchange \
      --rates 0.5 1.0 --n-scenarios 5 --seeds 0
"""

from __future__ import annotations

import argparse
import csv
import time

import functools

from dynamic_scheduler.data.traffic_generator import TrafficGenerator
from dynamic_scheduler.environment.gapfree_admission import (
    backpressure_admission_select,
    edf_admission_select,
    igreedy_admission_select,
    lifo_admission_select,
    run_admission_episode,
    run_admission_episode_commitment_exchange,
    run_admission_episode_priority_inherit,
)
from dynamic_scheduler.evaluation.gapfree_admission_solver import restricted_gapfree_optimal_episode

# name -> runner(arrivals, episode_duration, detection_window) -> (wt, comp, n).
EXCHANGE_METHODS = {
    "edf_commitment_exchange": "deadline",
    "arrival_commitment_exchange": "arrival",
    "gain_commitment_exchange": "gain",
}
EXCHANGE_METRICS = (
    "proposed", "evaluated", "profitable", "accepted", "planned_gain",
)
METHODS = {
    "igreedy": functools.partial(run_admission_episode, selector=igreedy_admission_select),
    "lifo": functools.partial(run_admission_episode, selector=lifo_admission_select),
    "backpressure": functools.partial(run_admission_episode, selector=backpressure_admission_select),
    "edf": functools.partial(run_admission_episode, selector=edf_admission_select),
    "edf_prio_inherit": run_admission_episode_priority_inherit,
    **{
        name: functools.partial(run_admission_episode_commitment_exchange, order=order)
        for name, order in EXCHANGE_METHODS.items()
    },
}


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def std(xs):
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5 if len(xs) > 1 else 0.0


def _run_methods_for_rate(rate, seeds, n_scenarios, episode_duration, detection_window,
                          methods=None):
    methods = METHODS if methods is None else methods
    per_seed_wt = {m: [] for m in methods}
    per_seed_comp = {m: [] for m in methods}
    diagnostics = {m: {"runtime_seconds": []} for m in methods}
    for name in methods.keys() & EXCHANGE_METHODS.keys():
        diagnostics[name].update({metric: [] for metric in EXCHANGE_METRICS})
    n_veh_all = []
    first_seed_scenarios = None
    first_seed_wt = None
    for seed in seeds:
        gen = TrafficGenerator(seed=seed * 100 + int(rate * 1000))
        scenarios = [
            gen.generate_episode_arrivals(episode_duration, arrival_rate=rate)
            for _ in range(n_scenarios)
        ]
        n_veh_all.extend(len(s) for s in scenarios)
        sums = {m: [] for m in methods}
        comps = {m: [] for m in methods}
        for arrivals in scenarios:
            for name, runner in methods.items():
                stats = {}
                kwargs = {"stats": stats} if name in EXCHANGE_METHODS else {}
                started = time.perf_counter()
                wt, comp, _ = runner(arrivals, episode_duration, detection_window, **kwargs)
                diagnostics[name]["runtime_seconds"].append(time.perf_counter() - started)
                for metric in EXCHANGE_METRICS if name in EXCHANGE_METHODS else ():
                    diagnostics[name][metric].append(stats[metric])
                sums[name].append(wt)
                comps[name].append(comp)
        for name in methods:
            per_seed_wt[name].append(mean(sums[name]))
            per_seed_comp[name].append(mean(comps[name]))
        if first_seed_scenarios is None:
            first_seed_scenarios = scenarios
            first_seed_wt = sums

    row = {"rate": rate, "n_veh_avg": mean(n_veh_all)}
    for name in methods:
        row[f"{name}_wt_mean"] = mean(per_seed_wt[name])
        row[f"{name}_wt_std"] = std(per_seed_wt[name])
        row[f"{name}_comp_mean"] = mean(per_seed_comp[name])
        for metric, values in diagnostics[name].items():
            row[f"{name}_{metric}_mean"] = mean(values)
    for key, candidate, baseline in (
        ("edf_vs_igreedy_pct", "edf", "igreedy"),
        ("edf_vs_bp_pct", "edf", "backpressure"),
        ("edf_prio_inherit_vs_igreedy_pct", "edf_prio_inherit", "igreedy"),
    ):
        base = row.get(f"{baseline}_wt_mean", float("nan"))
        value = row.get(f"{candidate}_wt_mean", float("nan"))
        row[key] = (base - value) / base * 100 if base > 0.01 else float("nan")
    return row, first_seed_scenarios, first_seed_wt


_PCT_GAP_WSTAR_FLOOR = 0.05


def _wstar_gap_for_rate(row, scenarios, wt_by_method, episode_duration, detection_window, time_limit,
                       methods=None):
    methods = METHODS if methods is None else methods
    abs_gaps = {name: [] for name in methods}
    pct_gaps = {name: [] for name in methods}
    n_unsolved = 0
    wstars = []
    for i, arrivals in enumerate(scenarios):
        wstar, _, _ = restricted_gapfree_optimal_episode(
            list(arrivals), episode_duration, detection_window, time_limit_seconds=time_limit,
        )
        if wstar is None:
            n_unsolved += 1
            continue
        wstars.append(wstar)
        for name in methods:
            abs_g = max(0.0, wt_by_method[name][i] - wstar)
            abs_gaps[name].append(abs_g)
            if wstar >= _PCT_GAP_WSTAR_FLOOR:
                pct_gaps[name].append(abs_g / wstar * 100.0)

    row["n_wstar_solved"] = len(wstars)
    row["n_wstar_unsolved"] = n_unsolved
    row["w_star"] = mean(wstars) if wstars else float("nan")
    for name in methods:
        key = f"{name}_gap_to_wstar_pct"
        row[key] = mean(pct_gaps[name]) if pct_gaps[name] else mean(abs_gaps[name])
        row[f"{name}_abs_gap_to_wstar"] = mean(abs_gaps[name])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", type=float, nargs="+",
                     default=[0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 1.00, 1.20])
    ap.add_argument("--n-scenarios", type=int, default=15)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--episode-duration", type=float, default=60.0)
    ap.add_argument("--detection-window", type=float, default=10.0)
    ap.add_argument("--output-csv", default=None)
    ap.add_argument("--optimality-gap", action="store_true")
    ap.add_argument("--gap-time-limit", type=float, default=5.0)
    ap.add_argument("--methods", nargs="+", choices=list(METHODS), default=list(METHODS))
    args = ap.parse_args()
    methods = {name: METHODS[name] for name in args.methods}

    rows = []
    header = ["rate", "n_veh_avg"]
    for name in methods:
        header += [f"{name}_wt_mean", f"{name}_wt_std", f"{name}_comp_mean", f"{name}_runtime_seconds_mean"]
        if name in EXCHANGE_METHODS:
            header += [f"{name}_{metric}_mean" for metric in EXCHANGE_METRICS]
    header += ["edf_vs_igreedy_pct", "edf_vs_bp_pct", "edf_prio_inherit_vs_igreedy_pct"]
    if args.optimality_gap:
        header += ["w_star", "n_wstar_solved", "n_wstar_unsolved"]
        header += [f"{name}_gap_to_wstar_pct" for name in methods]
        header += [f"{name}_abs_gap_to_wstar" for name in methods]

    for rate in args.rates:
        row, first_seed_scenarios, first_seed_wt = _run_methods_for_rate(
            rate, args.seeds, args.n_scenarios, args.episode_duration, args.detection_window,
            methods,
        )
        print(f"rate={rate:.2f}  n_veh={row['n_veh_avg']:.1f}  "
              + "  ".join(f"{name}={row[f'{name}_wt_mean']:.4f}±{row[f'{name}_wt_std']:.4f}"
                          for name in methods), flush=True)
        for name in methods:
            if name in EXCHANGE_METHODS:
                print(f"    {name}: " + "  ".join(
                    f"{metric}={row[f'{name}_{metric}_mean']:.3f}"
                    for metric in (*EXCHANGE_METRICS, "runtime_seconds")
                ), flush=True)

        if args.optimality_gap:
            _wstar_gap_for_rate(
                row, first_seed_scenarios, first_seed_wt,
                args.episode_duration, args.detection_window, args.gap_time_limit,
                methods,
            )
            seed0 = {name: mean(first_seed_wt[name]) for name in methods}
            print(f"    [seed0-only, n={len(first_seed_scenarios)}]  "
                  + "  ".join(f"{name}={seed0[name]:.4f}" for name in methods)
                  + f"  W*={row['w_star']:.4f} (n_solved={row['n_wstar_solved']}, n_unsolved={row['n_wstar_unsolved']})")
            print("    " + "  ".join(
                f"{name}_gap={row[f'{name}_gap_to_wstar_pct']:+.1f}%" for name in methods
            ))

        rows.append(row)

    if args.output_csv:
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved {args.output_csv}")


if __name__ == "__main__":
    main()
