"""Evaluate a trained HGT and a trained R-GCN admission policy side by side
against the gap-free baselines (and optionally W*) on identical scenarios.

Usage:
  PYTHONPATH=. python compare_eval.py \
      --hgt runs/admission_rl_v2/checkpoint_best.pt \
      --rgcn runs/admission_rgcn_v1/checkpoint_best.pt \
      --output-csv runs/compare_eval.csv [--wstar]
"""

from __future__ import annotations

import argparse
import csv
import importlib
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List

import torch

from dynamic_scheduler.evaluation.gapfree_admission_solver import restricted_gapfree_optimal_episode
from evaluation.eval_gapfree_admission import METHODS

DEFAULT_BASELINES = ["igreedy", "edf", "backpressure", "lifo", "edf_prio_inherit", "edf_commitment_exchange"]
TIE_EPS = 1e-6


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _load(pkg: str, checkpoint: str):
    common = importlib.import_module(f"{pkg}.common")
    rollout = importlib.import_module(f"{pkg}.rollout")
    train = importlib.import_module(f"{pkg}.train")
    ckpt = torch.load(checkpoint, weights_only=False)  # noqa: S301
    policy = common.make_policy(ckpt["config"]["model"])
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    return policy, ckpt["config"]["env"], rollout, train, common


def _task(models: Dict[str, Any], rate: float, seed: int, n_scenarios: int, baselines: List[str],
          wstar: bool, wstar_time_limit: float) -> Dict[str, Any]:
    loaded = {name: _load(pkg, ckpt) for name, (pkg, ckpt) in models.items()}
    env_cfg = next(iter(loaded.values()))[1]
    T, dw = env_cfg["episode_duration"], env_cfg["detection_window"]
    eval_scenarios = next(iter(loaded.values()))[4].eval_scenarios

    res: Dict[str, Any] = {"n_veh": [], "w_star": [], **{m: [] for m in [*models, *baselines]},
                           **{f"{n}_dec": 0 for n in models}, **{f"{n}_def": 0 for n in models}}
    for arrivals in eval_scenarios(rate, seed, n_scenarios, T):
        res["n_veh"].append(len(arrivals))
        for name, (policy, cfg, rollout, train, _) in loaded.items():
            _, stats = rollout.run_rl_episode(rollout.policy_decide(policy, deterministic=True), arrivals,
                                              **train._episode_kwargs(cfg), record=False)
            res[name].append(stats["waiting_time"])
            res[f"{name}_dec"] += stats["n_decisions"]
            res[f"{name}_def"] += stats["n_defers"]
        for b in baselines:
            res[b].append(METHODS[b](arrivals, T, dw)[0])
        if wstar:
            w, _, _ = restricted_gapfree_optimal_episode(list(arrivals), T, dw, time_limit_seconds=wstar_time_limit)
            res["w_star"].append(w)
    return res


def _worker_init() -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hgt", required=True)
    ap.add_argument("--rgcn", required=True)
    ap.add_argument("--rates", type=float, nargs="+", default=[0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 1.00, 1.20])
    ap.add_argument("--n-scenarios", type=int, default=15)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--baselines", nargs="+", choices=list(METHODS), default=DEFAULT_BASELINES)
    ap.add_argument("--wstar", action="store_true")
    ap.add_argument("--wstar-time-limit", type=float, default=5.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--output-csv", default=None)
    args = ap.parse_args()
    baselines = list(dict.fromkeys(["igreedy", *args.baselines]))
    models = {"hgt": ("admission_rl", args.hgt), "rgcn": ("admission_rgcn", args.rgcn)}
    names = [*models, *baselines]

    ctx = mp.get_context("spawn")
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx, initializer=_worker_init) as pool:
        jobs = {(r, s): pool.submit(_task, models, r, s, args.n_scenarios, baselines, args.wstar,
                                    args.wstar_time_limit)
                for r in args.rates for s in args.seeds}
        for rate in args.rates:
            per_seed = [jobs[(rate, s)].result() for s in args.seeds]
            row: Dict[str, Any] = {"rate": rate, "n_veh_avg": mean([n for r in per_seed for n in r["n_veh"]])}
            for n in names:
                row[f"{n}_wt_mean"] = mean([mean(r[n]) for r in per_seed])
            ig = [x for r in per_seed for x in r["igreedy"]]
            for n in models:
                v = [x for r in per_seed for x in r[n]]
                row[f"{n}_vs_igreedy_pct"] = (sum(ig) - sum(v)) / sum(ig) * 100 if sum(ig) > 1e-9 else float("nan")
                row[f"{n}_wins"] = sum(1 for a, g in zip(v, ig) if a < g - TIE_EPS)
                row[f"{n}_losses"] = sum(1 for a, g in zip(v, ig) if a > g + TIE_EPS)
                row[f"{n}_ties"] = len(v) - row[f"{n}_wins"] - row[f"{n}_losses"]
                nd = sum(r[f"{n}_dec"] for r in per_seed)
                row[f"{n}_defer_frac"] = sum(r[f"{n}_def"] for r in per_seed) / nd if nd else 0.0
            if args.wstar:
                ws = [w for r in per_seed for w in r["w_star"] if w is not None]
                row["w_star"] = mean(ws)
            rows.append(row)
            print(f"rate={rate:.2f}  n_veh={row['n_veh_avg']:.1f}  "
                  + "  ".join(f"{n}={row[f'{n}_wt_mean']:.4f}" for n in names)
                  + "  " + "  ".join(f"{n} vs iG={row[f'{n}_vs_igreedy_pct']:+.1f}% "
                                     f"W/T/L={row[f'{n}_wins']}/{row[f'{n}_ties']}/{row[f'{n}_losses']}"
                                     for n in models)
                  + (f"  W*={row['w_star']:.4f}" if args.wstar else ""), flush=True)

    print("\nMean over rates: " + "  ".join(f"{n}={mean([r[f'{n}_wt_mean'] for r in rows]):.4f}" for n in names))
    if args.output_csv:
        with open(args.output_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"Saved {args.output_csv}")


if __name__ == "__main__":
    main()
