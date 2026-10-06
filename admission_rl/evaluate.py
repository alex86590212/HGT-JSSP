"""Evaluate a trained admission HGT against the gap-free admission baselines
on the same scenario stream as evaluation/eval_gapfree_admission.py
(TrafficGenerator(seed=seed*100+int(rate*1000)), scenarios drawn in order).

Usage:
  PYTHONPATH=. python -m admission_rl.evaluate \
      --checkpoint runs/admission_rl/checkpoint_best.pt \
      --output-csv runs/admission_rl/eval_final.csv [--wstar]
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List

import torch

from admission_rl.common import eval_scenarios, make_policy, worker_init
from admission_rl.rollout import policy_decide, run_rl_episode
from admission_rl.train import TIE_EPS, _episode_kwargs
from dynamic_scheduler.evaluation.gapfree_admission_solver import restricted_gapfree_optimal_episode
from evaluation.eval_gapfree_admission import METHODS

DEFAULT_BASELINES = ["igreedy", "edf", "backpressure", "lifo", "edf_prio_inherit", "edf_commitment_exchange"]


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def std(xs):
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5 if len(xs) > 1 else 0.0


def _task(checkpoint: str, rate: float, seed: int, n_scenarios: int, baselines: List[str],
          wstar: bool, wstar_time_limit: float) -> Dict[str, Any]:
    ckpt = torch.load(checkpoint, weights_only=False)  # noqa: S301
    cfg = ckpt["config"]
    policy = make_policy(cfg["model"])
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    env_cfg = cfg["env"]
    T, dw = env_cfg["episode_duration"], env_cfg["detection_window"]

    res: Dict[str, Any] = {"n_veh": [], "hgt": [], "hgt_comp": [], "n_decisions": 0, "n_defers": 0,
                           "w_star": [], **{m: [] for m in baselines}}
    for arrivals in eval_scenarios(rate, seed, n_scenarios, T):
        res["n_veh"].append(len(arrivals))
        _, stats = run_rl_episode(policy_decide(policy, deterministic=True), arrivals,
                                  **_episode_kwargs(env_cfg), record=False)
        res["hgt"].append(stats["waiting_time"])
        res["hgt_comp"].append(stats["completion_rate"])
        res["n_decisions"] += stats["n_decisions"]
        res["n_defers"] += stats["n_defers"]
        for name in baselines:
            res[name].append(METHODS[name](arrivals, T, dw)[0])
        if wstar:
            w, _, _ = restricted_gapfree_optimal_episode(list(arrivals), T, dw,
                                                         time_limit_seconds=wstar_time_limit)
            res["w_star"].append(w)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--rates", type=float, nargs="+",
                    default=[0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 1.00, 1.20])
    ap.add_argument("--n-scenarios", type=int, default=15)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--baselines", nargs="+", choices=list(METHODS), default=DEFAULT_BASELINES)
    ap.add_argument("--wstar", action="store_true", help="also solve restricted-info W* (CP-SAT, slow)")
    ap.add_argument("--wstar-time-limit", type=float, default=5.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--output-csv", default=None)
    args = ap.parse_args()
    baselines = list(dict.fromkeys(["igreedy", *args.baselines]))

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx, initializer=worker_init) as pool:
        jobs = {(rate, seed): pool.submit(_task, args.checkpoint, rate, seed, args.n_scenarios, baselines,
                                          args.wstar, args.wstar_time_limit)
                for rate in args.rates for seed in args.seeds}
        rows = []
        for rate in args.rates:
            per_seed = [jobs[(rate, seed)].result() for seed in args.seeds]
            methods = ["hgt", *baselines]
            row: Dict[str, Any] = {"rate": rate, "n_veh_avg": mean([n for r in per_seed for n in r["n_veh"]])}
            for name in methods:
                seed_means = [mean(r[name]) for r in per_seed]
                row[f"{name}_wt_mean"] = mean(seed_means)
                row[f"{name}_wt_std"] = std(seed_means)
            row["hgt_comp_mean"] = mean([c for r in per_seed for c in r["hgt_comp"]])
            hgt = [x for r in per_seed for x in r["hgt"]]
            ig = [x for r in per_seed for x in r["igreedy"]]
            row["hgt_vs_igreedy_pct"] = (sum(ig) - sum(hgt)) / sum(ig) * 100 if sum(ig) > 1e-9 else float("nan")
            row["hgt_wins"] = sum(1 for h, g in zip(hgt, ig) if h < g - TIE_EPS)
            row["hgt_losses"] = sum(1 for h, g in zip(hgt, ig) if h > g + TIE_EPS)
            row["hgt_ties"] = len(hgt) - row["hgt_wins"] - row["hgt_losses"]
            n_dec = sum(r["n_decisions"] for r in per_seed)
            row["hgt_defer_frac"] = sum(r["n_defers"] for r in per_seed) / n_dec if n_dec else 0.0
            if args.wstar:
                pairs = [(h, w) for r in per_seed for h, w in zip(r["hgt"], r["w_star"]) if w is not None]
                row["w_star"] = mean([w for _, w in pairs])
                row["n_wstar_solved"] = len(pairs)
                row["hgt_abs_gap_to_wstar"] = mean([max(0.0, h - w) for h, w in pairs])
            rows.append(row)
            print(f"rate={rate:.2f}  n_veh={row['n_veh_avg']:.1f}  "
                  + "  ".join(f"{m}={row[f'{m}_wt_mean']:.4f}" for m in methods)
                  + f"  HGT vs iG={row['hgt_vs_igreedy_pct']:+.2f}%  "
                  f"W/T/L={row['hgt_wins']}/{row['hgt_ties']}/{row['hgt_losses']}  "
                  f"defer={row['hgt_defer_frac']:.3f}"
                  + (f"  W*={row['w_star']:.4f}" if args.wstar else ""), flush=True)

    overall_hgt = mean([r["hgt_wt_mean"] for r in rows])
    overall_ig = mean([r["igreedy_wt_mean"] for r in rows])
    print(f"\nMean over rates: HGT={overall_hgt:.4f}  iGreedy={overall_ig:.4f}  "
          f"({(overall_ig - overall_hgt) / overall_ig * 100:+.2f}%)  "
          f"worst rate {min(r['hgt_vs_igreedy_pct'] for r in rows):+.2f}%")

    if args.output_csv:
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {args.output_csv}")


if __name__ == "__main__":
    main()
