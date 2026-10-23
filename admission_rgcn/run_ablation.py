"""Seed sweep and deferral ablation: train and evaluate N seeds with and
without the DEFER action, one output directory per (variant, seed).

Each run trains `--episodes` episodes, then evaluates checkpoint_last.pt on
the paper protocol (eval seeds 1-3 x 15 scenarios x 8 rates) against iGreedy,
LIFO and Backpressure. Finished runs (eval_final.csv present) are skipped and
interrupted runs resume from checkpoint_last.pt, so the script can be re-run.

Usage:
  PYTHONPATH=. python -m admission_rgcn.run_ablation --out runs/ablation \
      --seeds 0 1 2 3 4 5 6 7 8 9 --episodes 4000 --parallel 2 --workers 4
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

VARIANTS = ("defer", "nodefer")


def run_cmd(cmd: List[str], log_path: str) -> int:
    with open(log_path, "a") as log:
        return subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False).returncode


def run_one(variant: str, seed: int, args: argparse.Namespace) -> Tuple[str, int, bool]:
    out = os.path.join(args.out, f"{variant}_s{seed}")
    os.makedirs(out, exist_ok=True)
    final_csv = os.path.join(out, "eval_final.csv")
    last = os.path.join(out, "checkpoint_last.pt")
    if os.path.exists(final_csv):
        return variant, seed, True

    py = [sys.executable, "-m"]
    train = py + ["admission_rgcn.train", "--output", out, "--workers", str(args.workers)]
    if os.path.exists(last):
        train += ["--resume", last]
    else:
        train += ["--config", args.config, "--seed", str(seed), "--episodes", str(args.episodes)]
        if variant == "nodefer":
            train += ["--no-defer"]
    if run_cmd(train, os.path.join(out, "train.log")) != 0:
        return variant, seed, False

    evaluate = py + ["admission_rgcn.evaluate", "--checkpoint", last, "--seeds", "1", "2", "3",
                     "--n-scenarios", "15", "--baselines", "igreedy", "lifo", "backpressure",
                     "--workers", str(args.workers), "--output-csv", final_csv]
    ok = run_cmd(evaluate, os.path.join(out, "eval.log")) == 0
    return variant, seed, ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/ablation")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    ap.add_argument("--episodes", type=int, default=4000)
    ap.add_argument("--parallel", type=int, default=2, help="runs executed at the same time")
    ap.add_argument("--workers", type=int, default=4, help="rollout/eval processes per run")
    args = ap.parse_args()

    jobs = [(v, s) for s in args.seeds for v in args.variants]
    print(f"{len(jobs)} runs, {args.episodes} episodes each, {args.parallel} at a time, "
          f"{args.workers} workers per run -> {args.out}", flush=True)
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        for variant, seed, ok in pool.map(lambda j: run_one(*j, args), jobs):
            print(f"{variant} seed {seed}: {'done' if ok else 'FAILED (see train.log / eval.log)'}", flush=True)


if __name__ == "__main__":
    main()
