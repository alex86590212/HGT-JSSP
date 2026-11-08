"""Aggregate runs/ablation/<variant>_s<seed>/eval_final.csv across seeds.

Reports, per arrival rate and for the rate average, the mean +- std over
training seeds of the R-GCN mean waiting time with and without DEFER, next to
the iGreedy / Backpressure / LIFO references (identical in every run).

Usage:
  PYTHONPATH=. python -m admission_rgcn.summarize_ablation --dir runs/ablation \
      --output-csv runs/ablation/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import statistics
from typing import Dict, List

VARIANTS = ("defer", "nodefer")
REFS = ("igreedy", "backpressure", "lifo")


def load(dir_: str, variant: str) -> Dict[int, List[dict]]:
    runs = {}
    for path in sorted(glob.glob(os.path.join(dir_, f"{variant}_s*", "eval_final.csv"))):
        seed = int(os.path.basename(os.path.dirname(path)).rsplit("_s", 1)[1])
        with open(path) as f:
            runs[seed] = list(csv.DictReader(f))
    return runs


def mean_std(xs: List[float]):
    return statistics.fmean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/ablation")
    ap.add_argument("--output-csv", default=None)
    args = ap.parse_args()

    runs = {v: load(args.dir, v) for v in VARIANTS}
    if not any(runs.values()):
        raise SystemExit(f"no eval_final.csv found under {args.dir}")
    ref_rows = next(r for r in runs.values() if r)
    ref_rows = next(iter(ref_rows.values()))
    rates = [float(r["rate"]) for r in ref_rows]

    per_seed_avg = {v: {s: statistics.fmean(float(r["hgt_wt_mean"]) for r in rows) for s, rows in runs[v].items()}
                    for v in VARIANTS}
    ig_avg = statistics.fmean(float(r["igreedy_wt_mean"]) for r in ref_rows)

    out_rows = []
    for i, rate in enumerate(rates):
        row = {"rate": rate}
        for ref in REFS:
            row[ref] = float(ref_rows[i][f"{ref}_wt_mean"])
        for v in VARIANTS:
            vals = [float(rows[i]["hgt_wt_mean"]) for rows in runs[v].values()]
            row[f"{v}_mean"], row[f"{v}_std"] = mean_std(vals) if vals else (float("nan"), float("nan"))
            defer = [float(rows[i]["hgt_defer_frac"]) for rows in runs[v].values()]
            row[f"{v}_defer_frac"] = statistics.fmean(defer) if defer else float("nan")
        out_rows.append(row)

    print(f"seeds finished: defer={len(runs['defer'])}  nodefer={len(runs['nodefer'])}\n")
    print(f"{'rate':>5} {'iGreedy':>8} {'B.P.':>8} {'LIFO':>8} {'R-GCN (defer)':>18} {'R-GCN (no defer)':>18} {'defer frac':>10}")
    for r in out_rows:
        print(f"{r['rate']:>5.2f} {r['igreedy']:>8.3f} {r['backpressure']:>8.3f} {r['lifo']:>8.3f} "
              f"{r['defer_mean']:>10.3f} ±{r['defer_std']:.3f} {r['nodefer_mean']:>10.3f} ±{r['nodefer_std']:.3f} "
              f"{r['defer_defer_frac']:>10.2f}")
    print()
    for v in VARIANTS:
        avgs = list(per_seed_avg[v].values())
        if not avgs:
            continue
        m, sd = mean_std(avgs)
        beats = sum(a < ig_avg for a in avgs)
        print(f"{v:>8}: rate-averaged wait {m:.4f} ± {sd:.4f}  (min {min(avgs):.4f}, max {max(avgs):.4f}); "
              f"{beats}/{len(avgs)} seeds below iGreedy ({ig_avg:.4f}); "
              f"reduction vs iGreedy {(ig_avg - m) / ig_avg * 100:+.1f}%")

    if args.output_csv:
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(out_rows[0]))
            writer.writeheader()
            writer.writerows(out_rows)
        print(f"\nSaved {args.output_csv}")


if __name__ == "__main__":
    main()
