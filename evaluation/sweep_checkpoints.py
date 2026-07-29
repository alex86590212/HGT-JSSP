"""Run eval_dynamic.py --optimality-gap over every checkpoint in a training
run directory, and print one summary table across all of them.

Exists because checkpoint_best.pt tracks the single lowest eval/waiting_time
sample, which we've measured to swing 2x+ episode-to-episode from arrival
noise alone -- picking "best" off one number invites exactly the confusion
this script is meant to resolve. Sweeping every numbered checkpoint with the
same frozen scenarios shows the actual trend across training instead.

Usage:
    PYTHONPATH=. python evaluation/sweep_checkpoints.py \
        --checkpoint-dir results_dynamic_reward_fix \
        --n-scenarios 15 --gap-time-limit 5.0
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path


def find_checkpoints(checkpoint_dir: Path) -> list[tuple[str, Path]]:
    numbered = []
    for p in checkpoint_dir.glob("checkpoint_*.pt"):
        m = re.match(r"checkpoint_(\d+)\.pt$", p.name)
        if m:
            numbered.append((int(m.group(1)), p))
    numbered.sort()
    result = [(str(ep), p) for ep, p in numbered]
    best = checkpoint_dir / "checkpoint_best.pt"
    if best.exists():
        result.append(("best", best))
    return result


def summarize_csv(csv_path: Path) -> dict:
    rows = list(csv.DictReader(csv_path.open()))
    tiers = ["easy", "medium", "hard"]
    out = {}
    for tier in tiers:
        trows = [r for r in rows if r["tier"] == tier]
        if not trows:
            continue
        n = len(trows)
        hgt_wt = sum(float(r["hgt_waiting_time"]) for r in trows) / n
        edf_wt = sum(float(r["edf_waiting_time"]) for r in trows) / n
        bp_wt = sum(float(r["backpressure_waiting_time"]) for r in trows) / n
        gaps = [float(r["hgt_gap_pct"]) for r in trows
                if r["w_star_solved"] == "True" and r["hgt_gap_pct"] != ""]
        gap_pct = sum(gaps) / len(gaps) if gaps else float("nan")
        vs_edf = (edf_wt - hgt_wt) / edf_wt * 100 if edf_wt else float("nan")
        vs_bp = (bp_wt - hgt_wt) / bp_wt * 100 if bp_wt else float("nan")
        out[tier] = dict(hgt_wt=hgt_wt, vs_edf=vs_edf, vs_bp=vs_bp, gap_pct=gap_pct, n_solved=len(gaps), n=n)
    all_rows = rows
    n = len(all_rows)
    if n:
        hgt_wt = sum(float(r["hgt_waiting_time"]) for r in all_rows) / n
        edf_wt = sum(float(r["edf_waiting_time"]) for r in all_rows) / n
        bp_wt = sum(float(r["backpressure_waiting_time"]) for r in all_rows) / n
        vs_edf = (edf_wt - hgt_wt) / edf_wt * 100 if edf_wt else float("nan")
        vs_bp = (bp_wt - hgt_wt) / bp_wt * 100 if bp_wt else float("nan")
        out["overall"] = dict(hgt_wt=hgt_wt, vs_edf=vs_edf, vs_bp=vs_bp)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True, type=Path)
    ap.add_argument("--config", default="configs/default_dynamic.yaml")
    ap.add_argument("--n-scenarios", type=int, default=15)
    ap.add_argument("--gap-time-limit", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None,
                     help="Where to write per-checkpoint CSVs (default: <checkpoint-dir>/sweep)")
    args = ap.parse_args()

    ckpts = find_checkpoints(args.checkpoint_dir)
    if not ckpts:
        print(f"No checkpoint_*.pt found in {args.checkpoint_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir) if args.out_dir else args.checkpoint_dir / "sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for label, ckpt_path in ckpts:
        csv_path = out_dir / f"eval_ep{label}.csv"
        print(f"--- episode {label} ({ckpt_path.name}) ---", flush=True)
        cmd = [
            sys.executable, "evaluation/eval_dynamic.py",
            "--checkpoint", str(ckpt_path),
            "--config", args.config,
            "--n-scenarios", str(args.n_scenarios),
            "--seed", str(args.seed),
            "--optimality-gap",
            "--gap-time-limit", str(args.gap_time_limit),
            "--output-csv", str(csv_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stderr[-2000:], file=sys.stderr)
            print(f"  FAILED, skipping episode {label}", file=sys.stderr)
            continue
        results.append((label, summarize_csv(csv_path)))

    print()
    print(f"{'episode':>9s} {'tier':>7s} {'hgt_wt':>8s} {'vs_edf':>8s} {'vs_bp':>8s} {'gap_to_W*':>10s} {'solved':>7s}")
    print("-" * 62)
    for label, summary in results:
        for tier in ["easy", "medium", "hard"]:
            if tier not in summary:
                continue
            s = summary[tier]
            print(f"{label:>9s} {tier:>7s} {s['hgt_wt']:8.4f} {s['vs_edf']:+7.1f}% {s['vs_bp']:+7.1f}% "
                  f"{s['gap_pct']:+9.1f}% {s['n_solved']:3d}/{s['n']:<3d}")
        if "overall" in summary:
            o = summary["overall"]
            print(f"{label:>9s} {'OVERALL':>7s} {o['hgt_wt']:8.4f} {o['vs_edf']:+7.1f}% {o['vs_bp']:+7.1f}%")
        print()


if __name__ == "__main__":
    main()
