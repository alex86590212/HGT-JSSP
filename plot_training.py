"""Plot training curves from TensorBoard event file."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


CURRICULUM_PHASES = [
    (0,     5_000,  "easy\n(2v)",    "#d4f0d4"),
    (5_000, 15_000, "medium\n(3-5v)", "#fff3cd"),
    (15_000,30_000, "hard\n(4-7v)",   "#fde2e2"),
    (30_000, None,  "random\n(3-9v)", "#e2eafd"),
]


def rolling(values, window=20):
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def load_scalar(acc, tag):
    events = acc.Scalars(tag)
    steps  = np.array([e.step  for e in events])
    vals   = np.array([e.value for e in events])
    return steps, vals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="results_v2/tb")
    parser.add_argument("--out",    default="results_v2/training_curves.png")
    args = parser.parse_args()

    acc = EventAccumulator(args.logdir)
    acc.Reload()
    print("Available tags:", acc.Tags()["scalars"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("HGT-JSSP Training Curves", fontsize=14, fontweight="bold")

    def shade_phases(ax, ymin, ymax):
        for start, end, label, color in CURRICULUM_PHASES:
            x0 = start
            x1 = end if end is not None else ax.get_xlim()[1]
            ax.axvspan(x0, x1, color=color, alpha=0.4, zorder=0)
            mid = (x0 + (x1 if end else x0 + 5000)) / 2
            ax.text(mid, ymax * 0.97, label, ha="center", va="top",
                    fontsize=7, color="#555555")

    # ── 1. Eval waiting time ──────────────────────────────────────────────
    ax = axes[0, 0]
    steps, vals = load_scalar(acc, "eval/waiting_time")
    ax.plot(steps, vals, "o-", color="#2196F3", linewidth=1.5,
            markersize=4, label="eval wt")
    # best-so-far line
    best = np.minimum.accumulate(vals)
    ax.plot(steps, best, "--", color="#F44336", linewidth=1, label="best so far")
    ax.set_title("Eval Waiting Time (hard, 5 vehicles)")
    ax.set_ylabel("mean waiting time (s)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    shade_phases(ax, 0, vals.max() * 1.05)
    ax.set_ylim(0, vals.max() * 1.05)

    # ── 2. Train waiting time (smoothed) ─────────────────────────────────
    ax = axes[0, 1]
    steps_wt, vals_wt = load_scalar(acc, "train/waiting_time")
    smooth = rolling(vals_wt, window=30)
    smooth_steps = steps_wt[len(steps_wt) - len(smooth):]
    ax.plot(steps_wt, vals_wt, color="#90CAF9", linewidth=0.5, alpha=0.5)
    ax.plot(smooth_steps, smooth, color="#1565C0", linewidth=1.8,
            label="30-ep avg")
    ax.set_title("Train Waiting Time")
    ax.set_ylabel("waiting time (s)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    shade_phases(ax, 0, vals_wt.max() * 1.05)
    ax.set_ylim(0, vals_wt.max() * 1.05)

    # ── 3. Steps per episode ──────────────────────────────────────────────
    ax = axes[1, 0]
    steps_s, vals_s = load_scalar(acc, "train/steps")
    smooth_s = rolling(vals_s, window=30)
    smooth_steps_s = steps_s[len(steps_s) - len(smooth_s):]
    ax.plot(steps_s, vals_s, color="#C8E6C9", linewidth=0.5, alpha=0.5)
    ax.plot(smooth_steps_s, smooth_s, color="#2E7D32", linewidth=1.8,
            label="30-ep avg")
    ax.set_title("Steps per Episode (= num vehicles × route length)")
    ax.set_ylabel("steps")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    shade_phases(ax, 0, vals_s.max() * 1.1)
    ax.set_ylim(0, vals_s.max() * 1.1)

    # ── 4. Total reward (smoothed) ────────────────────────────────────────
    ax = axes[1, 1]
    steps_r, vals_r = load_scalar(acc, "train/total_reward")
    smooth_r = rolling(vals_r, window=30)
    smooth_steps_r = steps_r[len(steps_r) - len(smooth_r):]
    ax.plot(steps_r, vals_r, color="#F8BBD0", linewidth=0.5, alpha=0.5)
    ax.plot(smooth_steps_r, smooth_r, color="#C62828", linewidth=1.8,
            label="30-ep avg")
    ax.set_title("Episode Total Reward")
    ax.set_ylabel("reward")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    shade_phases(ax, vals_r.min() * 1.1, 0)
    ax.set_ylim(vals_r.min() * 1.1, 0.5)

    for ax in axes.flat:
        ax.set_xlabel("episode")

    # Phase legend
    handles = [
        mpatches.Patch(color=color, alpha=0.6, label=label.replace("\n", " "))
        for _, _, label, color in CURRICULUM_PHASES
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               fontsize=8, title="Curriculum phase", title_fontsize=8,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.show()


if __name__ == "__main__":
    main()
