"""Entry point: train the HGT scheduling policy."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from intersection_scheduler.training.trainer import train


def main():
    parser = argparse.ArgumentParser(description="Train HGT intersection scheduler")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "configs" / "default.yaml"),
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--output",
        default="results",
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to checkpoint to resume from (e.g. results/checkpoint_30000.pt)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg, output_dir=args.output, resume=args.resume)


if __name__ == "__main__":
    main()
