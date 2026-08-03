"""Fine-tune a converged dynamic-scheduler checkpoint on real SinD traffic.

Two-phase workflow: train_dynamic.py runs the full synthetic curriculum
first (that's where the policy learns the mechanics of zone-priority
scheduling — 78 real scenarios is nowhere near enough data to learn from
scratch). This script takes that converged checkpoint and continues PPO
training for a short additional phase using ONLY real SinD departure
scenarios (dynamic_scheduler/data/sind_loader.py), so the policy specializes
toward real-world arrival timing patterns before final evaluation.

Held out by INTERSECTION, not by random scenario (adjacent windows from the
same recording are strongly correlated — see sind_dataset.md's own
recommendation): Xi'an (the largest site, ~26 scenarios) is reserved for
eval; fine-tuning trains only on Chongqing + Tianjin (~52 scenarios).

Reuses dynamic_scheduler.training.trainer.train unchanged via its
arrivals_fn/eval_fn seams — same PPO loop, buffer cadence, checkpointing,
and CPU/GPU device split as the main curriculum run.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import torch
import yaml

from dynamic_scheduler.data.sind_loader import SinDCatalog
from dynamic_scheduler.data.traffic_generator import TrafficGenerator
from dynamic_scheduler.environment.dynamic_intersection import DynamicVehicle
from dynamic_scheduler.training.trainer import run_episode, train
from intersection_scheduler.model.policy import SchedulingPolicy

HELD_OUT_INTERSECTION = "xi_an_xi_an_shanglin"


def _split_by_intersection(
    catalog: SinDCatalog, held_out: str,
) -> Tuple[List[Path], List[Path]]:
    train_paths = [p for p in catalog.scenario_paths if held_out not in str(p)]
    eval_paths = [p for p in catalog.scenario_paths if held_out in str(p)]
    return train_paths, eval_paths


def main():
    parser = argparse.ArgumentParser(description="Fine-tune dynamic HGT on real SinD scenarios")
    parser.add_argument("--checkpoint", required=True,
                        help="Converged checkpoint from the main synthetic-curriculum run")
    parser.add_argument("--config", default="configs/default_dynamic.yaml",
                        help="Same config used for the main run (model arch + env windows must match)")
    parser.add_argument("--output", default="results_dynamic_sind_finetune",
                        help="Output directory for the fine-tuned checkpoints/logs")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Passes over the training-split scenarios (78 total scenarios "
                             "is little data for PPO, so repeat with a reshuffled order each epoch)")
    parser.add_argument("--eval-interval", type=int, default=1,
                        help="Run the held-out-intersection eval every N epochs (in episodes: "
                             "every N * n_train_scenarios)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    catalog = SinDCatalog()
    train_paths, eval_paths = _split_by_intersection(catalog, HELD_OUT_INTERSECTION)
    print(f"SinD split: {len(train_paths)} train scenarios "
          f"(all intersections except {HELD_OUT_INTERSECTION}), "
          f"{len(eval_paths)} held-out eval scenarios ({HELD_OUT_INTERSECTION})", flush=True)
    if not train_paths or not eval_paths:
        raise RuntimeError(
            f"Empty split: {len(train_paths)} train / {len(eval_paths)} eval scenarios. "
            "Check dynamic_scheduler/data/processed/sind/ is populated."
        )

    rng = random.Random(args.seed)
    # Fixed episode order for this run: shuffled once per epoch, deterministic
    # given --seed, so a resumed/repeated run is reproducible.
    n_train = len(train_paths)
    episode_order: List[Path] = []
    for _ in range(args.epochs):
        epoch_paths = list(train_paths)
        rng.shuffle(epoch_paths)
        episode_order.extend(epoch_paths)

    num_episodes = len(episode_order)
    cfg.setdefault("training", {})["num_episodes"] = num_episodes
    # Every eval_interval epochs, in episode units; at least every episode.
    cfg["training"]["eval_interval"] = max(1, args.eval_interval * n_train)
    cfg["training"]["log_interval"] = min(cfg["training"].get("log_interval", 1), n_train)
    cfg["training"].setdefault("checkpoint_interval", num_episodes)  # only at the end unless overridden
    cfg["training"]["num_workers"] = 1  # tiny dataset; multiprocessing overhead isn't worth it here

    def arrivals_fn(episode: int, gen: TrafficGenerator, episode_duration: float):
        path = episode_order[episode - 1]
        arrivals, ep_duration, meta = catalog.load_scenario(path, rng=rng)
        return arrivals, ep_duration

    def eval_fn(policy: SchedulingPolicy) -> "tuple[float, float]":
        from dynamic_scheduler.environment.dynamic_intersection import DynamicIntersectionEnv
        from dynamic_scheduler.utils.metrics import completion_rate
        from intersection_scheduler.data.scenario_generator_4x4 import ZONE_POSITIONS

        env_cfg = cfg.get("environment", {})
        total_wt = 0.0
        total_seen = 0
        total_completed = 0
        policy.eval()
        eval_rng = random.Random(args.seed)
        for path in eval_paths:
            arrivals, ep_duration, _ = catalog.load_scenario(path, rng=eval_rng)
            env = DynamicIntersectionEnv(
                detection_window=env_cfg.get("detection_window", 10.0),
                commit_window=env_cfg.get("commit_window", 2.5),
                zone_positions=ZONE_POSITIONS,
                penalty_coef=env_cfg.get("penalty_coef", 0.1),
                max_proximity_weight=env_cfg.get("max_proximity_weight", 2.0),
            )
            _, stats = run_episode(policy, env, arrivals, ep_duration, deterministic=True)
            total_wt += stats["waiting_time"] * stats["n_vehicles_seen"]
            total_seen += stats["n_vehicles_seen"]
            total_completed += stats["n_vehicles_completed"]
        policy.train()
        mean_wt = total_wt / total_seen if total_seen > 0 else 0.0
        comp_rate = total_completed / total_seen if total_seen > 0 else 0.0
        return mean_wt, comp_rate

    print(f"Fine-tuning for {num_episodes} episodes ({args.epochs} epochs x {n_train} scenarios), "
          f"eval on {len(eval_paths)} held-out scenarios every {cfg['training']['eval_interval']} episodes",
          flush=True)

    # Load weights only (not resume=): fine-tuning is a fresh PPO phase on a
    # different data distribution — a stale Adam moment/variance state from
    # the synthetic-curriculum optimizer isn't appropriate to carry over, and
    # `resume=` also offsets start_episode past 1, which would desync
    # episode_order's 1-based indexing into the shuffled scenario list above.
    _pretrained = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    initial_state = _pretrained["policy"] if isinstance(_pretrained, dict) and "policy" in _pretrained else _pretrained

    train(
        cfg,
        output_dir=args.output,
        resume=None,
        arrivals_fn=arrivals_fn,
        eval_fn=eval_fn,
        initial_state_dict=initial_state,
    )


if __name__ == "__main__":
    main()
