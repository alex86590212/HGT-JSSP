from __future__ import annotations

import argparse

import torch
import yaml

from dynamic_scheduler.data.pileup_scenarios import pileup_batch
from dynamic_scheduler.environment.dynamic_intersection import DynamicIntersectionEnv
from dynamic_scheduler.utils.metrics import episode_waiting_time_all
from evaluation.eval_dynamic import edf_select, make_hgt_selector, run_episode_with
from intersection_scheduler.data.scenario_generator_4x4 import ZONE_POSITIONS
from intersection_scheduler.model.policy import SchedulingPolicy


def run_one(env, arrivals, episode_duration, selector):
    env.reset(list(arrivals), episode_duration)
    wt, comp, n_seen = run_episode_with(env, arrivals, episode_duration, selector)
    per_vehicle = {}
    for entry in env.completed_log:
        per_vehicle[entry["vehicle_id"]] = entry["waiting_time"]
    for vid, wtv in zip(
        [v.id for v in env.vehicles.values()],
        env.inflight_waiting_times(),
    ):
        per_vehicle[vid] = wtv
    return wt, comp, per_vehicle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="results_dynamic_reward_fix/checkpoint_best_30_so_far (1).pt")
    ap.add_argument("--config", default="configs/default_dynamic.yaml")
    ap.add_argument("--n-scenarios", type=int, default=20)
    ap.add_argument("--n-trailing", type=int, default=3)
    ap.add_argument("--episode-duration", type=float, default=30.0)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg.get("model", {})
    env_cfg = cfg.get("environment", {})

    policy = SchedulingPolicy(
        hidden_dim=model_cfg.get("hidden_dim", 128),
        num_heads=model_cfg.get("num_heads", 4),
        num_layers=model_cfg.get("num_layers", 3),
    )
    ckpt = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    state_dict = ckpt["policy"] if isinstance(ckpt, dict) and "policy" in ckpt else ckpt
    policy.load_state_dict(state_dict)
    policy.eval()
    hgt_select = make_hgt_selector(policy)

    scenarios = pileup_batch(args.n_scenarios, n_trailing=args.n_trailing)

    print(f"pileup scenario: 1 urgent + {args.n_trailing} trailing (same contested route) "
          f"vs 1 clear-path vehicle, n_scenarios={args.n_scenarios}\n")

    methods = {"edf": edf_select, "hgt": hgt_select}
    results = {name: {"wt": [], "clear_vehicle_wt": [], "trailing_wt": []} for name in methods}

    for arrivals in scenarios:
        clear_vid = arrivals[-1].id
        trailing_vids = [v.id for v in arrivals[1:-1]]
        for name, selector in methods.items():
            env = DynamicIntersectionEnv(
                detection_window=env_cfg.get("detection_window", 10.0),
                commit_window=env_cfg.get("commit_window", 2.5),
                zone_positions=ZONE_POSITIONS,
                penalty_coef=env_cfg.get("penalty_coef", 0.1),
                max_proximity_weight=env_cfg.get("max_proximity_weight", 2.0),
            )
            wt, comp, per_vehicle = run_one(env, arrivals, args.episode_duration, selector)
            results[name]["wt"].append(wt)
            if clear_vid in per_vehicle:
                results[name]["clear_vehicle_wt"].append(per_vehicle[clear_vid])
            trailing_vals = [per_vehicle[v] for v in trailing_vids if v in per_vehicle]
            if trailing_vals:
                results[name]["trailing_wt"].append(sum(trailing_vals) / len(trailing_vals))

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    print(f"{'method':8s} {'overall_wt':>12s} {'clear_veh_wt':>14s} {'trailing_avg_wt':>16s}")
    for name in methods:
        r = results[name]
        print(f"{name:8s} {mean(r['wt']):12.4f} {mean(r['clear_vehicle_wt']):14.4f} {mean(r['trailing_wt']):16.4f}")

    print()
    edf_wt, hgt_wt = mean(results["edf"]["wt"]), mean(results["hgt"]["wt"])
    if hgt_wt < edf_wt:
        print(f"HGT beats EDF on this scenario: {hgt_wt:.4f} < {edf_wt:.4f} "
              f"({(edf_wt - hgt_wt) / edf_wt * 100:+.1f}%)")
    else:
        print(f"HGT does NOT beat EDF here: {hgt_wt:.4f} vs {edf_wt:.4f} "
              f"({(edf_wt - hgt_wt) / edf_wt * 100:+.1f}%)")


if __name__ == "__main__":
    main()
