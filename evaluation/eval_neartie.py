from __future__ import annotations

import argparse

import torch
import yaml

from dynamic_scheduler.data.neartie_scenarios import neartie_batch
from dynamic_scheduler.environment.dynamic_intersection import DynamicIntersectionEnv
from evaluation.eval_dynamic import edf_select, make_hgt_selector, run_episode_with
from intersection_scheduler.data.scenario_generator_4x4 import ZONE_POSITIONS
from intersection_scheduler.model.policy import SchedulingPolicy


def run_one(env, arrivals, episode_duration, selector):
    env.reset(list(arrivals), episode_duration)
    wt, comp, n_seen = run_episode_with(env, arrivals, episode_duration, selector)
    per_vehicle = {e["vehicle_id"]: e["waiting_time"] for e in env.completed_log}
    for vid, wtv in zip([v.id for v in env.vehicles.values()], env.inflight_waiting_times()):
        per_vehicle[vid] = wtv
    return wt, comp, per_vehicle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="results_dynamic_reward_fix/checkpoint_best_30_so_far (1).pt")
    ap.add_argument("--config", default="configs/default_dynamic.yaml")
    ap.add_argument("--n-scenarios", type=int, default=30)
    ap.add_argument("--n-queue", type=int, default=2)
    ap.add_argument("--episode-duration", type=float, default=20.0)
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

    scenarios = neartie_batch(args.n_scenarios, n_queue=args.n_queue)

    print(f"near-tie scenario: congested-candidate (queue of {args.n_queue} ahead) vs "
          f"clear-candidate (empty zone), deadlines ~0.02s apart, n_scenarios={args.n_scenarios}\n")

    methods = {"edf": edf_select, "hgt": hgt_select}
    results = {name: {"wt": [], "congested_wt": [], "clear_wt": []} for name in methods}

    for arrivals in scenarios:
        congested_vid = arrivals[-2].id
        clear_vid = arrivals[-1].id
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
            if congested_vid in per_vehicle:
                results[name]["congested_wt"].append(per_vehicle[congested_vid])
            if clear_vid in per_vehicle:
                results[name]["clear_wt"].append(per_vehicle[clear_vid])

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    print(f"{'method':8s} {'overall_wt':>12s} {'congested_wt':>14s} {'clear_wt':>10s}")
    for name in methods:
        r = results[name]
        print(f"{name:8s} {mean(r['wt']):12.4f} {mean(r['congested_wt']):14.4f} {mean(r['clear_wt']):10.4f}")

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
