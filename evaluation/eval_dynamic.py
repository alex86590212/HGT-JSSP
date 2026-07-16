"""Compare dynamic HGT checkpoint vs iGreedy and FIFO on the online 4x4.

Mirrors eval_4x4.py (per-tier comparison table, CSV export) for the dynamic
scheduler: frozen Poisson arrival streams per tier (seeded, identical across
methods via deepcopy), all methods driven through the SAME episode loop —
affected-set replanning, zone queues, lock transitions — differing only in
which feasible operation they pick each step:

  HGT     — deterministic policy argmax (the trained checkpoint)
  iGreedy — pick the candidate that could START earliest if planned now
            (online analog of the offline igreedy baseline)
  FIFO    — pick the candidate whose vehicle arrived first (route order
            within a vehicle): queue priorities collapse to arrival order

Metrics per episode: unbiased mean waiting time (completed + in-flight, see
dynamic_scheduler.utils.metrics) and completion rate. Self-contained episode
loop (does not import from trainer.py, same convention as eval_4x4.py).
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
import yaml

from dynamic_scheduler.data.traffic_generator import TrafficGenerator
from dynamic_scheduler.environment.dynamic_intersection import (
    DynamicIntersectionEnv,
    DynamicOperation,
    OpState,
)
from dynamic_scheduler.environment.feasibility import compute_feasible_set
from dynamic_scheduler.environment.graph_builder import build_hetero_graph
from dynamic_scheduler.utils.metrics import (
    completion_rate,
    episode_waiting_time_all,
)
from intersection_scheduler.data.scenario_generator_4x4 import ZONE_POSITIONS
from intersection_scheduler.model.policy import SchedulingPolicy

# Same hard caps as trainer.run_episode: generous for any real episode,
# finite so a non-convergence bug fails loudly instead of hanging.
MAX_REPLAN_ITERS = 500
MAX_OUTER_ITERS = 5000

Selector = Callable[[DynamicIntersectionEnv, torch.Tensor], int]


def run_episode_with(
    env: DynamicIntersectionEnv,
    arrivals,
    episode_duration: float,
    select_action: Selector,
) -> Tuple[float, float, int]:
    """One online episode with the given action selector.

    Identical loop mechanics to trainer.run_episode (affected-op passes,
    visited-once-per-pass, event-driven time advance) so every method is
    compared under exactly the same replanning regime.

    Returns (waiting_time, completion_rate, n_vehicles_seen)."""
    env.reset(arrivals, episode_duration)
    n_seen = len(env.vehicles)
    trigger_vids = list(env.vehicles.keys())

    done = False
    outer_iters = 0
    while not done:
        outer_iters += 1
        if outer_iters > MAX_OUTER_ITERS:
            raise RuntimeError("episode outer loop exceeded MAX_OUTER_ITERS")

        affected = set(env.affected_op_indices(trigger_vids))
        visited: set = set()
        replan_iters = 0
        while True:
            replan_iters += 1
            if replan_iters > MAX_REPLAN_ITERS:
                raise RuntimeError("replan pass exceeded MAX_REPLAN_ITERS")
            candidates = affected - visited
            if not candidates:
                break
            mask = compute_feasible_set(env, candidates)
            if not mask.any():
                break
            action = select_action(env, mask)
            env, _, _ = env.plan_operation(action)
            visited.add(action)

        done, newly_detected = env.advance_time()
        n_seen += len(newly_detected)
        trigger_vids = newly_detected

    inflight = env.inflight_waiting_times()
    wt = episode_waiting_time_all(env.completed_log, inflight)
    comp = completion_rate(len(env.completed_log), n_seen)
    return wt, comp, n_seen


# ----------------------------------------------------------------------
# Action selectors
# ----------------------------------------------------------------------

def make_hgt_selector(policy: SchedulingPolicy) -> Selector:
    device = next(policy.parameters()).device

    def select(env: DynamicIntersectionEnv, mask: torch.Tensor) -> int:
        data = build_hetero_graph(env, feasible_mask=mask)
        with torch.no_grad():
            dist, _ = policy(data.to(device), mask.to(device))
        return int(dist.probs.argmax().item())

    return select


def fifo_select(env: DynamicIntersectionEnv, mask: torch.Tensor) -> int:
    """Earliest-arrived vehicle first; a vehicle's own ops in route order."""
    best, best_key = -1, None
    for i in mask.nonzero(as_tuple=True)[0].tolist():
        op = env.operations[i]
        vehicle = env.vehicles[op.vehicle_id]
        key = (vehicle.arrival_time, op.vehicle_id, op.route_position)
        if best_key is None or key < best_key:
            best, best_key = i, key
    return best


def igreedy_select(env: DynamicIntersectionEnv, mask: torch.Tensor) -> int:
    """Pick the candidate with the earliest achievable start if planned now."""
    best, best_key = -1, None
    for i in mask.nonzero(as_tuple=True)[0].tolist():
        op = env.operations[i]
        start = _start_if_planned_now(env, op)
        vehicle = env.vehicles[op.vehicle_id]
        key = (start, vehicle.arrival_time, op.vehicle_id, op.route_position)
        if best_key is None or key < best_key:
            best, best_key = i, key
    return best


def _start_if_planned_now(env: DynamicIntersectionEnv, op: DynamicOperation) -> float:
    """Start time op would get if appended to its zone queue right now —
    same rule as env._recompute_times for a single op: chain/arrival base,
    then behind the tentative queue tail, then bumped over locked windows."""
    base = env.current_time
    if op.route_position == 0:
        vehicle = env.vehicles.get(op.vehicle_id)
        if vehicle is not None:
            base = max(base, vehicle.arrival_time)
    else:
        pred: Optional[DynamicOperation] = next(
            (o for o in env.operations
             if o.vehicle_id == op.vehicle_id
             and o.route_position == op.route_position - 1),
            None,
        )
        if pred is not None and pred.state != OpState.UNSCHEDULED:
            base = max(base, pred.earliest_finish)

    queue = env._zone_queue.get(op.zone_id, [])
    for o in queue:
        if o is not op and o.earliest_finish > base:
            base = o.earliest_finish

    windows = sorted(
        (o.start_time, o.earliest_finish)
        for o in env.operations
        if o.state == OpState.LOCKED and o.zone_id == op.zone_id
    )
    for w_start, w_end in windows:
        if base + op.processing_time <= w_start + 1e-9:
            break
        if base < w_end - 1e-9:
            base = w_end
    return base


# ----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Evaluate dynamic HGT vs iGreedy vs FIFO")
    parser.add_argument("--checkpoint", default="results_dynamic/checkpoint_best.pt",
                        help="Path to .pt checkpoint")
    parser.add_argument("--config", default="configs/default_dynamic.yaml",
                        help="Config (model arch + env windows must match training)")
    parser.add_argument("--n-scenarios", type=int, default=30,
                        help="Frozen arrival streams per tier")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-csv", default=None,
                        help="Optional path to save per-scenario CSV results")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg.get("model", {})
    env_cfg = cfg.get("environment", {})
    episode_duration = env_cfg.get("episode_duration", 60.0)

    torch.set_num_threads(1)  # tiny sequential forwards: single-thread is faster

    policy = SchedulingPolicy(
        hidden_dim=model_cfg.get("hidden_dim", 128),
        num_heads=model_cfg.get("num_heads", 4),
        num_layers=model_cfg.get("num_layers", 3),
    )
    ckpt = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    if isinstance(ckpt, dict) and "policy" in ckpt:
        policy.load_state_dict(ckpt["policy"])
        print(f"Loaded checkpoint: {args.checkpoint}  (episode {ckpt.get('episode', '?')})")
    else:
        policy.load_state_dict(ckpt)
        print(f"Loaded weights: {args.checkpoint}")
    policy.eval()

    def make_env() -> DynamicIntersectionEnv:
        return DynamicIntersectionEnv(
            detection_window=env_cfg.get("detection_window", 10.0),
            commit_window=env_cfg.get("commit_window", 2.5),
            zone_positions=ZONE_POSITIONS,
        )

    # Frozen arrival streams: one generator per tier with a fixed seed, so
    # every method (and every future checkpoint) sees identical traffic.
    tier_offsets = {"easy": 0, "medium": 1, "hard": 2}
    scenarios_by_tier = {}
    for tier, offset in tier_offsets.items():
        gen = TrafficGenerator(seed=args.seed * 10 + offset)
        scenarios_by_tier[tier] = [
            getattr(gen, tier)(episode_duration) for _ in range(args.n_scenarios)
        ]

    methods = {
        "fifo": fifo_select,
        "igreedy": igreedy_select,
        "hgt": make_hgt_selector(policy),
    }

    rows: List[dict] = []
    results = {}
    print()
    for tier, scenarios in scenarios_by_tier.items():
        sums = {m: {"wt": 0.0, "comp": 0.0} for m in methods}
        for i, arrivals in enumerate(scenarios):
            row = {"tier": tier, "scenario_id": i, "n_vehicles": len(arrivals)}
            for name, selector in methods.items():
                wt, comp, _ = run_episode_with(
                    make_env(), copy.deepcopy(arrivals), episode_duration, selector,
                )
                sums[name]["wt"] += wt
                sums[name]["comp"] += comp
                row[f"{name}_waiting_time"] = wt
                row[f"{name}_completion_rate"] = comp
            rows.append(row)

        n = len(scenarios)
        means = {m: {k: v / n for k, v in s.items()} for m, s in sums.items()}
        results[tier] = means
        imp_fifo = (means["fifo"]["wt"] - means["hgt"]["wt"]) / (means["fifo"]["wt"] + 1e-9) * 100.0
        imp_ig = (means["igreedy"]["wt"] - means["hgt"]["wt"]) / (means["igreedy"]["wt"] + 1e-9) * 100.0
        print(
            f"[{tier:6s}]  "
            f"FIFO={means['fifo']['wt']:.3f}/{means['fifo']['comp']:.2f}  "
            f"iGreedy={means['igreedy']['wt']:.3f}/{means['igreedy']['comp']:.2f}  "
            f"HGT={means['hgt']['wt']:.3f}/{means['hgt']['comp']:.2f}  "
            f"| HGT vs FIFO {imp_fifo:+.1f}%  vs iGreedy {imp_ig:+.1f}%"
        )

    n_tiers = len(results)
    overall = {
        m: sum(results[t][m]["wt"] for t in results) / n_tiers
        for m in methods
    }
    imp_fifo = (overall["fifo"] - overall["hgt"]) / (overall["fifo"] + 1e-9) * 100.0
    imp_ig = (overall["igreedy"] - overall["hgt"]) / (overall["igreedy"] + 1e-9) * 100.0
    print(
        f"[{'overall':6s}]  FIFO={overall['fifo']:.3f}  iGreedy={overall['igreedy']:.3f}  "
        f"HGT={overall['hgt']:.3f}  | HGT vs FIFO {imp_fifo:+.1f}%  vs iGreedy {imp_ig:+.1f}%"
    )

    if args.output_csv:
        import csv as csv_module
        out_path = Path(args.output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["tier", "scenario_id", "n_vehicles"] + [
            f"{m}_{metric}" for m in methods for metric in ("waiting_time", "completion_rate")
        ]
        with out_path.open("w", newline="") as f:
            writer = csv_module.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
