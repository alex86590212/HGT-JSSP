"""Compare dynamic HGT checkpoint vs iGreedy and LIFO on the online 4x4.

Mirrors eval_4x4.py (per-tier comparison table, CSV export) for the dynamic
scheduler: frozen Poisson arrival streams per tier (seeded, identical across
methods via deepcopy), all methods driven through the SAME episode loop —
affected-set replanning, zone queues, lock transitions — differing only in
which feasible operation they pick each step:

  HGT     — deterministic policy argmax (the trained checkpoint)
  iGreedy — pick the candidate that could START earliest if planned now
            (online analog of the offline igreedy baseline; the primary
            baseline, matching the DATE paper's evaluation protocol)
  LIFO    — pick the candidate whose vehicle arrived MOST recently
            (inverse of a FIFO/arrival-order rule; a stress-test baseline,
            not in DATE's protocol — included for reference only)

Optionally (--optimality-gap), also reports the restricted-information
optimal W* from dynamic_scheduler.evaluation.optimal_solver_online — a
CP-SAT re-solve at every detection event using only currently-detected
vehicles and already-LOCKED ops as fixed constraints, i.e. optimal under the
SAME information boundary the online policy operates under (not the
full-episode-upfront oracle the offline eval uses, which would score every
method against a target no online method could ever reach). Off by default:
each episode needs one CP-SAT solve per detection event, materially slower
than the other three methods.

Metrics per episode: unbiased mean waiting time (completed + in-flight, see
dynamic_scheduler.utils.metrics) and completion rate. Self-contained episode
loop (does not import from trainer.py, same convention as eval_4x4.py).
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

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

from dynamic_scheduler.evaluation.optimal_solver_online import restricted_optimal_episode

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


def lifo_select(env: DynamicIntersectionEnv, mask: torch.Tensor) -> int:
    """Most-recently-arrived vehicle first — the inverse priority rule to
    FIFO. Same tie-break structure (route order within a vehicle), just
    maximizing arrival_time instead of minimizing it."""
    best, best_key = -1, None
    for i in mask.nonzero(as_tuple=True)[0].tolist():
        op = env.operations[i]
        vehicle = env.vehicles[op.vehicle_id]
        key = (-vehicle.arrival_time, op.vehicle_id, op.route_position)
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
    parser = argparse.ArgumentParser(description="Evaluate dynamic HGT vs iGreedy")
    parser.add_argument("--checkpoint", default="results_dynamic/checkpoint_best.pt",
                        help="Path to .pt checkpoint")
    parser.add_argument("--config", default="configs/default_dynamic.yaml",
                        help="Config (model arch + env windows must match training)")
    parser.add_argument("--n-scenarios", type=int, default=30,
                        help="Frozen arrival streams per tier")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-csv", default=None,
                        help="Optional path to save per-scenario CSV results")
    parser.add_argument("--optimality-gap", action="store_true",
                        help="Also compute the restricted-information optimal W* per "
                             "scenario (CP-SAT re-solve at every detection event) and "
                             "report HGT/iGreedy/LIFO gaps to it. Slow: one CP-SAT solve "
                             "per detection event per scenario.")
    parser.add_argument("--gap-time-limit", type=float, default=5.0,
                        help="Per-solve CP-SAT time limit (seconds) for --optimality-gap")
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
        "igreedy": igreedy_select,
        "lifo": lifo_select,
        "hgt": make_hgt_selector(policy),
    }

    # Percentage gaps are only meaningful when W* is not near zero — dividing
    # a small absolute gap by a ~0s optimum produces thousand-percent
    # artifacts. Same floor/convention as evaluation/eval_gap_4x4.py.
    _PCT_GAP_WSTAR_FLOOR = 0.05

    rows: List[dict] = []
    results = {}
    print()
    for tier, scenarios in scenarios_by_tier.items():
        sums = {m: {"wt": 0.0, "comp": 0.0} for m in methods}
        pct_gaps: Dict[str, List[float]] = {m: [] for m in methods}
        abs_gaps: Dict[str, List[float]] = {m: [] for m in methods}
        n_gap_solved = 0
        for i, arrivals in enumerate(scenarios):
            row = {"tier": tier, "scenario_id": i, "n_vehicles": len(arrivals)}
            wt_by_method = {}
            for name, selector in methods.items():
                wt, comp, _ = run_episode_with(
                    make_env(), copy.deepcopy(arrivals), episode_duration, selector,
                )
                sums[name]["wt"] += wt
                sums[name]["comp"] += comp
                row[f"{name}_waiting_time"] = wt
                row[f"{name}_completion_rate"] = comp
                wt_by_method[name] = wt

            if args.optimality_gap:
                w_star, _n_solved, _n_unsolved = restricted_optimal_episode(
                    arrivals, episode_duration,
                    detection_window=env_cfg.get("detection_window", 10.0),
                    commit_window=env_cfg.get("commit_window", 2.5),
                    time_limit_seconds=args.gap_time_limit,
                )
                row["w_star"] = w_star if w_star is not None else ""
                row["w_star_solved"] = w_star is not None
                if w_star is not None:
                    n_gap_solved += 1
                    for name, wt in wt_by_method.items():
                        abs_g = wt - w_star
                        abs_gaps[name].append(abs_g)
                        row[f"{name}_abs_gap"] = abs_g
                        if w_star >= _PCT_GAP_WSTAR_FLOOR:
                            pct_g = abs_g / w_star * 100.0
                            pct_gaps[name].append(pct_g)
                            row[f"{name}_gap_pct"] = pct_g

            rows.append(row)

        n = len(scenarios)
        means = {m: {k: v / n for k, v in s.items()} for m, s in sums.items()}
        results[tier] = means
        imp_ig = (means["igreedy"]["wt"] - means["hgt"]["wt"]) / (means["igreedy"]["wt"] + 1e-9) * 100.0
        imp_lifo = (means["lifo"]["wt"] - means["hgt"]["wt"]) / (means["lifo"]["wt"] + 1e-9) * 100.0
        print(
            f"[{tier:6s}]  "
            f"iGreedy={means['igreedy']['wt']:.3f}/{means['igreedy']['comp']:.2f}  "
            f"LIFO={means['lifo']['wt']:.3f}/{means['lifo']['comp']:.2f}  "
            f"HGT={means['hgt']['wt']:.3f}/{means['hgt']['comp']:.2f}  "
            f"| HGT vs iGreedy {imp_ig:+.1f}%  vs LIFO {imp_lifo:+.1f}%"
        )
        if args.optimality_gap:
            gap_strs = []
            for name in methods:
                if abs_gaps[name]:
                    mean_abs = sum(abs_gaps[name]) / len(abs_gaps[name])
                    if pct_gaps[name]:
                        mean_pct = sum(pct_gaps[name]) / len(pct_gaps[name])
                        gap_strs.append(f"{name}={mean_abs:+.3f}s ({mean_pct:+.1f}%)")
                    else:
                        # No scenario cleared the pct-gap W* floor (all
                        # near-zero optimums) — percentage is unstable there,
                        # but the absolute-second gap is still meaningful.
                        gap_strs.append(f"{name}={mean_abs:+.3f}s (pct n/a, W*~0)")
                else:
                    gap_strs.append(f"{name}=n/a")
            print(
                f"          restricted-optimal solved {n_gap_solved}/{n} scenarios  "
                f"| gap to W*: {'  '.join(gap_strs)}"
            )

    n_tiers = len(results)
    overall = {
        m: sum(results[t][m]["wt"] for t in results) / n_tiers
        for m in methods
    }
    imp_ig = (overall["igreedy"] - overall["hgt"]) / (overall["igreedy"] + 1e-9) * 100.0
    imp_lifo = (overall["lifo"] - overall["hgt"]) / (overall["lifo"] + 1e-9) * 100.0
    print(
        f"[{'overall':6s}]  iGreedy={overall['igreedy']:.3f}  LIFO={overall['lifo']:.3f}  "
        f"HGT={overall['hgt']:.3f}  | HGT vs iGreedy {imp_ig:+.1f}%  vs LIFO {imp_lifo:+.1f}%"
    )

    if args.output_csv:
        import csv as csv_module
        out_path = Path(args.output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["tier", "scenario_id", "n_vehicles"] + [
            f"{m}_{metric}" for m in methods for metric in ("waiting_time", "completion_rate")
        ]
        if args.optimality_gap:
            fieldnames += ["w_star", "w_star_solved"] + [
                f"{m}_{metric}" for m in methods for metric in ("abs_gap", "gap_pct")
            ]
        with out_path.open("w", newline="") as f:
            writer = csv_module.DictWriter(f, fieldnames=fieldnames, restval="")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
