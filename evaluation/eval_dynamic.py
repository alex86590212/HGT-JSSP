"""Compare dynamic HGT checkpoint vs iGreedy, LIFO, Backpressure, and EDF on
the online 4x4.

Mirrors eval_4x4.py (per-tier comparison table, CSV export) for the dynamic
scheduler: frozen Poisson arrival streams per tier (seeded, identical across
methods via deepcopy), all methods driven through the SAME episode loop —
affected-set replanning, zone queues, lock transitions — differing only in
which feasible operation they pick each step:

  HGT         — deterministic policy argmax (the trained checkpoint)
  iGreedy     — pick the candidate that could START earliest if planned now
                (online analog of the offline igreedy baseline; the primary
                baseline). Adapted from the iGreedy comparator of Huang et
                al. (DATE 2023) — their paper states iGreedy "schedules an
                operation once it is presented in the current action space"
                plus a deadlock check, but does not itself spell out a
                specific scoring rule, tie-break convention, or queue-tail
                treatment for choosing among multiple simultaneously-
                feasible candidates. This earliest-achievable-start scoring
                is our own concretization of that comparator, not verified
                to be identical to theirs — do not describe this as
                matching their protocol without checking those details.
  LIFO        — pick the candidate whose vehicle arrived MOST recently
                (inverse of a FIFO/arrival-order rule; a stress-test
                baseline, not part of Huang et al.'s comparator set —
                included here for reference only)
  Backpressure — pick the candidate whose ZONE has the largest current
                backlog (Tassiulas & Ephremides 1992 MaxWeight scheduling,
                adapted from wireless link scheduling). Throughput/stability
                -oriented, not delay-oriented — a legitimate baseline
                precisely because it targets a different objective than
                waiting time.
  EDF         — pick the candidate with the earliest theoretical-minimum
                arrival time AT ITS OWN ZONE (free-flow ETA accounting for
                route position), not overall episode arrival time. Adapted
                from Earliest-Deadline-First / Virtual-Clock packet
                scheduling; a tighter conceptual match to the waiting-time
                objective than FIFO.

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
import math
import random
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


def backpressure_select(env: DynamicIntersectionEnv, mask: torch.Tensor) -> int:
    """Backpressure / MaxWeight scheduling (Tassiulas & Ephremides, 1992):
    among contending candidates, serve whichever's ZONE currently has the
    largest backlog (queue differential) — the busiest contention point —
    rather than ordering by arrival time or soonest achievable start.

    Backlog here is the zone's current TENTATIVE queue length plus anyone
    already LOCKED into it (still occupying/blocking the zone), the direct
    analogue of queue-length in the wireless-link formulation this is
    adapted from. This is a throughput/stability-oriented policy, not a
    delay-oriented one (a well-documented property of backpressure in the
    queueing literature) — it's included as a baseline that optimizes a
    genuinely different objective than waiting time, not a delay heuristic."""
    best, best_key = -1, None
    for i in mask.nonzero(as_tuple=True)[0].tolist():
        op = env.operations[i]
        zone_id = op.zone_id
        backlog = len(env._zone_queue.get(zone_id, [])) + sum(
            1 for o in env.operations if o.zone_id == zone_id and o.state == OpState.LOCKED
        )
        vehicle = env.vehicles[op.vehicle_id]
        # Largest backlog wins; ties broken by arrival time then route order,
        # same tie-break convention as the other selectors.
        key = (-backlog, vehicle.arrival_time, op.vehicle_id, op.route_position)
        if best_key is None or key < best_key:
            best, best_key = i, key
    return best


def make_logit_routing_selector(eta: float, seed: int) -> Selector:
    """Baseline ADAPTED from Garjani et al., "Stability of Information-Based
    Routing in Dynamic Transportation Networks" (arXiv:2601.13066, 2026).

    That paper analyzes a continuous-time, macroscopic routing model: traffic
    density x_j(t) on each of p PARALLEL LINKS evolves via a mass-conservation
    ODE, and the population-level SPLIT of traffic across links r_j(t) follows
    logit/softmax dynamics driven by a bounded, density-dependent information
    signal u_j(x_j) (their eq. 2b): r_j <- e^{-eta*u_j(x_j)} / sum_i e^{-eta*
    u_i(x_i)}. It proves conditions on u(.) for a unique, stable, free-flow
    equilibrium. It contains NO discrete per-vehicle dispatch rule -- there is
    nothing in it shaped like a Selector(env, mask) -> action choice, since it
    never models discrete vehicles or a single shared resource contended by
    multiple requests at an instant. This is therefore an ADAPTATION of its
    core mechanism to this repo's discrete decision setting, not a
    reproduction of the paper's method or results:

      - Per candidate op, the paper's "path density" x_j is played by a
        bounded (Lipschitz-saturating) zone-congestion proxy in [0, 1):
        u_i = backlog_i / (backlog_i + 1), backlog_i defined identically to
        backpressure_select's (TENTATIVE queue length + LOCKED occupants of
        that zone). Boundedness matters here because the paper's stability
        result (Assumption 3 / eq. 12) explicitly requires u to be Lipschitz
        and bounded -- an unbounded raw backlog count (as backpressure_select
        uses) would not satisfy it.
      - Selection is genuinely stochastic (sampled, not argmax), matching the
        paper's own mechanism (2b is a population SPLIT across options, not a
        deterministic single winner): candidate i is drawn with probability
        proportional to e^{-eta*u_i}, i.e. congested zones are down-weighted
        rather than excluded. eta plays the paper's responsiveness parameter
        (eta -> 0: near-uniform random pick, ignoring congestion signal;
        eta -> inf: deterministic least-congested-first).
      - Sampling uses a dedicated random.Random(seed) instance (not the
        global RNG) so a run is exactly reproducible given (seed, eta) and
        independent of traffic-generation RNG state.

    Ties in u_i are broken implicitly by the sampling itself, not by the
    (arrival_time, vehicle_id, route_position) convention the other
    selectors use, since this rule is stochastic by construction.
    """
    rng = random.Random(seed)

    def select(env: DynamicIntersectionEnv, mask: torch.Tensor) -> int:
        idxs = mask.nonzero(as_tuple=True)[0].tolist()
        weights = []
        for i in idxs:
            op = env.operations[i]
            zone_id = op.zone_id
            backlog = len(env._zone_queue.get(zone_id, [])) + sum(
                1 for o in env.operations if o.zone_id == zone_id and o.state == OpState.LOCKED
            )
            u = backlog / (backlog + 1.0)
            weights.append(math.exp(-eta * u))
        total = sum(weights)
        pick = rng.uniform(0.0, total)
        cumulative = 0.0
        for idx, w in zip(idxs, weights):
            cumulative += w
            if pick <= cumulative:
                return idx
        return idxs[-1]

    return select


def edf_select(env: DynamicIntersectionEnv, mask: torch.Tensor) -> int:
    """Earliest-Deadline-First / Virtual-Clock-style scheduling (Liu &
    Layland 1973; Zhang 1990): serve whichever candidate has the earliest
    THEORETICAL-MINIMUM arrival time at its own zone — vehicle arrival_time
    plus the free-flow sum of processing times for every zone earlier in
    its route — rather than the vehicle's overall episode arrival time
    (FIFO) or its currently-achievable start (iGreedy). This is a tighter
    match to the actual objective (finish - theoretical_minimum) than FIFO,
    since it accounts for vehicles with different route lengths/speeds
    reaching a given zone at different theoretical times even if they
    entered the system at the same moment."""
    best, best_key = -1, None
    for i in mask.nonzero(as_tuple=True)[0].tolist():
        op = env.operations[i]
        vehicle = env.vehicles[op.vehicle_id]
        free_flow_offset = sum(vehicle.processing_times[:op.route_position])
        deadline = vehicle.arrival_time + free_flow_offset
        key = (deadline, op.vehicle_id, op.route_position)
        if best_key is None or key < best_key:
            best, best_key = i, key
    return best


# Fraction of the original free-flow route-position offset RETAINED by
# edf_discounted_select — see that function's docstring. Chosen by sweeping
# {0.0 ... 1.0} against iGreedy under the blocking-fixed environment across
# all 8 scaling-study rates (evaluation ad hoc, not wired into a CLI flag):
# every value in roughly [0.6, 1.0] beat iGreedy in aggregate, and 0.9 was
# the best compromise between aggregate improvement and still using most of
# the discount (values near 1.0 approach discarding route position entirely,
# which stops being recognizably "iEDF"). Confirmed over 10 seeds x 15
# scenarios x 8 rates (1200 episodes/method): beats original EDF at every
# rate (+1.7% to +15.6%) and iGreedy at 6 of 8 rates (+1.7% to +7.1%), with
# two small losses to iGreedy (rate 0.35: -0.5%, rate 0.65: -0.9%).
_EDF_DISCOUNT_RETAINED_FRACTION = 0.1


def edf_discounted_select(env: DynamicIntersectionEnv, mask: torch.Tensor) -> int:
    """iEDF with a route-position discount (methodological revision, not a
    bug fix): under blocking, a vehicle denied its next zone keeps occupying
    (blocking) its current one, so delaying an operation deep in a long
    route is far more expensive than delaying a fresh entry — a cost the
    original fixed free-flow tag cannot see, since that tag structurally
    grows with route depth even under zero contention (a vehicle 4 zones
    into a 5-zone route always has a "later" free-flow tag than a vehicle
    just entering a 2-zone route competing for the same downstream zone,
    regardless of which is more urgent to unblock — see the vid604/vid608
    case study this was diagnosed from). This rule keeps the original tag's
    structure but retains only a small fraction of the free-flow offset
    accrued from zones already traversed, so route position still matters
    but no longer dominates raw arrival order. Still a fixed, per-vehicle
    static quantity computed once at detection — no queue or schedule state
    involved, so it keeps iEDF's queue-state-independence property."""
    best, best_key = -1, None
    for i in mask.nonzero(as_tuple=True)[0].tolist():
        op = env.operations[i]
        vehicle = env.vehicles[op.vehicle_id]
        free_flow_offset = sum(vehicle.processing_times[:op.route_position])
        deadline = vehicle.arrival_time + _EDF_DISCOUNT_RETAINED_FRACTION * free_flow_offset
        key = (deadline, op.vehicle_id, op.route_position)
        if best_key is None or key < best_key:
            best, best_key = i, key
    return best


def edf_priority_inheritance_select(env: DynamicIntersectionEnv, mask: torch.Tensor) -> int:
    """iEDF adapted via priority inheritance (Sha, Rajkumar & Lehoczky 1990),
    applied to the zone-blocking graph instead of a real-time resource lock —
    the MAIN adapted-iEDF candidate (supersedes edf_discounted_select above,
    kept for reference/ablation). Motivation: under blocking, an urgent
    vehicle stuck behind another vehicle's zone occupancy gains nothing from
    a high priority tag while it remains infeasible — the fix is to
    temporarily lend that urgency to whichever vehicle must move to release
    the blockage, not to change anyone's own tag.

    The original fixed tag is untouched:
        d_i^k = a_i + sum_{j<k} p_i^j
    Each vehicle i has b_i = d_i^{k_i}, its tag at its own next PENDING
    (not-yet-LOCKED) operation. Build a dependency graph over vehicles: edge
    i -> u means "i must release a resource before u can advance" — i.e. i
    is immediately ahead of u in a zone's queue, or i is that zone's current
    LOCKED occupant with u heading its tentative queue. Both sources are
    genuine blocking dependencies, not mere future-zone co-membership: a
    zone's tentative queue is the scheduler's own already-committed (if
    still revisable) relative order for that zone, and a LOCKED occupant is
    an actual, physically-enforced resource hold — never an edge built from
    two vehicles simply routing through the same zone at different times.

    Edges are read directly off the CURRENT tentative schedule
    (env._zone_queue), not only currently-LOCKED occupants, and this
    matters: operations lock based on the VEHICLE's arrival proximity
    (dynamic_intersection.py's _lock_near_vehicles), not each operation's
    own position, so a downstream operation can go irrevocable well before
    its vehicle physically nears the contested zone. Propagating through the
    tentative structure as it's built catches that dependency before it
    locks and becomes unreachable; restricting to only-currently-LOCKED
    occupants would act too late in exactly that case. LOCKED operations are
    never edge sources being revised here, only real resource-holds feeding
    the graph — every existing locked decision is left untouched.

    Effective priority:
        d~_i(t) = min({b_i} union {b_u : u transitively depends on i})
    computed via memoized DFS over the dependency graph (guaranteed acyclic
    by the same deadlock check that already applies to the ordering graph).
    Rank feasible candidates by increasing d~_i, tie-broken by the vehicle's
    own tag, vehicle id, and route position.

    Unlike edf_discounted_select, this has no tunable constant — b_i, the
    graph, and d~_i are all read directly off the environment's own
    zone-queue/lock state, nothing swept or fit. The tradeoff is the
    opposite of edf_discounted_select's: it gives up iEDF's
    queue-state-independence (d~_i depends on live blocking relationships,
    not just each vehicle's own detection-time data) in exchange for not
    needing a hyperparameter to defend. Confirmed over 5 seeds x 15
    scenarios x 5 rates (0.20-0.80, heuristics-only): beats iGreedy at 4 of
    5 rates (aggregate total 2.2355s vs iGreedy's 2.3604s), losing only at
    rate 0.35 — the same one weak rate every tested adapted-iEDF variant
    (7+) has lost at, suggesting a real structural regime there rather than
    noise in this particular rule."""
    candidates = mask.nonzero(as_tuple=True)[0].tolist()
    if not candidates:
        return -1

    def base_tag(op: DynamicOperation) -> float:
        vehicle = env.vehicles[op.vehicle_id]
        offset = sum(vehicle.processing_times[:op.route_position])
        return vehicle.arrival_time + offset

    pending_op: Dict[int, DynamicOperation] = {}
    for o in env.operations:
        if o.state == OpState.LOCKED:
            continue
        cur = pending_op.get(o.vehicle_id)
        if cur is None or o.route_position < cur.route_position:
            pending_op[o.vehicle_id] = o
    b = {vid: base_tag(op) for vid, op in pending_op.items()}

    enables: Dict[int, List[int]] = {}

    def add_edge(source_vid: int, dependent_vid: int) -> None:
        if source_vid != dependent_vid:
            enables.setdefault(source_vid, []).append(dependent_vid)

    for queue in env._zone_queue.values():
        for a, bop in zip(queue, queue[1:]):
            add_edge(a.vehicle_id, bop.vehicle_id)
        if queue:
            head = queue[0]
            for o in env.operations:
                if o.state == OpState.LOCKED and o.zone_id == head.zone_id:
                    add_edge(o.vehicle_id, head.vehicle_id)

    cache: Dict[int, float] = {}
    visiting: set = set()

    def d_tilde(vid: int) -> float:
        if vid in cache:
            return cache[vid]
        if vid in visiting:
            return b.get(vid, float("inf"))  # defensive: DAG guarantees this never fires
        visiting.add(vid)
        best = b.get(vid, float("inf"))
        for dep in enables.get(vid, []):
            dv = d_tilde(dep)
            if dv < best:
                best = dv
        visiting.discard(vid)
        cache[vid] = best
        return best

    best_i, best_key = -1, None
    for i in candidates:
        op = env.operations[i]
        vid = op.vehicle_id
        eff = d_tilde(vid)
        key = (eff, base_tag(op), vid, op.route_position)
        if best_key is None or key < best_key:
            best_i, best_key = i, key
    return best_i


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
        "backpressure": backpressure_select,
        "edf": edf_select,
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
        baselines = ["igreedy", "lifo", "backpressure", "edf"]
        imps = {
            name: (means[name]["wt"] - means["hgt"]["wt"]) / (means[name]["wt"] + 1e-9) * 100.0
            for name in baselines
        }
        wt_line = "  ".join(
            f"{name.capitalize()}={means[name]['wt']:.3f}/{means[name]['comp']:.2f}"
            for name in baselines
        )
        imp_line = "  ".join(f"vs {name.capitalize()} {imps[name]:+.1f}%" for name in baselines)
        print(
            f"[{tier:6s}]  {wt_line}  HGT={means['hgt']['wt']:.3f}/{means['hgt']['comp']:.2f}  "
            f"| HGT {imp_line}"
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
    baselines = ["igreedy", "lifo", "backpressure", "edf"]
    overall_imps = {
        name: (overall[name] - overall["hgt"]) / (overall[name] + 1e-9) * 100.0
        for name in baselines
    }
    overall_wt_line = "  ".join(f"{name.capitalize()}={overall[name]:.3f}" for name in baselines)
    overall_imp_line = "  ".join(f"vs {name.capitalize()} {overall_imps[name]:+.1f}%" for name in baselines)
    print(
        f"[{'overall':6s}]  {overall_wt_line}  HGT={overall['hgt']:.3f}  "
        f"| HGT {overall_imp_line}"
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
