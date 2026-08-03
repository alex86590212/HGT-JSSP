"""Restricted (information-honest) optimal solver for the online scheduler.

The offline optimality gap (evaluation/optimal_solver.py) hands CP-SAT the
FULL vehicle batch upfront and lets it freely reorder anyone against anyone
across the whole horizon — a valid W* for the offline problem, where the
scheduler genuinely has that information.

The online policy never has it: it only sees a vehicle once it's within
detection_window of arriving, and once a vehicle is within commit_window its
plan is frozen (LOCKED). Solving the offline way here would score the online
policy against a target no online method — not HGT, not iGreedy — could ever
reach, conflating "cost of not knowing the future" with "cost of a worse
decision." That's not what DATE (Huang et al. 2023) does either: their
grouping strategy solves OR-Tools optimally per fixed batch, then commits it
before the next batch arrives — optimal, but under the same information
boundary the policy operates under.

This module is the continuous-time analog: re-solve at EVERY detection event
(matching this env's actual event-driven replanning, finer-grained than
DATE's discrete batching), using only currently-detected vehicles, with
already-LOCKED operations passed in as FIXED windows the solver cannot move
— exactly the constraint the online policy itself is under at that instant.
Each solve's answer is committed back as LOCKED windows for any operation
that crosses commit_window before the NEXT re-solve, so later solves see the
same accumulating, frozen history any online policy would — and each
vehicle's reported finish time is its LAST solved value before it locked,
directly comparable to HGT/iGreedy's own completed_log convention.
"""

from __future__ import annotations

import copy
import os
from typing import Dict, List, Optional, Tuple

from ortools.sat.python import cp_model

# CP-SAT's parallel workers can hang/severely thrash on a machine with fewer
# real cores than requested (confirmed directly: num_search_workers=8 hung
# indefinitely — 0.6s CPU time over 18+s wall time — on this single-CPU dev
# laptop, while num_search_workers=1 solved the same trivial model in 0ms).
# os.cpu_count() is NOT a safe stand-in (reports 10 here despite this being
# a single-CPU machine per direct instruction — likely counting logical/
# virtualized cores that aren't actually available for parallel work), so
# default conservatively to 1 and let call sites raise it explicitly on
# hardware where it's confirmed safe (e.g. the cluster).
DEFAULT_SEARCH_WORKERS = 1

from dynamic_scheduler.environment.dynamic_intersection import DynamicVehicle
from intersection_scheduler.data.scenario_generator_4x4 import _MANOEUVRE_TO_LANE

# Same rationale/value as evaluation/optimal_solver.py: fine enough that
# rounding continuous times to integers doesn't let a solved schedule drift
# measurably from the real (float) env.
SCALE = 100_000


def _solve_restricted(
    detected: List[DynamicVehicle],
    locked_windows: Dict[Tuple[int, int], Tuple[float, float]],
    current_time: float,
    time_limit_seconds: float,
    num_search_workers: int = DEFAULT_SEARCH_WORKERS,
) -> Optional[Dict[Tuple[int, int], Tuple[float, float]]]:
    """Solve one restricted-information instance.

    detected: vehicles currently known to the scheduler (arrival_time may be
        in the future — matches the online env, which detects a vehicle
        detection_window seconds before it actually arrives).
    locked_windows: {(vehicle_id, route_position): (start, end)} for every
        operation already LOCKED — the solver treats these as fixed, exactly
        like the online env (a LOCKED op is never revised).
    current_time: "now" — nothing may be scheduled to start before this,
        matching the online env's own timing rule.

    Returns {(vehicle_id, route_position): (start, end)} for every detected
    operation (locked ones echoed back unchanged), or None if CP-SAT didn't
    prove optimality within the time limit.
    """
    if not detected:
        return {}

    model = cp_model.CpModel()
    horizon = int(
        SCALE * (
            max(v.arrival_time for v in detected)
            + sum(sum(v.processing_times) for v in detected)
            + max((e for _, e in locked_windows.values()), default=0.0)
            + 1.0
        )
    )

    start_vars: Dict[Tuple[int, int], object] = {}
    end_vars: Dict[Tuple[int, int], object] = {}
    intervals_by_zone: Dict[int, list] = {}
    now = int(round(current_time * SCALE))

    for v in detected:
        arrival = int(round(v.arrival_time * SCALE))
        for j, (zone_id, p) in enumerate(zip(v.route, v.processing_times)):
            key = (v.id, j)
            dur = max(1, int(round(p * SCALE)))

            if key in locked_windows:
                # Frozen: fixed-length interval at the exact locked window,
                # not a decision variable — the oracle cannot move it, same
                # as the online policy cannot revise a LOCKED op.
                w_start, w_end = locked_windows[key]
                s = int(round(w_start * SCALE))
                e = int(round(w_end * SCALE))
                start = model.NewConstant(s)
                end = model.NewConstant(e)
                interval = model.NewIntervalVar(start, e - s, end, f"iv_{v.id}_{j}")
            else:
                lb = max(arrival, now)
                start = model.NewIntVar(lb, horizon, f"s_{v.id}_{j}")
                end = model.NewIntVar(lb, horizon, f"e_{v.id}_{j}")
                interval = model.NewIntervalVar(start, dur, end, f"iv_{v.id}_{j}")

            start_vars[key] = start
            end_vars[key] = end
            intervals_by_zone.setdefault(zone_id, []).append(interval)

        # Type-1: route order (skipped between two already-locked ops — both
        # constant, and locking already guaranteed their consistency).
        for j in range(len(v.route) - 1):
            if (v.id, j) in locked_windows and (v.id, j + 1) in locked_windows:
                continue
            model.Add(start_vars[(v.id, j + 1)] >= end_vars[(v.id, j)])

    # Type-2: same-lane order on the first operation only, mirrors
    # feasibility._same_lane_leaders_planned / optimal_solver's Type-2.
    by_lane: Dict[str, List[DynamicVehicle]] = {}
    for v in detected:
        lane = _MANOEUVRE_TO_LANE.get(v.manoeuvre)
        if lane is not None:
            by_lane.setdefault(lane, []).append(v)
    for lane_vehicles in by_lane.values():
        ordered = sorted(lane_vehicles, key=lambda v: (v.arrival_time, v.id))
        for k in range(len(ordered) - 1):
            leader, follower = ordered[k], ordered[k + 1]
            if (leader.id, 0) in locked_windows and (follower.id, 0) in locked_windows:
                continue
            model.Add(start_vars[(follower.id, 0)] >= end_vars[(leader.id, 0)])

    # Type-3: zone exclusivity.
    for zone_id, intervals in intervals_by_zone.items():
        if len(intervals) > 1:
            model.AddNoOverlap(intervals)

    delay_terms = []
    for v in detected:
        last_j = len(v.route) - 1
        arrival = int(round(v.arrival_time * SCALE))
        min_finish = arrival + sum(max(1, int(round(p * SCALE))) for p in v.processing_times)
        delay = model.NewIntVar(0, horizon, f"delay_{v.id}")
        model.Add(delay >= end_vars[(v.id, last_j)] - min_finish)
        delay_terms.append(delay)

    total_delay = model.NewIntVar(0, horizon * max(len(detected), 1), "total_delay")
    model.Add(total_delay == sum(delay_terms))
    model.Minimize(total_delay)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.num_search_workers = num_search_workers
    status = solver.Solve(model)

    if status != cp_model.OPTIMAL:
        return None

    result: Dict[Tuple[int, int], Tuple[float, float]] = {}
    for v in detected:
        for j in range(len(v.route)):
            key = (v.id, j)
            result[key] = (
                solver.Value(start_vars[key]) / SCALE,
                solver.Value(end_vars[key]) / SCALE,
            )
    return result


def restricted_optimal_episode(
    arrivals: List[DynamicVehicle],
    episode_duration: float,
    detection_window: float,
    commit_window: float,
    time_limit_seconds: float = 10.0,
    num_search_workers: int = DEFAULT_SEARCH_WORKERS,
) -> Tuple[Optional[float], int, int]:
    """Information-restricted optimum for one episode's arrival stream.

    Re-solves CP-SAT at every detection event over currently-detected
    vehicles, with prior solves' LOCKED (commit_window-crossed) operations
    passed back in as fixed windows — so each re-solve sees exactly the
    frozen history an online policy would. Each vehicle's reported finish
    time is its LAST solved value before its final operation locked.

    episode_duration caps the solve exactly like the online env: only
    detection events at or before episode_duration are solved (matching
    DynamicIntersectionEnv, which stops advancing once current_time reaches
    it), and any vehicle whose final op never locked by then contributes its
    last SOLVED projected finish time — same "in-flight" convention as
    DynamicIntersectionEnv.inflight_waiting_times, so the oracle is capped by
    the identical wall-clock budget HGT/iGreedy/LIFO are scored under; it is
    not given extra time to let every route run to full completion.

    This drives detection/lock timing directly (not via DynamicIntersectionEnv,
    since it needs to write solver results back as authoritative locked
    windows rather than compute them from a policy's plan_operation calls).

    Returns (mean_waiting_time, n_solved_calls, n_unsolved_calls). If ANY
    call fails to prove optimality within time_limit_seconds, waiting_time
    is None — a partial/suboptimal oracle would understate the true W* and
    make every online method look artificially better than it is, the same
    reasoning evaluation/optimal_solver.py already applies.
    """
    arrivals = sorted(copy.deepcopy(arrivals), key=lambda v: v.arrival_time)
    detected_by_end = [v for v in arrivals if v.arrival_time - episode_duration <= detection_window + 1e-9]

    locked_windows: Dict[Tuple[int, int], Tuple[float, float]] = {}
    last_finish: Dict[Tuple[int, int], Tuple[float, float]] = {}
    finalized_finish: Dict[int, float] = {}  # vehicle_id -> its last op's finish, once locked

    n_solved = 0
    n_unsolved = 0

    def detection_events() -> List[float]:
        # A vehicle is detected detection_window before it arrives, and locks
        # commit_window before it arrives (matches
        # DynamicIntersectionEnv.advance_time, which folds BOTH kinds of
        # timestamp into its next-event candidates — not just detections of
        # OTHER vehicles). Missing the commit-transition times here was a
        # real bug: a vehicle detected early with no other arrival nearby
        # would never get re-solved again until some unrelated vehicle's
        # detection happened to land after its arrival, letting the NEXT
        # solve treat "now" as its earliest bound instead of its true
        # already-passed arrival time — inflating W* by over 10s in testing.
        # Also fold in t=0 for anything already within range at episode
        # start. Capped at episode_duration: the online env never advances
        # current_time past it, so no re-solve should be scored using
        # information from beyond the same cutoff.
        times = {max(0.0, v.arrival_time - detection_window) for v in arrivals}
        times |= {max(0.0, v.arrival_time - commit_window) for v in arrivals}
        times = {t for t in times if t <= episode_duration + 1e-9}
        times.add(episode_duration)  # final snapshot, for the in-flight tally
        return sorted(times)

    events = detection_events()
    vid_to_vehicle = {v.id: v for v in arrivals}

    for t in events:
        detected = [v for v in arrivals if v.arrival_time - t <= detection_window + 1e-9]
        if not detected:
            continue

        result = _solve_restricted(detected, locked_windows, t, time_limit_seconds, num_search_workers)
        if result is None:
            n_unsolved += 1
            continue
        n_solved += 1
        last_finish = result

        # Commit: any operation whose vehicle is now within commit_window of
        # arrival (mirrors DynamicIntersectionEnv._lock_near_vehicles) locks
        # at this solve's answer and is frozen for all future re-solves.
        for v in detected:
            if v.arrival_time - t <= commit_window + 1e-9:
                for j in range(len(v.route)):
                    key = (v.id, j)
                    if key not in locked_windows and key in result:
                        locked_windows[key] = result[key]
                last_j = len(v.route) - 1
                if (v.id, last_j) in locked_windows:
                    finalized_finish[v.id] = locked_windows[(v.id, last_j)][1]

    if n_unsolved > 0:
        return None, n_solved, n_unsolved

    # Any vehicle never locked by episode_duration (detected but the episode
    # ended first, or never detected at all) contributes its last SOLVED
    # projected finish time — same in-flight convention as
    # DynamicIntersectionEnv.inflight_waiting_times, which reports the
    # currently-planned finish at the cutoff snapshot, not a hypothetical
    # full completion. A never-detected vehicle has no plan at all, so its
    # own arrival + route processing time floor stands in (zero delay,
    # consistent with inflight_waiting_times never going negative).
    for v in detected_by_end:
        if v.id not in finalized_finish:
            last_j = len(v.route) - 1
            key = (v.id, last_j)
            if key in last_finish:
                finalized_finish[v.id] = last_finish[key][1]
            else:
                finalized_finish[v.id] = v.arrival_time + sum(v.processing_times)

    if not finalized_finish:
        return 0.0, n_solved, n_unsolved

    total_delay = 0.0
    for vid, finish in finalized_finish.items():
        v = vid_to_vehicle[vid]
        min_finish = v.arrival_time + sum(v.processing_times)
        total_delay += max(0.0, finish - min_finish)

    return total_delay / len(finalized_finish), n_solved, n_unsolved
