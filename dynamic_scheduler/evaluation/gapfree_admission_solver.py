"""Restricted-information optimal (W*) reference for the gap-free
full-route admission model (dynamic_scheduler.environment.gapfree_admission).

Structurally different from evaluation/optimal_solver_online.py, which
encodes the BLOCKING constraint (a vehicle may hold its current zone while
waiting for the next). Under gap-free admission, a vehicle's route shape is
rigid once it departs -- s_{i,k} = s_i + offset_i(k), fixed relative
offsets -- so there is exactly ONE free variable per vehicle (its departure
time s_i), not one per operation. Waiting time is then exactly s_i - a_i,
since there is no internal delay by construction once a vehicle departs.

Re-solves at every detection event using only currently-detected vehicles
(same restricted-information convention as optimal_solver_online.py), with
already-committed departures fixed as constants. A vehicle's solved
departure becomes irrevocably committed the instant it equals
max(current_time, arrival_time) -- i.e. the optimal joint solution says
"depart now" -- mirroring the moment AdmissionEnv.admit() would actually
commit it in the online heuristic policies.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

from ortools.sat.python import cp_model

from dynamic_scheduler.environment.dynamic_intersection import DynamicVehicle
from intersection_scheduler.data.scenario_generator_4x4 import _MANOEUVRE_TO_LANE

DEFAULT_SEARCH_WORKERS = 1
SCALE = 100_000


def _solve_gapfree_restricted(
    detected: List[DynamicVehicle],
    committed: Dict[int, float],
    current_time: float,
    time_limit_seconds: float,
    num_search_workers: int = DEFAULT_SEARCH_WORKERS,
) -> Optional[Dict[int, float]]:
    if not detected:
        return {}

    model = cp_model.CpModel()
    now = int(round(current_time * SCALE))
    horizon = int(
        SCALE * (
            max(v.arrival_time for v in detected)
            + sum(sum(v.processing_times) for v in detected)
            + max(committed.values(), default=0.0)
            + 1.0
        )
    )

    s_vars: Dict[int, object] = {}
    intervals_by_zone: Dict[int, list] = {}

    for v in detected:
        arrival = int(round(v.arrival_time * SCALE))
        if v.id in committed:
            s = model.NewConstant(int(round(committed[v.id] * SCALE)))
        else:
            lb = max(arrival, now)
            s = model.NewIntVar(lb, horizon, f"s_{v.id}")
        s_vars[v.id] = s

        offset = 0
        for zone_id, p in zip(v.route, v.processing_times):
            dur = max(1, int(round(p * SCALE)))
            start = s + offset
            end = start + dur
            interval = model.NewIntervalVar(start, dur, end, f"iv_{v.id}_{zone_id}_{offset}")
            intervals_by_zone.setdefault(zone_id, []).append(interval)
            offset += dur

    # Type-2: same-lane order on the whole-route departure (mirrors the
    # existing first-operation FIFO gate -- under gap-free admission a
    # vehicle's "first operation" starts exactly at s_i).
    by_lane: Dict[str, List[DynamicVehicle]] = {}
    for v in detected:
        lane = _MANOEUVRE_TO_LANE.get(v.manoeuvre)
        if lane is not None:
            by_lane.setdefault(lane, []).append(v)
    for lane_vehicles in by_lane.values():
        ordered = sorted(lane_vehicles, key=lambda v: (v.arrival_time, v.id))
        for k in range(len(ordered) - 1):
            leader, follower = ordered[k], ordered[k + 1]
            if leader.id in committed and follower.id in committed:
                continue
            leader_first_dur = max(1, int(round(leader.processing_times[0] * SCALE)))
            model.Add(s_vars[follower.id] >= s_vars[leader.id] + leader_first_dur)

    # Type-3: zone exclusivity (no blocking term needed -- see module
    # docstring: a vehicle's route is rigid once departed, so each zone's
    # occupancy is exactly [s_i+offset, s_i+offset+p) with no gap-fill).
    for zone_id, intervals in intervals_by_zone.items():
        if len(intervals) > 1:
            model.AddNoOverlap(intervals)

    delay_terms = []
    for v in detected:
        arrival = int(round(v.arrival_time * SCALE))
        delay = model.NewIntVar(0, horizon, f"delay_{v.id}")
        model.Add(delay == s_vars[v.id] - arrival)
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

    return {v.id: solver.Value(s_vars[v.id]) / SCALE for v in detected}


def restricted_gapfree_optimal_episode(
    arrivals: List[DynamicVehicle],
    episode_duration: float,
    detection_window: float,
    time_limit_seconds: float = 5.0,
    num_search_workers: int = DEFAULT_SEARCH_WORKERS,
) -> Tuple[Optional[float], int, int]:
    """Returns (mean_waiting_time, n_solved_calls, n_unsolved_calls), same
    convention as optimal_solver_online.restricted_optimal_episode. None if
    any re-solve fails to prove optimality within time_limit_seconds.

    Event-driven with a retry heap, NOT a fixed pre-computed event list: the
    `now` lower bound on an uncommitted vehicle's departure is a HARD
    constraint in the CP-SAT model (s_i >= max(arrival, now)), so solving
    later than a vehicle's true earliest feasible departure artificially
    FORCES its committed delay upward -- worse than the analogous "missing
    commit-transition times" bug in optimal_solver_online.py, where `now`
    only ever bounded per-operation lower bounds, not a hard departure
    floor. After every solve, each still-uncommitted vehicle's OWN solved
    departure is pushed back as a future retry event, so it is re-checked
    almost exactly when it becomes truly feasible rather than at whatever
    unrelated event happens to fire next."""
    import heapq

    arrivals = sorted(copy.deepcopy(arrivals), key=lambda v: v.arrival_time)
    detected_by_end = [v for v in arrivals if v.arrival_time - episode_duration <= detection_window + 1e-9]

    committed: Dict[int, float] = {}
    last_solved: Dict[int, float] = {}
    n_solved = 0
    n_unsolved = 0

    events: List[float] = []
    for v in arrivals:
        heapq.heappush(events, max(0.0, v.arrival_time - detection_window))
    heapq.heappush(events, episode_duration)

    processed: set = set()
    while events:
        t = heapq.heappop(events)
        if t in processed or t > episode_duration + 1e-9:
            continue
        processed.add(t)

        detected = [v for v in arrivals if v.arrival_time - t <= detection_window + 1e-9]
        if not detected:
            continue

        result = _solve_gapfree_restricted(detected, committed, t, time_limit_seconds, num_search_workers)
        if result is None:
            n_unsolved += 1
            continue
        n_solved += 1
        last_solved = result

        for v in detected:
            if v.id in committed:
                continue
            s = result[v.id]
            # Compare at the SAME integer-scale resolution the solver used
            # internally (arrival/now are both int(round(x*SCALE)) inside
            # _solve_gapfree_restricted): a raw-float tolerance here can be
            # tighter than SCALE's own rounding error (up to 0.5/SCALE),
            # which was silently missing legitimate "depart now" solutions
            # and letting `now`'s hard floor permanently inflate delay once
            # the missed moment passed.
            s_scaled = int(round(s * SCALE))
            floor_scaled = max(int(round(t * SCALE)), int(round(v.arrival_time * SCALE)))
            if s_scaled <= floor_scaled:
                committed[v.id] = s
            elif s <= episode_duration + 1e-9:
                heapq.heappush(events, s)

    if n_unsolved > 0:
        return None, n_solved, n_unsolved

    total_delay = 0.0
    counted = 0
    for v in detected_by_end:
        s = committed.get(v.id, last_solved.get(v.id, v.arrival_time))
        total_delay += max(0.0, s - v.arrival_time)
        counted += 1

    if counted == 0:
        return 0.0, n_solved, n_unsolved
    return total_delay / counted, n_solved, n_unsolved
