"""Exact JSSP solver via OR-Tools CP-SAT, used as an optimality gap baseline.

Models the same three constraint types as the trained env/feasibility checker:
  Type-1 (route order): operations of a vehicle happen in route order.
  Type-2 (same-lane order): among vehicles entering from the same lane, the
    earlier-arriving vehicle's first operation must finish before the later
    vehicle's first operation starts (matches
    intersection_scheduler.environment.feasibility._same_lane_leaders_done).
  Type-3 (zone exclusivity): no two vehicles occupy the same zone at once.
"""

from __future__ import annotations

from typing import List, Optional

from ortools.sat.python import cp_model

from intersection_scheduler.data.scenario_generator import _MANOEUVRE_TO_LANE
from intersection_scheduler.environment.intersection import Vehicle

# Scale seconds -> integer ticks for CP-SAT. At 1e5 the resolution is 0.01ms,
# small enough that rounding the continuous processing times to integers does
# not let the exact solver's W* drift measurably from the real (float) env —
# at SCALE=1000 (1ms) rounding produced spurious ~0.5ms negative gaps where
# HGT appeared to beat the "optimal".
SCALE = 100_000


def solve_optimal(
    vehicles: List[Vehicle],
    manoeuvres: Optional[List[str]] = None,
    time_limit_seconds: float = 60.0,
    return_schedule: bool = False,
):
    """Solve for the exact minimum mean waiting time W* via CP-SAT.

    Returns W* in seconds (mean waiting time across vehicles, matching
    episode_waiting_time's convention), or None if no solution was found
    within the time limit.

    If return_schedule is True, returns (W*, schedule) where schedule is a
    list of dicts {vehicle_id, route_position, zone_id, start, end} in seconds
    for inspecting/verifying the solved timings. schedule is None if unsolved.
    """
    if not vehicles:
        return (0.0, []) if return_schedule else 0.0

    model = cp_model.CpModel()

    horizon = int(
        SCALE * (
            max(v.arrival_time for v in vehicles)
            + sum(sum(v.processing_times) for v in vehicles)
            + 1.0
        )
    )

    # start_vars[(vehicle_id, route_position)] -> IntVar
    start_vars = {}
    end_vars = {}
    intervals_by_zone: dict = {}
    # Per-zone op records for the non-delay (no idle-insertion) constraint:
    # (op_key, start_var, end_var). ready_lb is start's lower bound = arrival
    # (or predecessor finish, handled via the Type-1 constraint chain).
    ops_by_zone: dict = {}
    # Per-vehicle scaled durations, kept so min_finish uses exactly the same
    # rounded integers as the interval variables (avoids a ~1ms rounding
    # artifact where round(sum(p)) != sum(round(p))).
    scaled_durations: dict = {}

    for v in vehicles:
        arrival = int(round(v.arrival_time * SCALE))
        durs = []
        for j, (zone_id, p) in enumerate(zip(v.route, v.processing_times)):
            dur = max(1, int(round(p * SCALE)))
            durs.append(dur)
            start = model.NewIntVar(arrival, horizon, f"s_{v.id}_{j}")
            end = model.NewIntVar(arrival, horizon, f"e_{v.id}_{j}")
            interval = model.NewIntervalVar(start, dur, end, f"iv_{v.id}_{j}")

            start_vars[(v.id, j)] = start
            end_vars[(v.id, j)] = end
            intervals_by_zone.setdefault(zone_id, []).append(interval)
            ops_by_zone.setdefault(zone_id, []).append((v.id, j, start, end))

            # Vehicle cannot start before it arrives (position 0) — already
            # enforced by the variable's lower bound above.
            if j == 0:
                model.Add(start >= arrival)
        scaled_durations[v.id] = durs

        # Type-1: route order — op(i,j+1) starts no earlier than op(i,j) ends
        for j in range(len(v.route) - 1):
            model.Add(start_vars[(v.id, j + 1)] >= end_vars[(v.id, j)])

    # Type-2: same-lane order on the first operation only (mirrors
    # _same_lane_leaders_done, which only gates route_position == 0).
    if manoeuvres is not None:
        by_lane: dict = {}
        for v, m in zip(vehicles, manoeuvres):
            lane = _MANOEUVRE_TO_LANE.get(m)
            if lane is None:
                continue
            by_lane.setdefault(lane, []).append(v)

        for lane_vehicles in by_lane.values():
            ordered = sorted(lane_vehicles, key=lambda v: (v.arrival_time, v.id))
            for k in range(len(ordered) - 1):
                leader, follower = ordered[k], ordered[k + 1]
                model.Add(
                    start_vars[(follower.id, 0)] >= end_vars[(leader.id, 0)]
                )

    # Type-3: zone exclusivity — no two vehicles overlap in the same zone
    for zone_id, intervals in intervals_by_zone.items():
        if len(intervals) > 1:
            model.AddNoOverlap(intervals)

    # Objective: minimise total delay = sum over vehicles of
    # (end_time(i, last_zone) - arrival_time(i) - sum(processing_times(i)))
    delay_terms = []
    for v in vehicles:
        last_j = len(v.route) - 1
        # Use the same rounded per-op durations as the interval vars so that
        # a conflict-free schedule yields exactly zero delay (matching the
        # env's episode_waiting_time, which subtracts sum(processing_times)).
        arrival = int(round(v.arrival_time * SCALE))
        min_finish = arrival + sum(scaled_durations[v.id])
        delay = model.NewIntVar(0, horizon, f"delay_{v.id}")
        model.Add(delay >= end_vars[(v.id, last_j)] - min_finish)
        delay_terms.append(delay)

    total_delay = model.NewIntVar(0, horizon * len(vehicles), "total_delay")
    model.Add(total_delay == sum(delay_terms))
    model.Minimize(total_delay)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.num_search_workers = 8

    status = solver.Solve(model)

    # Only OPTIMAL is a valid W* — FEASIBLE means the solver found a valid
    # schedule but did not prove it minimal within the time limit, which
    # would let a suboptimal (too-high) value masquerade as ground truth
    # and make HGT/iGreedy look artificially better or worse than they are.
    if status != cp_model.OPTIMAL:
        return (None, None) if return_schedule else None

    total_delay_seconds = solver.Value(total_delay) / SCALE
    w_star = total_delay_seconds / len(vehicles)

    if not return_schedule:
        return w_star

    schedule = []
    for v in vehicles:
        for j, zone_id in enumerate(v.route):
            schedule.append({
                "vehicle_id": v.id,
                "route_position": j,
                "zone_id": zone_id,
                "start": solver.Value(start_vars[(v.id, j)]) / SCALE,
                "end": solver.Value(end_vars[(v.id, j)]) / SCALE,
            })
    return w_star, schedule


def compute_gap(w_hgt: float, w_star: Optional[float]) -> Optional[float]:
    """Return the optimality gap of W_hgt relative to W* as a percentage."""
    if w_star is None or w_star < 1e-6:
        return None
    return (w_hgt - w_star) / w_star * 100.0
