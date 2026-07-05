"""IntersectionEnv: MDP for JSSP-based intersection scheduling."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class Vehicle:
    id: int
    arrival_time: float
    route: List[int]           # ordered zone ids
    processing_times: List[float]
    velocity: float


@dataclass
class Operation:
    vehicle_id: int
    zone_id: int
    route_position: int        # j
    route_length: int          # n_i
    processing_time: float
    scheduled: bool = False
    earliest_finish: float = 0.0   # c(i,j)


@dataclass
class Zone:
    id: int
    x: float
    y: float
    occupied: bool = False
    time_free: float = 0.0
    n_competing: int = 0


class IntersectionEnv:
    def __init__(self) -> None:
        self.vehicles: List[Vehicle] = []
        self.operations: List[Operation] = []   # flat list, index = action
        self.zones: Dict[int, Zone] = {}
        self.current_time: float = 0.0  # advances to earliest feasible start
        # Type-3 conflict edges: set of frozensets {op_idx_a, op_idx_b}
        self.conflict_edges: List[Tuple[int, int]] = []
        # Normalisation stats (set at reset)
        self.norm_max_p: float = 1.0
        self.norm_max_c: float = 1.0
        self.norm_max_r: float = 1.0
        self.norm_max_n: float = 1.0
        # For reward computation
        self._prev_completion_times: List[float] = []
        # Track which Type-3 pairs are still unresolved (both ops unscheduled)
        self.active_conflict_edges: List[Tuple[int, int]] = []

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(self, vehicles: List[Vehicle]):
        """Initialise all operations, compute initial c(i,j), return env."""
        self.vehicles = vehicles
        self.operations = []
        self.zones = {}
        self.current_time = min(v.arrival_time for v in vehicles) if vehicles else 0.0
        self.conflict_edges = []
        self.active_conflict_edges = []

        # Build operations and collect all zone ids
        zone_ids = set()
        for v in vehicles:
            for j, (zone_id, pt) in enumerate(zip(v.route, v.processing_times)):
                op = Operation(
                    vehicle_id=v.id,
                    zone_id=zone_id,
                    route_position=j,
                    route_length=len(v.route),
                    processing_time=pt,
                )
                self.operations.append(op)
                zone_ids.add(zone_id)

        # Populate zones using positions from the scenario generator if present,
        # otherwise use a simple grid placeholder.
        from intersection_scheduler.data.scenario_generator import ZONE_POSITIONS
        for zid in zone_ids:
            pos = ZONE_POSITIONS.get(zid, (0.0, 0.0))
            self.zones[zid] = Zone(id=zid, x=pos[0] / 2.0, y=pos[1] / 2.0)

        # Compute n_competing per zone
        for z in self.zones.values():
            z.n_competing = sum(
                1 for op in self.operations if op.zone_id == z.id and not op.scheduled
            )

        # Compute initial earliest_finish values (no conflicts resolved yet)
        # For each vehicle, propagate along route assuming no zone contention.
        self._init_finish_times()

        # Build Type-3 conflict edges: two ops need the same zone and aren't
        # the same vehicle (same vehicle => Type-1, not Type-3).
        self._build_conflict_edges()

        # Normalisation constants computed once at reset
        all_p = [op.processing_time for op in self.operations] or [1.0]
        all_c = [op.earliest_finish for op in self.operations] or [1.0]
        all_r = [v.arrival_time for v in vehicles] or [1.0]
        self.norm_max_p = max(all_p) or 1.0
        self.norm_max_c = max(all_c) or 1.0
        self.norm_max_r = max(all_r) or 1.0
        self.norm_max_n = float(len(vehicles)) or 1.0

        self._prev_completion_times = self._last_finish_per_vehicle()
        return self

    def _init_finish_times(self) -> None:
        """Set initial c(i,j) ignoring zone conflicts (lower bound)."""
        ops_by_vehicle: Dict[int, List[Operation]] = {}
        for op in self.operations:
            ops_by_vehicle.setdefault(op.vehicle_id, []).append(op)

        for vid, ops in ops_by_vehicle.items():
            ops_sorted = sorted(ops, key=lambda o: o.route_position)
            vehicle = next(v for v in self.vehicles if v.id == vid)
            t = vehicle.arrival_time
            for op in ops_sorted:
                t = t + op.processing_time
                op.earliest_finish = t

    def _build_conflict_edges(self) -> None:
        """Identify all Type-3 conflicts (same zone, different vehicle)."""
        self.conflict_edges = []
        n = len(self.operations)
        for i in range(n):
            for j in range(i + 1, n):
                oi, oj = self.operations[i], self.operations[j]
                if oi.zone_id == oj.zone_id and oi.vehicle_id != oj.vehicle_id:
                    self.conflict_edges.append((i, j))
        self.active_conflict_edges = list(self.conflict_edges)

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(self, action: int) -> Tuple["IntersectionEnv", float, bool]:
        """Schedule the operation at index `action`."""
        op = self.operations[action]
        assert not op.scheduled, f"Operation {action} already scheduled"

        # Record prev completion times for reward
        self._prev_completion_times = self._last_finish_per_vehicle()

        # Determine when this op can actually start
        vehicle = next(v for v in self.vehicles if v.id == op.vehicle_id)
        start = max(self.current_time, vehicle.arrival_time)

        # Respect route predecessor
        if op.route_position > 0:
            pred = self._predecessor_op(op)
            if pred is not None:
                start = max(start, pred.earliest_finish)

        # Respect zone availability
        zone = self.zones[op.zone_id]
        start = max(start, zone.time_free)

        finish = start + op.processing_time
        op.earliest_finish = finish
        op.scheduled = True

        # Update zone state
        zone.occupied = True
        zone.time_free = finish
        zone.n_competing -= 1

        # Remove resolved Type-3 edges (this op's conflicts are now resolved)
        self.active_conflict_edges = [
            (a, b) for (a, b) in self.active_conflict_edges
            if a != action and b != action
        ]

        # Advance simulation time to when this op starts
        self.current_time = start

        # Propagate finish times downstream through Type-1 edges
        self._propagate_finish_times(op)

        # Update n_competing for all zones
        for z in self.zones.values():
            z.n_competing = sum(
                1 for o in self.operations
                if o.zone_id == z.id and not o.scheduled
            )

        # Compute reward
        reward = self._compute_reward()

        done = all(o.scheduled for o in self.operations)
        return self, reward, done

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _predecessor_op(self, op: Operation) -> Optional[Operation]:
        """Return the route-predecessor Operation for op, or None."""
        for o in self.operations:
            if o.vehicle_id == op.vehicle_id and o.route_position == op.route_position - 1:
                return o
        return None

    def _propagate_finish_times(self, op: Operation) -> None:
        """Forward-propagate c values from op through its vehicle's route."""
        # Collect the vehicle's ops in route order
        ops_sorted = sorted(
            [o for o in self.operations if o.vehicle_id == op.vehicle_id],
            key=lambda o: o.route_position,
        )
        # Find starting index
        start_idx = next(
            (i for i, o in enumerate(ops_sorted) if o.route_position == op.route_position),
            None,
        )
        if start_idx is None:
            return

        for i in range(start_idx, len(ops_sorted) - 1):
            cur = ops_sorted[i]
            nxt = ops_sorted[i + 1]
            if nxt.scheduled:
                continue
            zone_nxt = self.zones.get(nxt.zone_id)
            zone_avail = zone_nxt.time_free if zone_nxt else 0.0
            new_finish = max(cur.earliest_finish, zone_avail) + nxt.processing_time
            if abs(new_finish - nxt.earliest_finish) < 1e-9:
                break
            nxt.earliest_finish = new_finish

    def _last_finish_per_vehicle(self) -> List[float]:
        """c(i, n_i) for each vehicle — last op's earliest_finish."""
        result = []
        for v in self.vehicles:
            ops = [o for o in self.operations if o.vehicle_id == v.id]
            last = max(ops, key=lambda o: o.route_position)
            result.append(last.earliest_finish)
        return result

    def _compute_reward(self) -> float:
        """w(t) = -Σ_i (c(i,n_i)(t) - c(i,n_i)(t-1)).

        Penalises the increase in total completion time caused by the scheduling
        decision. Always ≤ 0: scheduling can only keep or worsen completion times.
        """
        current = self._last_finish_per_vehicle()
        reward = -sum(c - p for c, p in zip(current, self._prev_completion_times))
        reward = reward / len(self.vehicles)
        return reward

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def op_index(self, vehicle_id: int, route_position: int) -> int:
        for idx, op in enumerate(self.operations):
            if op.vehicle_id == vehicle_id and op.route_position == route_position:
                return idx
        raise ValueError(f"Operation not found: v={vehicle_id} j={route_position}")

    def ops_for_vehicle(self, vehicle_id: int) -> List[Tuple[int, Operation]]:
        """Return [(global_index, op)] sorted by route_position."""
        return sorted(
            [(i, o) for i, o in enumerate(self.operations) if o.vehicle_id == vehicle_id],
            key=lambda x: x[1].route_position,
        )
