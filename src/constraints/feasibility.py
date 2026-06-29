"""Scheduler-independent feasibility checks for intersection JSSP actions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from src.graph.conflict_zones import CONFLICT_ZONES, PROCESSING_TIME_BY_TYPE, ROUTE_TO_CONFLICT_ZONES
from src.sumo_interface.traci_runner import (
    LiveVehicleRecord,
    OperationRecord,
    OperationReservation,
    VehicleReservation,
    build_operations,
    fcfs_priority_key,
    operation_id,
)


TYPE1_VIOLATION = "type1_trajectory_order_violation"
TYPE2_VIOLATION = "same_lane_order_violation"
TYPE3_VIOLATION = "conflict_zone_order_violation"
ARRIVAL_TIME_VIOLATION = "arrival_time_violation"
ZONE_OCCUPIED = "zone_occupied"
BLOCKING_VIOLATION = "blocking_violation"
DEADLOCK_RISK = "deadlock_risk"
ALREADY_SCHEDULED = "already_scheduled"
UNKNOWN_OPERATION = "unknown_operation"


@dataclass(frozen=True)
class FeasibilityResult:
    node_id: str
    feasible: bool
    reasons: Tuple[str, ...]
    earliest_start: float
    earliest_exit: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DeadlockReport:
    has_deadlock: bool
    wait_edges: Tuple[Tuple[str, str], ...]
    cycle: Tuple[str, ...] = ()


@dataclass
class ScheduleState:
    """Mutable reservation state shared by schedulers.

    The state deliberately contains only generic scheduling facts. FCFS, a
    future greedy rule, and a future PPO-GNN action mask can all use the same
    state and feasibility checks.
    """

    zone_available_at: Dict[str, float]
    lane_available_at: Dict[str, float]
    scheduled_operations: Dict[str, OperationReservation] = field(default_factory=dict)
    reservations_by_vehicle: Dict[str, List[OperationReservation]] = field(default_factory=dict)
    vehicle_reservations: Dict[str, VehicleReservation] = field(default_factory=dict)
    zone_reservations: Dict[str, List[OperationReservation]] = field(default_factory=dict)
    scheduled_vehicle_order: List[str] = field(default_factory=list)
    held_zones_by_vehicle: Dict[str, Set[str]] = field(default_factory=dict)
    next_zone_by_vehicle: Dict[str, str] = field(default_factory=dict)

    def next_operation_index(self, vehicle_id: str) -> int:
        return len(self.reservations_by_vehicle.get(vehicle_id, []))


def same_lane_order_key(record: LiveVehicleRecord) -> Tuple[float, float, float, str]:
    """Stable same-lane order: arrival time, lane position, detection time, ID."""

    return (
        record.estimated_arrival_time,
        -record.lane_position,
        record.first_detected_time,
        record.vehicle_id,
    )


def intervals_overlap(
    first_start: float,
    first_exit: float,
    second_start: float,
    second_exit: float,
    tolerance: float = 0.0,
) -> bool:
    return min(first_exit, second_exit) - max(first_start, second_start) > tolerance


def detect_blocking_deadlock(
    held_zones_by_vehicle: Mapping[str, Iterable[str]],
    next_zone_by_vehicle: Mapping[str, str],
) -> DeadlockReport:
    """Detect cycles in vehicle wait-for dependencies.

    If vehicle A holds z1 and waits for z2, while vehicle B holds z2 and waits
    for z1, both vehicles are blocked. This conservative check is independent
    of any scheduler policy.
    """

    holder_by_zone: Dict[str, str] = {}
    for vehicle_id, zones in held_zones_by_vehicle.items():
        for zone in zones:
            holder_by_zone[zone] = vehicle_id

    wait_edges: List[Tuple[str, str]] = []
    for vehicle_id, next_zone in next_zone_by_vehicle.items():
        holder = holder_by_zone.get(next_zone)
        if holder is not None and holder != vehicle_id:
            wait_edges.append((vehicle_id, holder))

    graph: Dict[str, List[str]] = {}
    for source, target in wait_edges:
        graph.setdefault(source, []).append(target)

    visiting: Set[str] = set()
    visited: Set[str] = set()
    stack: List[str] = []

    def dfs(node: str) -> Optional[List[str]]:
        visiting.add(node)
        stack.append(node)
        for neighbor in graph.get(node, []):
            if neighbor in visiting:
                cycle_start = stack.index(neighbor)
                return stack[cycle_start:] + [neighbor]
            if neighbor not in visited:
                cycle = dfs(neighbor)
                if cycle:
                    return cycle
        visiting.remove(node)
        visited.add(node)
        stack.pop()
        return None

    for node in list(graph):
        if node in visited:
            continue
        cycle = dfs(node)
        if cycle:
            return DeadlockReport(
                has_deadlock=True,
                wait_edges=tuple(wait_edges),
                cycle=tuple(cycle),
            )

    return DeadlockReport(has_deadlock=False, wait_edges=tuple(wait_edges))


class FeasibilityChecker:
    """Central action-mask and reservation feasibility logic."""

    def __init__(
        self,
        vehicle_records: Iterable[LiveVehicleRecord],
        jssp_graph: Any = None,
        *,
        conflict_zone_clearance: float = 1.0,
        same_lane_clearance: float = 1.0,
    ) -> None:
        self.vehicle_records = sorted(vehicle_records, key=fcfs_priority_key)
        self.records_by_vehicle = {record.vehicle_id: record for record in self.vehicle_records}
        self.operations, self.operations_by_vehicle = build_operations(self.vehicle_records)
        self.operations_by_node = {operation.node_id: operation for operation in self.operations}
        self.jssp_graph = jssp_graph
        self.conflict_zone_clearance = conflict_zone_clearance
        self.same_lane_clearance = same_lane_clearance
        self.lane_order = self._build_lane_order()

    def initial_state(self) -> ScheduleState:
        return ScheduleState(
            zone_available_at={zone_id: 0.0 for zone_id in CONFLICT_ZONES},
            lane_available_at={},
            zone_reservations={zone_id: [] for zone_id in CONFLICT_ZONES},
        )

    def _build_lane_order(self) -> Dict[str, List[str]]:
        by_lane: Dict[str, List[LiveVehicleRecord]] = {}
        for record in self.vehicle_records:
            by_lane.setdefault(record.source_lane, []).append(record)
        return {
            lane: [record.vehicle_id for record in sorted(records, key=same_lane_order_key)]
            for lane, records in by_lane.items()
        }

    def same_lane_leaders(self, vehicle_id: str) -> List[str]:
        record = self.records_by_vehicle[vehicle_id]
        ordered = self.lane_order.get(record.source_lane, [])
        if vehicle_id not in ordered:
            return []
        return ordered[: ordered.index(vehicle_id)]

    def next_operations(self, state: ScheduleState) -> List[OperationRecord]:
        operations = []
        for record in self.vehicle_records:
            index = state.next_operation_index(record.vehicle_id)
            route_operations = self.operations_by_vehicle.get(record.vehicle_id, [])
            if index < len(route_operations):
                operations.append(route_operations[index])
        return operations

    def is_operation_feasible(
        self,
        operation: OperationRecord,
        schedule_state: ScheduleState,
        current_time: float,
        *,
        candidate_start: Optional[float] = None,
        allow_future_start: bool = False,
        allow_blocking_wait: bool = False,
    ) -> FeasibilityResult:
        """Evaluate whether an operation can start.

        With `allow_future_start=False`, this is an action-mask check for the
        current instant. With `allow_future_start=True`, the checker returns the
        earliest safe future start and is suitable for constructive schedulers.
        """

        if operation.node_id not in self.operations_by_node:
            return FeasibilityResult(
                node_id=operation.node_id,
                feasible=False,
                reasons=(UNKNOWN_OPERATION,),
                earliest_start=current_time,
                earliest_exit=current_time,
            )

        reasons: List[str] = []
        details: Dict[str, Any] = {}
        record = self.records_by_vehicle[operation.vehicle_id]
        start = current_time if candidate_start is None else candidate_start
        earliest_start = start

        if operation.node_id in schedule_state.scheduled_operations:
            reasons.append(ALREADY_SCHEDULED)

        expected_index = schedule_state.next_operation_index(operation.vehicle_id)
        if operation.operation_index != expected_index:
            reasons.append(TYPE1_VIOLATION)
            details["expected_operation_index"] = expected_index

        if operation.operation_index == 0:
            if earliest_start < record.estimated_arrival_time:
                reasons.append(ARRIVAL_TIME_VIOLATION)
                if allow_future_start:
                    earliest_start = record.estimated_arrival_time
            lane_ready = schedule_state.lane_available_at.get(record.source_lane, 0.0)
            if earliest_start < lane_ready:
                reasons.append(TYPE2_VIOLATION)
                details["lane_available_at"] = lane_ready
                if allow_future_start:
                    earliest_start = lane_ready
            for leader_id in self.same_lane_leaders(operation.vehicle_id):
                if leader_id not in schedule_state.reservations_by_vehicle:
                    reasons.append(TYPE2_VIOLATION)
                    details.setdefault("pending_same_lane_leaders", []).append(leader_id)
                    break
        else:
            previous_node_id = operation_id(
                operation.vehicle_id,
                ROUTE_TO_CONFLICT_ZONES[operation.route_id][operation.operation_index - 1],
                operation.operation_index - 1,
            )
            previous = schedule_state.scheduled_operations.get(previous_node_id)
            if previous is None:
                reasons.append(TYPE1_VIOLATION)
            elif earliest_start < previous.exit_time:
                reasons.append(TYPE1_VIOLATION)
                details["previous_exit_time"] = previous.exit_time
                if allow_future_start:
                    earliest_start = previous.exit_time
            elif previous.exit_time + 1e-9 < earliest_start:
                held_zones = schedule_state.held_zones_by_vehicle.get(operation.vehicle_id, set())
                if not allow_blocking_wait and previous.conflict_zone not in held_zones:
                    reasons.append(BLOCKING_VIOLATION)

        zone_ready = schedule_state.zone_available_at.get(operation.conflict_zone, 0.0)
        if earliest_start < zone_ready:
            reasons.append(ZONE_OCCUPIED)
            details["zone_available_at"] = zone_ready
            if allow_future_start:
                earliest_start = zone_ready

        for reservation in schedule_state.zone_reservations.get(operation.conflict_zone, []):
            candidate_exit = earliest_start + operation.processing_time
            if intervals_overlap(
                earliest_start,
                candidate_exit,
                reservation.enter_time,
                reservation.exit_time + self.conflict_zone_clearance,
            ):
                reasons.append(ZONE_OCCUPIED)
                details.setdefault("overlapping_reservations", []).append(reservation.node_id)
                if allow_future_start:
                    earliest_start = max(
                        earliest_start,
                        reservation.exit_time + self.conflict_zone_clearance,
                    )

        held = dict(schedule_state.held_zones_by_vehicle)
        next_zone = dict(schedule_state.next_zone_by_vehicle)
        held.setdefault(operation.vehicle_id, set())
        if operation.operation_index > 0:
            previous_zone = ROUTE_TO_CONFLICT_ZONES[operation.route_id][operation.operation_index - 1]
            held[operation.vehicle_id] = set(held.get(operation.vehicle_id, set())) | {previous_zone}
        next_zone[operation.vehicle_id] = operation.conflict_zone
        deadlock = detect_blocking_deadlock(held, next_zone)
        if deadlock.has_deadlock:
            reasons.append(DEADLOCK_RISK)
            details["deadlock_cycle"] = list(deadlock.cycle)

        if allow_future_start:
            structural_reasons = {
                ALREADY_SCHEDULED,
                UNKNOWN_OPERATION,
                DEADLOCK_RISK,
                BLOCKING_VIOLATION,
            }
            # Time and lane delays are resolved by earliest_start in planning
            # mode, but true order/cycle violations remain infeasible.
            unresolved = [
                reason
                for reason in reasons
                if reason in structural_reasons or reason == TYPE1_VIOLATION and operation.operation_index != expected_index
            ]
            reasons = unresolved

        earliest_exit = earliest_start + operation.processing_time
        return FeasibilityResult(
            node_id=operation.node_id,
            feasible=not reasons,
            reasons=tuple(dict.fromkeys(reasons)),
            earliest_start=earliest_start,
            earliest_exit=earliest_exit,
            details=details,
        )

    def action_mask(
        self,
        schedule_state: ScheduleState,
        current_time: float,
        *,
        allow_future_start: bool = False,
    ) -> List[FeasibilityResult]:
        return [
            self.is_operation_feasible(
                operation,
                schedule_state,
                current_time,
                allow_future_start=allow_future_start,
            )
            for operation in self.next_operations(schedule_state)
        ]

    def plan_vehicle_route(
        self,
        vehicle_id: str,
        schedule_state: ScheduleState,
        current_time: float,
    ) -> Tuple[List[OperationReservation], List[FeasibilityResult]]:
        """Plan one full vehicle route with blocking-aware zone exits."""

        record = self.records_by_vehicle[vehicle_id]
        route_operations = self.operations_by_vehicle[vehicle_id]
        start_times: List[float] = []
        min_exit_times: List[float] = []
        feasibility_results: List[FeasibilityResult] = []

        rolling_time = current_time
        provisional_nodes: List[str] = []
        for operation in route_operations:
            base_start = max(rolling_time, schedule_state.zone_available_at.get(operation.conflict_zone, 0.0))
            if operation.operation_index == 0:
                base_start = max(
                    base_start,
                    record.estimated_arrival_time,
                    schedule_state.lane_available_at.get(record.source_lane, 0.0),
                )
            result = self.is_operation_feasible(
                operation,
                schedule_state,
                current_time,
                candidate_start=base_start,
                allow_future_start=True,
                allow_blocking_wait=True,
            )
            feasibility_results.append(result)
            if not result.feasible:
                for node_id in provisional_nodes:
                    schedule_state.scheduled_operations.pop(node_id, None)
                schedule_state.reservations_by_vehicle.pop(vehicle_id, None)
                return [], feasibility_results
            start = result.earliest_start
            min_exit = start + operation.processing_time
            start_times.append(start)
            min_exit_times.append(min_exit)
            rolling_time = min_exit

            # Temporarily mark the operation as scheduled enough for the next
            # Type-1 feasibility check. The final blocking-aware exit is fixed
            # after all start times are known.
            provisional = OperationReservation(
                node_id=operation.node_id,
                vehicle_id=operation.vehicle_id,
                vehicle_type=operation.vehicle_type,
                route_id=operation.route_id,
                source_lane=operation.source_lane,
                conflict_zone=operation.conflict_zone,
                operation_index=operation.operation_index,
                processing_time=operation.processing_time,
                enter_time=start,
                exit_time=min_exit,
            )
            schedule_state.scheduled_operations[operation.node_id] = provisional
            schedule_state.reservations_by_vehicle.setdefault(vehicle_id, []).append(provisional)
            provisional_nodes.append(operation.node_id)

        # Remove provisional route entries before committing the finalized route.
        for node_id in provisional_nodes:
            schedule_state.scheduled_operations.pop(node_id, None)
        schedule_state.reservations_by_vehicle.pop(vehicle_id, None)

        reservations: List[OperationReservation] = []
        for index, operation in enumerate(route_operations):
            exit_time = min_exit_times[index]
            if index + 1 < len(route_operations):
                # Blocking constraint: the current zone remains occupied until
                # the next zone can be entered.
                exit_time = max(exit_time, start_times[index + 1])
            reservations.append(
                OperationReservation(
                    node_id=operation.node_id,
                    vehicle_id=operation.vehicle_id,
                    vehicle_type=operation.vehicle_type,
                    route_id=operation.route_id,
                    source_lane=operation.source_lane,
                    conflict_zone=operation.conflict_zone,
                    operation_index=operation.operation_index,
                    processing_time=operation.processing_time,
                    enter_time=start_times[index],
                    exit_time=exit_time,
                )
            )

        if self.route_has_occupancy_overlap(reservations, schedule_state):
            return [], [
                FeasibilityResult(
                    node_id=reservation.node_id,
                    feasible=False,
                    reasons=(ZONE_OCCUPIED,),
                    earliest_start=reservation.enter_time,
                    earliest_exit=reservation.exit_time,
                    details={"message": "route reservation overlaps existing zone occupancy"},
                )
                for reservation in reservations
            ]

        return reservations, feasibility_results

    def route_has_occupancy_overlap(
        self,
        reservations: Sequence[OperationReservation],
        schedule_state: ScheduleState,
    ) -> bool:
        for reservation in reservations:
            for existing in schedule_state.zone_reservations.get(reservation.conflict_zone, []):
                if intervals_overlap(
                    reservation.enter_time,
                    reservation.exit_time + self.conflict_zone_clearance,
                    existing.enter_time,
                    existing.exit_time + self.conflict_zone_clearance,
                ):
                    return True
        return False

    def commit_vehicle_route(
        self,
        vehicle_id: str,
        reservations: Sequence[OperationReservation],
        schedule_state: ScheduleState,
    ) -> None:
        if not reservations:
            return
        reservation_list = list(reservations)
        schedule_state.reservations_by_vehicle[vehicle_id] = reservation_list
        schedule_state.scheduled_vehicle_order.append(vehicle_id)
        for reservation in reservation_list:
            schedule_state.scheduled_operations[reservation.node_id] = reservation
            schedule_state.zone_reservations.setdefault(reservation.conflict_zone, []).append(reservation)
            schedule_state.zone_available_at[reservation.conflict_zone] = (
                reservation.exit_time + self.conflict_zone_clearance
            )
        first = reservation_list[0]
        last = reservation_list[-1]
        schedule_state.lane_available_at[first.source_lane] = first.exit_time + self.same_lane_clearance
        schedule_state.vehicle_reservations[vehicle_id] = VehicleReservation(
            vehicle_id=vehicle_id,
            route_id=first.route_id,
            source_lane=first.source_lane,
            estimated_arrival_time=self.records_by_vehicle[vehicle_id].estimated_arrival_time,
            enter_time=first.enter_time,
            exit_time=last.exit_time,
            operations=tuple(reservation_list),
        )

    def detect_deadlock_risk(self, schedule_state: ScheduleState) -> DeadlockReport:
        return detect_blocking_deadlock(
            schedule_state.held_zones_by_vehicle,
            schedule_state.next_zone_by_vehicle,
        )


def feasibility_report_rows(results: Iterable[FeasibilityResult]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for result in results:
        row = asdict(result)
        row["reasons"] = " ".join(result.reasons)
        rows.append(row)
    return rows


__all__ = [
    "ALREADY_SCHEDULED",
    "ARRIVAL_TIME_VIOLATION",
    "BLOCKING_VIOLATION",
    "DEADLOCK_RISK",
    "TYPE1_VIOLATION",
    "TYPE2_VIOLATION",
    "TYPE3_VIOLATION",
    "ZONE_OCCUPIED",
    "DeadlockReport",
    "FeasibilityChecker",
    "FeasibilityResult",
    "ScheduleState",
    "detect_blocking_deadlock",
    "feasibility_report_rows",
    "intervals_overlap",
    "same_lane_order_key",
]
