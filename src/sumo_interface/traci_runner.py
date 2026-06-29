"""Run a simple SUMO -> JSSP graph -> FCFS scheduler -> SUMO control loop.

This is a deliberately small closed-loop demo. It observes vehicles on the
single-intersection SUMO network, turns currently approaching vehicles into a
JSSP-style timing-conflict graph, applies a first-come-first-served scheduler,
and uses TraCI stop/release commands to meter vehicles into the intersection.

The FCFS scheduler is only a placeholder. A future PPO-GNN scheduler can replace
`fcfs_schedule()` while keeping the same vehicle records and graph construction.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

try:
    import networkx as nx
except ModuleNotFoundError:
    nx = None

try:
    import traci
except ModuleNotFoundError:
    traci = None

from src.graph.conflict_zones import (
    CONFLICT_ZONES,
    PROCESSING_TIME_BY_TYPE,
    ROUTE_TO_CONFLICT_ZONES as STATIC_ROUTE_TO_CONFLICT_ZONES,
)


DEFAULT_SUMO_CONFIG = Path("sumo") / "single_intersection" / "single_intersection.sumocfg"
DEFAULT_OUTPUT_DIR = Path("outputs") / "fcfs_controller"

INCOMING_LANES = {"N_in_0", "S_in_0", "E_in_0", "W_in_0"}
INCOMING_EDGES = {"N_in", "S_in", "E_in", "W_in"}
STOP_DURATION_SECONDS = 3600.0
SUMO_DEFAULT_SPEED_MODE = 31
# Keep SUMO's normal safe-speed/collision checks active for released vehicles.
# The scheduler chooses who may enter; SUMO still handles low-level car-following.
SCHEDULED_VEHICLE_SPEED_MODE = SUMO_DEFAULT_SPEED_MODE
EDGE_RELEASE_SPEED_LIMIT_MPS = 8.0
RELEASE_SPEED_FALLBACK_MPS = 8.0
LANE_END_NUDGE_DISTANCE_M = 0.25
LANE_END_NUDGE_MAX_SPEED_MPS = 0.1
LANE_END_NUDGE_POSITION_M = 0.1
DEFAULT_CONFLICT_ZONE_CLEARANCE_SECONDS = 1.0
DEFAULT_SAME_LANE_CLEARANCE_SECONDS = 1.0
DEFAULT_RELEASE_LOOKAHEAD_SECONDS = 0.0

# The static JSSP demo used the eight routes requested in the first example.
# The generated SUMO demand also contains the four remaining left turns, so they
# are added here to keep every SUMO route controllable.
ROUTE_TO_CONFLICT_ZONES: Dict[str, List[str]] = {
    **STATIC_ROUTE_TO_CONFLICT_ZONES,
    "E_to_S": ["z1", "z2", "z3"],
    "N_to_E": ["z2", "z3", "z4"],
    "W_to_N": ["z3", "z4", "z1"],
    "S_to_W": ["z4", "z1", "z2"],
}


@dataclass(frozen=True)
class LiveVehicleRecord:
    vehicle_id: str
    route_id: str
    vehicle_type: str
    lane_id: str
    source_lane: str
    source_direction: str
    lane_position: float
    speed: float
    distance_to_intersection: float
    estimated_arrival_time: float
    first_detected_time: float


@dataclass(frozen=True)
class OperationRecord:
    node_id: str
    vehicle_id: str
    vehicle_type: str
    route_id: str
    source_lane: str
    conflict_zone: str
    operation_index: int
    estimated_arrival_time: float
    processing_time: float


@dataclass(frozen=True)
class EdgeRecord:
    edge_type: str
    source: str
    target: str
    description: str
    conflict_zone: Optional[str] = None
    source_lane: Optional[str] = None
    candidate_pair_id: Optional[str] = None


@dataclass(frozen=True)
class OperationReservation:
    node_id: str
    vehicle_id: str
    vehicle_type: str
    route_id: str
    source_lane: str
    conflict_zone: str
    operation_index: int
    processing_time: float
    enter_time: float
    exit_time: float


@dataclass(frozen=True)
class VehicleReservation:
    vehicle_id: str
    route_id: str
    source_lane: str
    estimated_arrival_time: float
    enter_time: float
    exit_time: float
    operations: Tuple[OperationReservation, ...]


@dataclass(frozen=True)
class FCFSReservationSchedule:
    """Conflict-zone-aware FCFS reservation schedule.

    FCFS still defines the priority order by estimated vehicle arrival time.
    The schedule itself reserves individual conflict zones, so independent
    routes can overlap instead of locking the whole intersection for one job.
    """

    vehicle_order: List[str]
    operation_reservations: List[OperationReservation]
    reservations_by_vehicle: Dict[str, Tuple[OperationReservation, ...]]
    vehicle_reservations: Dict[str, VehicleReservation]
    conflict_zone_clearance: float
    same_lane_clearance: float
    feasibility_report: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ControllerState:
    first_eta_by_vehicle: Dict[str, float]
    first_detected_by_vehicle: Dict[str, float]
    stopped_vehicles: Set[str]
    released_vehicles: Set[str]
    cleared_vehicles: Set[str]
    warned_unmapped_routes: Set[str]
    active_vehicle_ids: Set[str] = field(default_factory=set)
    active_vehicle_zones: Dict[str, Set[str]] = field(default_factory=dict)
    active_vehicle_source_lanes: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class TraCIRunnerResult:
    """Compact run summary used by headless experiment code."""

    sim_time: float
    vehicles_completed: int
    cleared_vehicles: Tuple[str, ...]
    number_of_stops: int
    total_delay: float
    total_waiting_time: float
    maximum_waiting_time: float
    class_delay: Dict[str, float]
    scheduler_runtime_avg_ms: float
    scheduler_runtime_max_ms: float
    sumo_collisions: int
    step_count: int


class TraCIRunner:
    """Small adapter around the legacy TraCI loop.

    The controller can later receive any scheduler implementing BaseScheduler;
    for now this runner preserves the proven FCFS loop and exposes a stable
    place for experiment orchestration.
    """

    def run(self, args: argparse.Namespace) -> TraCIRunnerResult:
        return run_controller(args)


def scheduler_from_args(args: argparse.Namespace) -> Any:
    scheduler = getattr(args, "scheduler", None)
    if scheduler is not None:
        return scheduler

    from src.schedulers.fcfs_scheduler import FCFSScheduler

    return FCFSScheduler(
        conflict_zone_clearance=getattr(
            args,
            "conflict_zone_clearance",
            DEFAULT_CONFLICT_ZONE_CLEARANCE_SECONDS,
        ),
        same_lane_clearance=getattr(
            args,
            "same_lane_clearance",
            DEFAULT_SAME_LANE_CLEARANCE_SECONDS,
        ),
    )


def require_networkx() -> None:
    if nx is None:
        raise SystemExit(
            "Missing dependency: networkx. Install dependencies with "
            "`python -m pip install -r requirements.txt`."
        )


def require_dependencies() -> None:
    require_networkx()
    if traci is None:
        raise SystemExit(
            "Missing dependency: traci. Install SUMO's Python tools or install "
            "the `traci` Python package in this environment."
        )


def source_direction_from_route(route_id: str) -> str:
    return route_id.split("_to_", maxsplit=1)[0]


def source_lane_from_route(route_id: str) -> str:
    return f"{source_direction_from_route(route_id)}_in"


def edge_from_lane(lane_id: str) -> str:
    return lane_id.rsplit("_", maxsplit=1)[0]


def operation_id(vehicle_id: str, conflict_zone: str, index: int) -> str:
    return f"{vehicle_id}__{conflict_zone}__op{index}"


def estimate_arrival_time(
    vehicle_id: str,
    sim_time: float,
    distance_to_intersection: float,
    speed: float,
    state: ControllerState,
    min_eta_speed: float,
) -> float:
    """Estimate when a vehicle reaches the intersection stop line.

    The first estimate is cached so a vehicle that is stopped by this controller
    does not move to the end of the FCFS queue simply because its speed becomes
    zero.
    """

    if vehicle_id in state.first_eta_by_vehicle:
        return state.first_eta_by_vehicle[vehicle_id]

    effective_speed = max(speed, min_eta_speed)
    eta = sim_time + (distance_to_intersection / effective_speed)
    state.first_eta_by_vehicle[vehicle_id] = eta
    state.first_detected_by_vehicle[vehicle_id] = sim_time
    return eta


def get_approaching_vehicles(
    sim_time: float,
    state: ControllerState,
    detection_distance: float,
    min_eta_speed: float,
) -> List[LiveVehicleRecord]:
    """Read live vehicles from SUMO and keep the ones approaching J0.

    Vehicles are considered approaching when they are on one of the four inbound
    lanes and inside the configured detection distance.
    """

    records: List[LiveVehicleRecord] = []

    for vehicle_id in traci.vehicle.getIDList():
        lane_id = traci.vehicle.getLaneID(vehicle_id)
        if lane_id not in INCOMING_LANES:
            continue

        route_id = traci.vehicle.getRouteID(vehicle_id)
        if route_id not in ROUTE_TO_CONFLICT_ZONES:
            if route_id not in state.warned_unmapped_routes:
                print(f"Warning: route {route_id!r} has no conflict-zone mapping; skipping control.")
                state.warned_unmapped_routes.add(route_id)
            continue

        lane_position = traci.vehicle.getLanePosition(vehicle_id)
        lane_length = traci.lane.getLength(lane_id)
        distance_to_intersection = max(0.0, lane_length - lane_position)
        if distance_to_intersection > detection_distance:
            continue

        speed = traci.vehicle.getSpeed(vehicle_id)
        eta = estimate_arrival_time(
            vehicle_id=vehicle_id,
            sim_time=sim_time,
            distance_to_intersection=distance_to_intersection,
            speed=speed,
            state=state,
            min_eta_speed=min_eta_speed,
        )
        source_direction = source_direction_from_route(route_id)

        records.append(
            LiveVehicleRecord(
                vehicle_id=vehicle_id,
                route_id=route_id,
                vehicle_type=traci.vehicle.getTypeID(vehicle_id),
                lane_id=lane_id,
                source_lane=source_lane_from_route(route_id),
                source_direction=source_direction,
                lane_position=lane_position,
                speed=speed,
                distance_to_intersection=distance_to_intersection,
                estimated_arrival_time=eta,
                first_detected_time=state.first_detected_by_vehicle[vehicle_id],
            )
        )

    return sorted(
        records,
        key=lambda record: (
            record.estimated_arrival_time,
            record.first_detected_time,
            record.source_lane,
            record.vehicle_id,
        ),
    )


def build_operations(
    vehicle_records: Iterable[LiveVehicleRecord],
) -> Tuple[List[OperationRecord], Dict[str, List[OperationRecord]]]:
    operations: List[OperationRecord] = []
    operations_by_vehicle: Dict[str, List[OperationRecord]] = {}

    for record in vehicle_records:
        processing_time = PROCESSING_TIME_BY_TYPE.get(record.vehicle_type, 2.0)
        route_operations: List[OperationRecord] = []

        for index, conflict_zone in enumerate(ROUTE_TO_CONFLICT_ZONES[record.route_id]):
            operation = OperationRecord(
                node_id=operation_id(record.vehicle_id, conflict_zone, index),
                vehicle_id=record.vehicle_id,
                vehicle_type=record.vehicle_type,
                route_id=record.route_id,
                source_lane=record.source_lane,
                conflict_zone=conflict_zone,
                operation_index=index,
                estimated_arrival_time=record.estimated_arrival_time,
                processing_time=processing_time,
            )
            operations.append(operation)
            route_operations.append(operation)

        operations_by_vehicle[record.vehicle_id] = route_operations

    return operations, operations_by_vehicle


def build_jssp_graph(vehicle_records: Iterable[LiveVehicleRecord]) -> Tuple[Any, Dict[str, Any]]:
    """Convert live vehicle records into a NetworkX JSSP/timing graph."""

    require_networkx()
    records = list(vehicle_records)
    operations, operations_by_vehicle = build_operations(records)

    graph = nx.DiGraph()
    graph.graph["name"] = "live_single_intersection_jssp"
    graph.graph["conflict_zones"] = CONFLICT_ZONES
    graph.graph["route_to_conflict_zones"] = ROUTE_TO_CONFLICT_ZONES

    for operation in operations:
        graph.add_node(
            operation.node_id,
            vehicle_id=operation.vehicle_id,
            vehicle_type=operation.vehicle_type,
            route_id=operation.route_id,
            source_lane=operation.source_lane,
            conflict_zone=operation.conflict_zone,
            conflict_zone_name=CONFLICT_ZONES[operation.conflict_zone],
            operation_index=operation.operation_index,
            estimated_arrival_time=operation.estimated_arrival_time,
            processing_time=operation.processing_time,
        )

    type1_edges = add_type1_edges(graph, operations_by_vehicle)
    type2_edges = add_type2_edges(graph, operations_by_vehicle)
    type3_edges = add_type3_edges(graph, operations)

    return graph, {
        "vehicle_records": records,
        "operations": operations,
        "type1_edges": type1_edges,
        "type2_edges": type2_edges,
        "type3_edges": type3_edges,
    }


def add_type1_edges(
    graph: Any,
    operations_by_vehicle: Dict[str, List[OperationRecord]],
) -> List[EdgeRecord]:
    """Type-1 edges: route precedence inside one vehicle/job."""

    edges: List[EdgeRecord] = []
    for vehicle_id, operations in operations_by_vehicle.items():
        for previous, current in zip(operations, operations[1:]):
            graph.add_edge(
                previous.node_id,
                current.node_id,
                edge_type="type1_precedence",
                label="T1",
                vehicle_id=vehicle_id,
            )
            edges.append(
                EdgeRecord(
                    edge_type="type1_precedence",
                    source=previous.node_id,
                    target=current.node_id,
                    description=f"{vehicle_id} route order",
                )
            )
    return edges


def add_type2_edges(
    graph: Any,
    operations_by_vehicle: Dict[str, List[OperationRecord]],
) -> List[EdgeRecord]:
    """Type-2 edges: FIFO order among vehicles from the same inbound lane."""

    first_operations = [operations[0] for operations in operations_by_vehicle.values()]
    by_source_lane: Dict[str, List[OperationRecord]] = {}
    for operation in first_operations:
        by_source_lane.setdefault(operation.source_lane, []).append(operation)

    edges: List[EdgeRecord] = []
    for source_lane, lane_operations in by_source_lane.items():
        lane_operations.sort(
            key=lambda operation: (
                operation.estimated_arrival_time,
                operation.vehicle_id,
            )
        )
        for leader, follower in zip(lane_operations, lane_operations[1:]):
            graph.add_edge(
                leader.node_id,
                follower.node_id,
                edge_type="type2_same_lane_order",
                label="T2",
                source_lane=source_lane,
            )
            edges.append(
                EdgeRecord(
                    edge_type="type2_same_lane_order",
                    source=leader.node_id,
                    target=follower.node_id,
                    description=f"{source_lane} arrival order",
                    source_lane=source_lane,
                )
            )
    return edges


def add_type3_edges(graph: Any, operations: Iterable[OperationRecord]) -> List[EdgeRecord]:
    """Type-3 edges: paired candidate orderings for shared conflict zones."""

    by_zone: Dict[str, List[OperationRecord]] = {}
    for operation in operations:
        by_zone.setdefault(operation.conflict_zone, []).append(operation)

    edges: List[EdgeRecord] = []
    pair_index = 0
    for conflict_zone, zone_operations in by_zone.items():
        zone_operations.sort(
            key=lambda operation: (
                operation.estimated_arrival_time,
                operation.vehicle_id,
            )
        )
        for first, second in combinations(zone_operations, 2):
            if first.vehicle_id == second.vehicle_id:
                continue
            if first.source_lane == second.source_lane:
                continue

            pair_id = f"live_conflict_pair_{pair_index:04d}"
            pair_index += 1
            for source, target in [(first, second), (second, first)]:
                graph.add_edge(
                    source.node_id,
                    target.node_id,
                    edge_type="type3_conflict_candidate",
                    label="T3",
                    conflict_zone=conflict_zone,
                    candidate_pair_id=pair_id,
                )
            edges.append(
                EdgeRecord(
                    edge_type="type3_conflict_candidate_pair",
                    source=first.node_id,
                    target=second.node_id,
                    description=f"shared {conflict_zone}; scheduler chooses one direction",
                    conflict_zone=conflict_zone,
                    candidate_pair_id=pair_id,
                )
            )
    return edges


def fcfs_priority_key(record: LiveVehicleRecord) -> Tuple[float, float, str, str]:
    return (
        record.estimated_arrival_time,
        record.first_detected_time,
        record.source_lane,
        record.vehicle_id,
    )


def fcfs_vehicle_order(vehicle_records: Iterable[LiveVehicleRecord]) -> List[str]:
    """Return the FCFS priority order by first estimated arrival."""

    return [record.vehicle_id for record in sorted(vehicle_records, key=fcfs_priority_key)]


def fcfs_schedule(
    vehicle_records: Iterable[LiveVehicleRecord],
    *,
    conflict_zone_clearance: float = DEFAULT_CONFLICT_ZONE_CLEARANCE_SECONDS,
    same_lane_clearance: float = DEFAULT_SAME_LANE_CLEARANCE_SECONDS,
    current_time: float = 0.0,
) -> FCFSReservationSchedule:
    """Reserve conflict-zone operations using FCFS as the priority rule.

    This is a simple list scheduler for the intersection JSSP formulation:
    vehicles are considered in FCFS order, but each operation only reserves the
    conflict zone it uses. Therefore vehicles with disjoint zone sequences can
    have overlapping operation times. Same-lane FIFO order is enforced by
    delaying a follower's first operation until the previous same-lane vehicle
    has entered and cleared its first conflict zone plus a headway.
    """

    from src.constraints.feasibility import FeasibilityChecker

    records = sorted(vehicle_records, key=fcfs_priority_key)
    vehicle_order = [record.vehicle_id for record in records]
    checker = FeasibilityChecker(
        records,
        conflict_zone_clearance=conflict_zone_clearance,
        same_lane_clearance=same_lane_clearance,
    )
    state = checker.initial_state()
    feasibility_report: List[Dict[str, Any]] = []

    for record in records:
        mask = checker.action_mask(state, current_time, allow_future_start=False)
        feasibility_report.append(
            {
                "stage": "action_mask",
                "vehicle_id": record.vehicle_id,
                "current_time": current_time,
                "feasible_operations": [
                    result.node_id for result in mask if result.feasible
                ],
                "infeasible_operations": [
                    {
                        "node_id": result.node_id,
                        "reasons": list(result.reasons),
                        "earliest_start": result.earliest_start,
                    }
                    for result in mask
                    if not result.feasible
                ],
            }
        )
        route_reservations, route_results = checker.plan_vehicle_route(
            record.vehicle_id,
            state,
            current_time=current_time,
        )
        feasibility_report.append(
            {
                "stage": "route_plan",
                "vehicle_id": record.vehicle_id,
                "results": [
                    {
                        "node_id": result.node_id,
                        "feasible": result.feasible,
                        "reasons": list(result.reasons),
                        "earliest_start": result.earliest_start,
                        "earliest_exit": result.earliest_exit,
                        "details": result.details,
                    }
                    for result in route_results
                ],
            }
        )
        if not route_reservations:
            feasibility_report.append(
                {
                    "stage": "route_rejected",
                    "vehicle_id": record.vehicle_id,
                    "reason": "no_feasible_route_reservation",
                }
            )
            continue
        checker.commit_vehicle_route(record.vehicle_id, route_reservations, state)

    all_reservations: List[OperationReservation] = []
    by_vehicle: Dict[str, Tuple[OperationReservation, ...]] = {}
    for vehicle_id in vehicle_order:
        reservations = tuple(state.reservations_by_vehicle.get(vehicle_id, []))
        if not reservations:
            continue
        by_vehicle[vehicle_id] = reservations
        all_reservations.extend(reservations)

    return FCFSReservationSchedule(
        vehicle_order=vehicle_order,
        operation_reservations=all_reservations,
        reservations_by_vehicle=by_vehicle,
        vehicle_reservations=state.vehicle_reservations,
        conflict_zone_clearance=conflict_zone_clearance,
        same_lane_clearance=same_lane_clearance,
        feasibility_report=feasibility_report,
    )


def schedule_vehicle_ids(schedule: FCFSReservationSchedule | Iterable[str]) -> List[str]:
    if isinstance(schedule, FCFSReservationSchedule):
        return list(schedule.vehicle_order)
    return list(schedule)


def reservations_due_for_release(
    schedule: FCFSReservationSchedule,
    sim_time: float,
    release_lookahead: float,
    cleared_vehicles: Set[str],
) -> List[str]:
    """Return vehicles whose first reserved operation is due soon."""

    release_ids: List[str] = []
    release_threshold = sim_time + release_lookahead
    for vehicle_id in schedule.vehicle_order:
        if vehicle_id in cleared_vehicles:
            continue
        reservation = schedule.vehicle_reservations.get(vehicle_id)
        if reservation is None:
            continue
        if reservation.enter_time <= release_threshold:
            release_ids.append(vehicle_id)
    return release_ids


def vehicle_zone_set(schedule: FCFSReservationSchedule, vehicle_id: str) -> Set[str]:
    return {
        reservation.conflict_zone
        for reservation in schedule.reservations_by_vehicle.get(vehicle_id, ())
    }


def has_pending_same_lane_leader(
    vehicle_id: str,
    schedule: FCFSReservationSchedule,
    cleared_vehicles: Set[str],
    active_vehicle_source_lanes: Dict[str, str],
    selected_vehicle_ids: Set[str],
) -> bool:
    reservation = schedule.vehicle_reservations.get(vehicle_id)
    if reservation is None:
        return False

    for leader_id in schedule.vehicle_order:
        if leader_id == vehicle_id:
            return False

        leader_reservation = schedule.vehicle_reservations.get(leader_id)
        if leader_reservation is None:
            continue
        if leader_reservation.source_lane != reservation.source_lane:
            continue
        if leader_id in cleared_vehicles:
            continue

        # A follower must not be released while any earlier same-lane vehicle
        # is still waiting, entering, or traversing the intersection.
        if leader_id in selected_vehicle_ids:
            return True
        if active_vehicle_source_lanes.get(leader_id) == reservation.source_lane:
            return True
        return True

    return False


def select_releasable_vehicle_ids(
    schedule: FCFSReservationSchedule,
    *,
    sim_time: float,
    release_lookahead: float,
    cleared_vehicles: Set[str],
    active_vehicle_ids: Set[str],
    active_vehicle_zones: Dict[str, Set[str]],
    active_vehicle_source_lanes: Dict[str, str],
) -> List[str]:
    """Pick vehicles that can be released without live zone/lane conflicts.

    Planned reservation times are necessary but not sufficient in SUMO: a
    vehicle may arrive late or still be physically inside J0 after its planned
    exit. This selector treats active vehicles as occupying their route zones
    until they clear the intersection, then greedily admits due vehicles whose
    route-zone sets are disjoint. This preserves concurrency for non-conflicting
    routes while preventing overlaps from stale timing assumptions.
    """

    due_vehicle_ids = reservations_due_for_release(
        schedule=schedule,
        sim_time=sim_time,
        release_lookahead=release_lookahead,
        cleared_vehicles=cleared_vehicles,
    )
    due_set = set(due_vehicle_ids)

    selected: List[str] = [
        vehicle_id
        for vehicle_id in schedule.vehicle_order
        if vehicle_id in active_vehicle_ids and vehicle_id in due_set
    ]
    selected_set = set(selected)
    occupied_zones: Set[str] = set()
    occupied_lanes: Set[str] = set()

    for vehicle_id in active_vehicle_ids:
        if vehicle_id in cleared_vehicles:
            continue
        occupied_zones.update(active_vehicle_zones.get(vehicle_id, set()))
        source_lane = active_vehicle_source_lanes.get(vehicle_id)
        if source_lane:
            occupied_lanes.add(source_lane)

    for vehicle_id in due_vehicle_ids:
        if vehicle_id in selected_set:
            continue

        reservation = schedule.vehicle_reservations.get(vehicle_id)
        if reservation is None:
            continue
        if reservation.source_lane in occupied_lanes:
            continue
        if has_pending_same_lane_leader(
            vehicle_id=vehicle_id,
            schedule=schedule,
            cleared_vehicles=cleared_vehicles,
            active_vehicle_source_lanes=active_vehicle_source_lanes,
            selected_vehicle_ids=selected_set,
        ):
            continue

        route_zones = vehicle_zone_set(schedule, vehicle_id)
        if route_zones & occupied_zones:
            continue

        selected.append(vehicle_id)
        selected_set.add(vehicle_id)
        occupied_zones.update(route_zones)
        occupied_lanes.add(reservation.source_lane)

    return selected


def annotate_graph_with_schedule(graph: Any, schedule: FCFSReservationSchedule) -> None:
    """Attach reservation timing to graph operation nodes for logging/GNN use."""

    for reservation in schedule.operation_reservations:
        if reservation.node_id not in graph:
            continue
        graph.nodes[reservation.node_id]["reservation_enter_time"] = reservation.enter_time
        graph.nodes[reservation.node_id]["reservation_exit_time"] = reservation.exit_time


def vehicle_has_cleared_intersection(vehicle_id: str) -> bool:
    if vehicle_id not in traci.vehicle.getIDList():
        return True

    lane_id = traci.vehicle.getLaneID(vehicle_id)
    if not lane_id:
        return False

    edge_id = edge_from_lane(lane_id) if "_" in lane_id else lane_id
    return lane_id not in INCOMING_LANES and not lane_id.startswith(":") and edge_id not in INCOMING_EDGES


def restore_default_vehicle_control(vehicle_id: str) -> None:
    """Return a released vehicle to normal SUMO speed behavior after J0."""

    if vehicle_id not in traci.vehicle.getIDList():
        return

    try:
        traci.vehicle.setSpeed(vehicle_id, -1.0)
        traci.vehicle.setSpeedMode(vehicle_id, SUMO_DEFAULT_SPEED_MODE)
    except Exception as exc:
        print(f"Warning: failed to restore default control for {vehicle_id}: {exc}")


def clear_scheduled_stop(vehicle_id: str) -> None:
    stops = traci.vehicle.getStops(vehicle_id)
    if not stops:
        return
    traci.vehicle.replaceStop(vehicle_id, 0, "", teleport=0)


def release_speed_for_vehicle(vehicle_id: str) -> float:
    """Choose a modest positive speed for the vehicle currently scheduled."""

    try:
        return max(1.0, min(traci.vehicle.getAllowedSpeed(vehicle_id), EDGE_RELEASE_SPEED_LIMIT_MPS))
    except Exception:
        return RELEASE_SPEED_FALLBACK_MPS


def nudge_stuck_vehicle_onto_internal_link(vehicle_id: str) -> bool:
    """Move a released vehicle onto its open internal link if SUMO clips it at J0.

    Holding queued vehicles very close to the intersection can leave the active
    vehicle exactly at the inbound lane end even after it has been released. The
    scheduler has already selected this vehicle, so this tiny move places it on
    the same internal connection SUMO reports as open instead of letting the
    demo deadlock at the stop line.
    """

    lane_id = traci.vehicle.getLaneID(vehicle_id)
    if lane_id not in INCOMING_LANES:
        return False

    lane_length = traci.lane.getLength(lane_id)
    lane_position = traci.vehicle.getLanePosition(vehicle_id)
    distance_to_intersection = lane_length - lane_position
    if distance_to_intersection > LANE_END_NUDGE_DISTANCE_M:
        return False
    if traci.vehicle.getSpeed(vehicle_id) > LANE_END_NUDGE_MAX_SPEED_MPS:
        return False

    for next_link in traci.vehicle.getNextLinks(vehicle_id):
        if len(next_link) < 5:
            continue

        next_lane_id = next_link[0]
        is_open = bool(next_link[2])
        internal_lane_id = next_link[4]
        if not is_open:
            continue
        if not internal_lane_id or not str(internal_lane_id).startswith(":"):
            continue

        try:
            if traci.lane.getLastStepVehicleIDs(internal_lane_id):
                return False
            internal_lane_length = traci.lane.getLength(internal_lane_id)
            nudge_position = min(LANE_END_NUDGE_POSITION_M, max(0.0, internal_lane_length - 0.1))
            traci.vehicle.moveTo(vehicle_id, internal_lane_id, nudge_position)
            traci.vehicle.setSpeed(vehicle_id, release_speed_for_vehicle(vehicle_id))
            print(
                f"Debug: nudged released vehicle {vehicle_id} "
                f"from {lane_id} to {internal_lane_id} toward {next_lane_id}."
            )
            return True
        except Exception as exc:
            print(f"Warning: failed to nudge {vehicle_id} onto {internal_lane_id}: {exc}")
            return False

    return False


def release_vehicle(vehicle_id: str, state: ControllerState) -> bool:
    """Release a vehicle if it was held or had a pending hold stop."""

    if vehicle_id not in traci.vehicle.getIDList():
        return False

    did_release = False
    try:
        # Once the scheduler chooses a vehicle, the external scheduler is
        # responsible for permitting entry. A positive speed command avoids a
        # released vehicle lingering at the inbound lane end when other vehicles
        # are being held close to the junction.
        traci.vehicle.setSpeedMode(vehicle_id, SCHEDULED_VEHICLE_SPEED_MODE)
        if traci.vehicle.isStopped(vehicle_id):
            traci.vehicle.resume(vehicle_id)
            did_release = True
        else:
            before = len(traci.vehicle.getStops(vehicle_id))
            clear_scheduled_stop(vehicle_id)
            did_release = before > 0
        traci.vehicle.setSpeed(vehicle_id, release_speed_for_vehicle(vehicle_id))
        did_release = nudge_stuck_vehicle_onto_internal_link(vehicle_id) or did_release
    except Exception as exc:
        print(f"Warning: failed to release {vehicle_id}: {exc}")
        return False

    state.stopped_vehicles.discard(vehicle_id)
    state.released_vehicles.add(vehicle_id)
    return did_release


def stop_vehicle_if_needed(
    record: LiveVehicleRecord,
    stop_distance: float,
    hold_distance: float,
    state: ControllerState,
) -> bool:
    """Set a SUMO stop before J0 for a vehicle that is not allowed to enter."""

    if record.vehicle_id not in traci.vehicle.getIDList():
        return False
    if record.distance_to_intersection > hold_distance:
        return False

    lane_length = traci.lane.getLength(record.lane_id)
    stop_pos = max(1.0, lane_length - stop_distance)
    if record.lane_position >= stop_pos:
        try:
            if record.vehicle_id not in state.stopped_vehicles:
                traci.vehicle.setSpeedMode(record.vehicle_id, SUMO_DEFAULT_SPEED_MODE)
                traci.vehicle.setSpeed(record.vehicle_id, 0.0)
                state.stopped_vehicles.add(record.vehicle_id)
                return True
        except Exception as exc:
            print(f"Warning: failed to slow late-held {record.vehicle_id}: {exc}")
        return False

    try:
        if not traci.vehicle.getStops(record.vehicle_id):
            traci.vehicle.setStop(
                record.vehicle_id,
                edge_from_lane(record.lane_id),
                pos=stop_pos,
                laneIndex=0,
                duration=STOP_DURATION_SECONDS,
            )
            state.stopped_vehicles.add(record.vehicle_id)
            return True
    except Exception as exc:
        print(f"Warning: failed to stop {record.vehicle_id}: {exc}")
    return False


def apply_sumo_control(
    vehicle_records: Iterable[LiveVehicleRecord],
    schedule: FCFSReservationSchedule,
    state: ControllerState,
    stop_distance: float,
    hold_distance: float,
    sim_time: float,
    release_lookahead: float,
) -> Dict[str, Any]:
    """Apply the reservation schedule to SUMO with stop/release commands.

    The exact scheduler works at conflict-zone granularity. The current SUMO
    actuator is intentionally lightweight: it releases every vehicle whose first
    reserved conflict-zone operation is due soon and holds the rest near the
    intersection. This supports concurrent non-conflicting vehicles while
    keeping the controller simple enough to replace with PPO-GNN actions later.
    """

    records_by_id = {record.vehicle_id: record for record in vehicle_records}
    stopped_now: List[str] = []
    released_now: List[str] = []

    for vehicle_id in list(state.active_vehicle_ids):
        if vehicle_has_cleared_intersection(vehicle_id):
            restore_default_vehicle_control(vehicle_id)
            state.cleared_vehicles.add(vehicle_id)
            state.active_vehicle_ids.discard(vehicle_id)
            state.active_vehicle_zones.pop(vehicle_id, None)
            state.active_vehicle_source_lanes.pop(vehicle_id, None)

    allowed_vehicle_id_list = select_releasable_vehicle_ids(
        schedule=schedule,
        sim_time=sim_time,
        release_lookahead=release_lookahead,
        cleared_vehicles=state.cleared_vehicles,
        active_vehicle_ids=state.active_vehicle_ids,
        active_vehicle_zones=state.active_vehicle_zones,
        active_vehicle_source_lanes=state.active_vehicle_source_lanes,
    )
    allowed_vehicle_ids = set(allowed_vehicle_id_list)

    for vehicle_id in allowed_vehicle_id_list:
        record = records_by_id.get(vehicle_id)
        if vehicle_id not in traci.vehicle.getIDList():
            continue
        if record is None:
            continue
        if record.distance_to_intersection > hold_distance and vehicle_id not in state.stopped_vehicles:
            continue

        released = release_vehicle(vehicle_id, state)
        state.active_vehicle_ids.add(vehicle_id)
        state.active_vehicle_zones[vehicle_id] = vehicle_zone_set(schedule, vehicle_id)
        state.active_vehicle_source_lanes[vehicle_id] = record.source_lane
        if released:
            released_now.append(vehicle_id)

    for record in records_by_id.values():
        if record.vehicle_id in allowed_vehicle_ids:
            continue
        stopped = stop_vehicle_if_needed(
            record=record,
            stop_distance=stop_distance,
            hold_distance=hold_distance,
            state=state,
        )
        if stopped:
            stopped_now.append(record.vehicle_id)

    return {
        "allowed_vehicle_id": allowed_vehicle_id_list[0] if allowed_vehicle_id_list else None,
        "allowed_vehicle_ids": allowed_vehicle_id_list,
        "active_vehicle_ids": sorted(state.active_vehicle_ids),
        "stopped_now": stopped_now,
        "released_now": released_now,
        "currently_stopped": sorted(state.stopped_vehicles),
        "cleared": sorted(state.cleared_vehicles),
    }


def edge_type_counts(graph: Any) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for _, _, attrs in graph.edges(data=True):
        edge_type = attrs["edge_type"]
        counts[edge_type] = counts.get(edge_type, 0) + 1
    return counts


def print_debug_step(
    sim_time: float,
    vehicle_records: List[LiveVehicleRecord],
    graph: Any,
    graph_records: Dict[str, Any],
    schedule: FCFSReservationSchedule,
    control_result: Dict[str, Any],
) -> None:
    print(
        f"\n[t={sim_time:.1f}] detected={len(vehicle_records)} "
        f"allowed={control_result.get('allowed_vehicle_ids', [])}"
    )

    if vehicle_records:
        print("Detected vehicles:")
        for record in vehicle_records:
            print(
                "  "
                f"{record.vehicle_id:<8} route={record.route_id:<6} "
                f"type={record.vehicle_type:<9} lane={record.lane_id:<7} "
                f"pos={record.lane_position:>6.1f} speed={record.speed:>4.1f} "
                f"dist={record.distance_to_intersection:>6.1f} eta={record.estimated_arrival_time:>6.1f}"
            )
    else:
        print("Detected vehicles: none")

    node_ids = [operation.node_id for operation in graph_records["operations"]]
    print(f"Graph nodes ({graph.number_of_nodes()}): {node_ids}")
    print(f"Graph edges by type: {edge_type_counts(graph)}")
    print(f"FCFS order: {schedule.vehicle_order}")
    feasibility_report = getattr(schedule, "feasibility_report", [])
    if feasibility_report:
        latest_mask = next(
            (
                entry
                for entry in reversed(feasibility_report)
                if entry.get("stage") == "action_mask"
            ),
            None,
        )
        if latest_mask is not None:
            print(
                "Feasible operations now: "
                f"{latest_mask.get('feasible_operations', []) or []}"
            )
            infeasible = latest_mask.get("infeasible_operations", [])[:8]
            if infeasible:
                print("Infeasible operations:")
                for entry in infeasible:
                    print(
                        "  "
                        f"{entry.get('node_id')} reasons={entry.get('reasons', [])}"
                    )
    print("Reservations:")
    for reservation in schedule.operation_reservations[:20]:
        print(
            "  "
            f"{reservation.vehicle_id:<8} {reservation.conflict_zone} "
            f"op={reservation.operation_index} "
            f"[{reservation.enter_time:.1f}, {reservation.exit_time:.1f}]"
        )
    if len(schedule.operation_reservations) > 20:
        print(f"  ... +{len(schedule.operation_reservations) - 20} more operations")
    print(f"Stopped now: {control_result['stopped_now']}")
    print(f"Released now: {control_result['released_now']}")
    print(f"Currently stopped: {control_result['currently_stopped']}")


def jsonable_graph_step(
    sim_time: float,
    vehicle_records: List[LiveVehicleRecord],
    graph: Any,
    graph_records: Dict[str, Any],
    schedule: FCFSReservationSchedule,
    control_result: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "time": sim_time,
        "vehicles": [asdict(record) for record in vehicle_records],
        "nodes": [
            {"node_id": node_id, **attrs}
            for node_id, attrs in graph.nodes(data=True)
        ],
        "edge_counts": edge_type_counts(graph),
        "type1_edges": [asdict(edge) for edge in graph_records["type1_edges"]],
        "type2_edges": [asdict(edge) for edge in graph_records["type2_edges"]],
        "type3_candidate_pairs": [asdict(edge) for edge in graph_records["type3_edges"]],
        "fcfs_order": schedule.vehicle_order,
        "feasibility_report": getattr(schedule, "feasibility_report", []),
        "reservations": [asdict(reservation) for reservation in schedule.operation_reservations],
        "control": control_result,
    }


def write_csv_summary(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "time",
                "detected_count",
                "node_count",
                "directed_edge_count",
                "type1_edges",
                "type2_edges",
                "type3_edges",
                "active_vehicle",
                "allowed_vehicles",
                "active_vehicles",
                "reservation_count",
                "fcfs_order",
                "stopped_now",
                "released_now",
                "currently_stopped",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def append_jsonl(path: Path, step_record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        json.dump(step_record, handle)
        handle.write("\n")


def free_flow_route_processing_time(route_id: str, vehicle_type: str) -> float:
    zones = ROUTE_TO_CONFLICT_ZONES.get(route_id, [])
    processing_time = PROCESSING_TIME_BY_TYPE.get(vehicle_type, 2.0)
    return len(zones) * processing_time


def metrics_class_for_vehicle_type(vehicle_type: str) -> str:
    if vehicle_type == "truck":
        return "truck"
    if vehicle_type == "bus":
        return "bus"
    return "car"


def resolve_sumo_binary(gui: bool, explicit_binary: Optional[str]) -> str:
    if explicit_binary:
        return explicit_binary
    binary_name = "sumo-gui" if gui else "sumo"
    binary = shutil.which(binary_name)
    if not binary:
        raise SystemExit(f"Could not find {binary_name!r} on PATH.")
    return binary


def close_traci_connection(process_wait_seconds: float = 3.0) -> None:
    """Close TraCI without leaving a stuck SUMO/SUMO-GUI process behind.

    On this Windows setup, `traci.close(wait=True)` can block indefinitely when
    closing sumo-gui after a short debug run. Closing the socket first and then
    waiting on the exact process TraCI launched avoids global process cleanup and
    keeps runtime output files from staying locked.
    """

    process = None
    try:
        connection = traci.getConnection()
        process = getattr(connection, "_process", None)
    except Exception:
        process = None

    try:
        traci.close(False)
    except Exception as exc:
        print(f"Warning: TraCI close failed: {exc}")

    if process is None or process.poll() is not None:
        return

    try:
        process.wait(timeout=process_wait_seconds)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()


def run_controller(args: argparse.Namespace) -> TraCIRunnerResult:
    require_dependencies()

    sumo_binary = resolve_sumo_binary(args.gui, args.sumo_binary)
    sumo_cmd = [
        sumo_binary,
        "-c",
        str(args.config),
        "--duration-log.disable",
        "true",
    ]
    if args.gui and not args.no_gui_start:
        sumo_cmd.append("--start")
    if args.gui and args.gui_delay >= 0:
        sumo_cmd.extend(["--delay", f"{args.gui_delay:.1f}"])
    if getattr(args, "seed", None) is not None:
        sumo_cmd.extend(["--seed", str(args.seed)])
    if args.no_warnings:
        sumo_cmd.extend(["--no-warnings", "true"])

    jsonl_path = args.output_dir / "fcfs_controller_steps.jsonl"
    csv_path = args.output_dir / "fcfs_controller_summary.csv"
    if not args.no_logs:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path.write_text("", encoding="utf-8")

    state = ControllerState(
        first_eta_by_vehicle={},
        first_detected_by_vehicle={},
        stopped_vehicles=set(),
        released_vehicles=set(),
        cleared_vehicles=set(),
        warned_unmapped_routes=set(),
    )
    csv_rows: List[Dict[str, Any]] = []
    scheduler_runtimes_ms: List[float] = []
    number_of_stops = 0
    sumo_collisions = 0
    step_count = 0
    sim_time = 0.0
    route_by_vehicle: Dict[str, str] = {}
    type_by_vehicle: Dict[str, str] = {}
    metric_completed_vehicles: Set[str] = set()
    total_delay = 0.0
    total_waiting_time = 0.0
    maximum_waiting_time = 0.0
    class_delay_totals: Dict[str, float] = {"car": 0.0, "truck": 0.0, "bus": 0.0}
    class_delay_counts: Dict[str, int] = {"car": 0, "truck": 0, "bus": 0}
    safety_validator = getattr(args, "safety_validator", None)
    safety_seen_vehicle_ids: Set[str] = set()
    junction_id = getattr(args, "junction_id", "J0")
    scheduler = scheduler_from_args(args)

    print(f"Starting TraCI controller with: {' '.join(sumo_cmd)}")
    if args.gui:
        print("sumo-gui is paced for responsiveness; use the visual debugger for interactive pause/single-step controls.")
    traci.start(sumo_cmd)
    try:
        next_decision_time = 0.0
        step_seconds = traci.simulation.getDeltaT()
        realtime_factor = args.realtime_factor
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            sim_time = traci.simulation.getTime()
            step_count += 1
            try:
                colliding_vehicle_ids = list(traci.simulation.getCollidingVehiclesIDList())
                sumo_collisions += len(colliding_vehicle_ids)
                if colliding_vehicle_ids and safety_validator is not None:
                    safety_validator.record_collision(colliding_vehicle_ids, sim_time)
            except Exception:
                pass
            if safety_validator is not None:
                try:
                    center = traci.junction.getPosition(junction_id)
                except Exception:
                    center = (0.0, 0.0)
                current_vehicle_ids = set(traci.vehicle.getIDList())
                for missing_vehicle_id in sorted(safety_seen_vehicle_ids - current_vehicle_ids):
                    safety_validator.close_vehicle(missing_vehicle_id, sim_time)
                for vehicle_id in sorted(current_vehicle_ids):
                    try:
                        safety_validator.update_vehicle_position(
                            vehicle_id=vehicle_id,
                            route_id=traci.vehicle.getRouteID(vehicle_id),
                            sim_time=sim_time,
                            position=traci.vehicle.getPosition(vehicle_id),
                            center=center,
                        )
                    except Exception:
                        continue
                safety_seen_vehicle_ids = current_vehicle_ids
            if args.until is not None and sim_time > args.until:
                break

            vehicle_records = get_approaching_vehicles(
                sim_time=sim_time,
                state=state,
                detection_distance=args.detection_distance,
                min_eta_speed=args.min_eta_speed,
            )
            for record in vehicle_records:
                route_by_vehicle[record.vehicle_id] = record.route_id
                type_by_vehicle[record.vehicle_id] = record.vehicle_type
            graph, graph_records = build_jssp_graph(vehicle_records)
            scheduler_start = time.perf_counter()
            schedule = scheduler.schedule(
                vehicle_records,
                graph,
                sim_time,
            )
            scheduler_runtimes_ms.append((time.perf_counter() - scheduler_start) * 1000.0)
            control_result = apply_sumo_control(
                vehicle_records=vehicle_records,
                schedule=schedule,
                state=state,
                stop_distance=args.stop_distance,
                hold_distance=args.hold_distance,
                sim_time=sim_time,
                release_lookahead=args.release_lookahead,
            )
            number_of_stops += len(control_result["stopped_now"])
            newly_cleared = set(control_result["cleared"]) - metric_completed_vehicles
            for vehicle_id in sorted(newly_cleared):
                metric_completed_vehicles.add(vehicle_id)
                route_id = route_by_vehicle.get(vehicle_id, "")
                vehicle_type = type_by_vehicle.get(vehicle_id, "passenger")
                eta = state.first_eta_by_vehicle.get(vehicle_id, sim_time)
                baseline = free_flow_route_processing_time(route_id, vehicle_type)
                delay = max(0.0, sim_time - eta - baseline)
                waiting_time = 0.0
                try:
                    if vehicle_id in traci.vehicle.getIDList():
                        waiting_time = max(0.0, float(traci.vehicle.getAccumulatedWaitingTime(vehicle_id)))
                except Exception:
                    waiting_time = 0.0
                total_delay += delay
                total_waiting_time += waiting_time
                maximum_waiting_time = max(maximum_waiting_time, waiting_time)
                vehicle_class = metrics_class_for_vehicle_type(vehicle_type)
                class_delay_totals[vehicle_class] = class_delay_totals.get(vehicle_class, 0.0) + delay
                class_delay_counts[vehicle_class] = class_delay_counts.get(vehicle_class, 0) + 1

            should_print = sim_time + 1e-9 >= next_decision_time
            if should_print:
                annotate_graph_with_schedule(graph, schedule)
                print_debug_step(
                    sim_time=sim_time,
                    vehicle_records=vehicle_records,
                    graph=graph,
                    graph_records=graph_records,
                    schedule=schedule,
                    control_result=control_result,
                )
                next_decision_time = sim_time + args.decision_period

            if not args.no_logs and should_print:
                step_record = jsonable_graph_step(
                    sim_time=sim_time,
                    vehicle_records=vehicle_records,
                    graph=graph,
                    graph_records=graph_records,
                    schedule=schedule,
                    control_result=control_result,
                )
                append_jsonl(jsonl_path, step_record)
                edge_counts = edge_type_counts(graph)
                csv_rows.append(
                    {
                        "time": f"{sim_time:.1f}",
                        "detected_count": len(vehicle_records),
                        "node_count": graph.number_of_nodes(),
                        "directed_edge_count": graph.number_of_edges(),
                        "type1_edges": edge_counts.get("type1_precedence", 0),
                        "type2_edges": edge_counts.get("type2_same_lane_order", 0),
                        "type3_edges": edge_counts.get("type3_conflict_candidate", 0),
                        "active_vehicle": control_result["allowed_vehicle_id"] or "",
                        "allowed_vehicles": " ".join(control_result["allowed_vehicle_ids"]),
                        "active_vehicles": " ".join(control_result["active_vehicle_ids"]),
                        "reservation_count": len(schedule.operation_reservations),
                        "fcfs_order": " ".join(schedule.vehicle_order),
                        "stopped_now": " ".join(control_result["stopped_now"]),
                        "released_now": " ".join(control_result["released_now"]),
                        "currently_stopped": " ".join(control_result["currently_stopped"]),
                    }
                )
            if realtime_factor > 0:
                time.sleep(step_seconds / realtime_factor)
    finally:
        if safety_validator is not None:
            for vehicle_id in sorted(safety_seen_vehicle_ids):
                safety_validator.close_vehicle(vehicle_id, sim_time)
            safety_validator.close_open_intervals(sim_time)
        close_traci_connection()

    if not args.no_logs:
        write_csv_summary(csv_path, csv_rows)
        print(f"Wrote JSONL step log: {jsonl_path}")
        print(f"Wrote CSV summary: {csv_path}")

    avg_runtime = (
        sum(scheduler_runtimes_ms) / len(scheduler_runtimes_ms)
        if scheduler_runtimes_ms
        else 0.0
    )
    max_runtime = max(scheduler_runtimes_ms) if scheduler_runtimes_ms else 0.0
    class_delay = {
        vehicle_class: (
            class_delay_totals.get(vehicle_class, 0.0)
            / class_delay_counts.get(vehicle_class, 1)
            if class_delay_counts.get(vehicle_class, 0)
            else 0.0
        )
        for vehicle_class in ("car", "truck", "bus")
    }
    return TraCIRunnerResult(
        sim_time=sim_time,
        vehicles_completed=len(state.cleared_vehicles),
        cleared_vehicles=tuple(sorted(state.cleared_vehicles)),
        number_of_stops=number_of_stops,
        total_delay=total_delay,
        total_waiting_time=total_waiting_time,
        maximum_waiting_time=maximum_waiting_time,
        class_delay=class_delay,
        scheduler_runtime_avg_ms=avg_runtime,
        scheduler_runtime_max_ms=max_runtime,
        sumo_collisions=sumo_collisions,
        step_count=step_count,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a TraCI FCFS controller for the single SUMO intersection."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_SUMO_CONFIG,
        help=f"SUMO config path (default: {DEFAULT_SUMO_CONFIG})",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Run with sumo-gui instead of headless sumo.",
    )
    parser.add_argument(
        "--no-gui-start",
        action="store_true",
        help="Do not pass --start to sumo-gui. Most TraCI GUI runs should keep the default.",
    )
    parser.add_argument(
        "--gui-delay",
        type=float,
        default=40.0,
        help="Delay in ms passed to sumo-gui with --delay when --gui is used. Use -1 to disable.",
    )
    parser.add_argument(
        "--sumo-binary",
        default=None,
        help="Explicit SUMO binary path. Overrides --gui binary selection.",
    )
    parser.add_argument(
        "--until",
        type=float,
        default=120.0,
        help="Stop the demo after this simulation time in seconds. Use -1 for full config duration.",
    )
    parser.add_argument(
        "--detection-distance",
        type=float,
        default=180.0,
        help="Detect inbound vehicles within this many meters of the intersection.",
    )
    parser.add_argument(
        "--hold-distance",
        type=float,
        default=90.0,
        help="Issue stop commands to non-selected vehicles within this distance.",
    )
    parser.add_argument(
        "--stop-distance",
        type=float,
        default=8.0,
        help="Place the hold stop this many meters before the intersection.",
    )
    parser.add_argument(
        "--min-eta-speed",
        type=float,
        default=0.1,
        help="Minimum speed used for first ETA estimates.",
    )
    parser.add_argument(
        "--decision-period",
        type=float,
        default=1.0,
        help="Seconds between verbose debug/log records.",
    )
    parser.add_argument(
        "--conflict-zone-clearance",
        type=float,
        default=DEFAULT_CONFLICT_ZONE_CLEARANCE_SECONDS,
        help="Safety clearance in seconds after a conflict-zone operation.",
    )
    parser.add_argument(
        "--same-lane-clearance",
        type=float,
        default=DEFAULT_SAME_LANE_CLEARANCE_SECONDS,
        help="FIFO headway in seconds between same-lane vehicle entries.",
    )
    parser.add_argument(
        "--release-lookahead",
        type=float,
        default=DEFAULT_RELEASE_LOOKAHEAD_SECONDS,
        help="Release vehicles this many seconds before their first reserved zone entry.",
    )
    parser.add_argument(
        "--realtime-factor",
        type=float,
        default=0.0,
        help=(
            "Wall-clock pacing factor. 1.0 means about real time. "
            "Default 0 disables pacing for headless batch runs."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for JSONL/CSV logs (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--no-logs",
        action="store_true",
        help="Disable JSONL and CSV log output.",
    )
    parser.add_argument(
        "--no-warnings",
        action="store_true",
        help="Suppress SUMO warning output.",
    )
    args = parser.parse_args()
    if args.until is not None and args.until < 0:
        args.until = None
    if args.decision_period <= 0:
        raise ValueError("--decision-period must be positive")
    if args.conflict_zone_clearance < 0:
        raise ValueError("--conflict-zone-clearance must be non-negative")
    if args.same_lane_clearance < 0:
        raise ValueError("--same-lane-clearance must be non-negative")
    if args.release_lookahead < 0:
        raise ValueError("--release-lookahead must be non-negative")
    if args.realtime_factor < 0:
        raise ValueError("--realtime-factor must be non-negative")
    if args.gui and args.realtime_factor == 0.0:
        args.realtime_factor = 1.0
    return args


if __name__ == "__main__":
    run_controller(parse_args())
