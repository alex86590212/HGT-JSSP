from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Set

from sumo_validation.translate import VehicleSchedule

STOP_DISTANCE_M = 5.0
SPEED_MODE_DEFAULT = 31


def _ensure_traci_on_path() -> None:
    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home:
        tools = os.path.join(sumo_home, "tools")
        if tools not in sys.path:
            sys.path.append(tools)


_ensure_traci_on_path()
import traci  # noqa: E402


@dataclass
class ControllerState:
    stopped_vehicles: Set[str] = field(default_factory=set)
    released_vehicles: Set[str] = field(default_factory=set)
    cleared_vehicles: Set[str] = field(default_factory=set)


def vehicle_has_cleared(vehicle_id: str) -> bool:
    if vehicle_id not in traci.vehicle.getIDList():
        return True
    current_edge = traci.vehicle.getRoadID(vehicle_id)
    remaining = traci.vehicle.getRoute(vehicle_id)[traci.vehicle.getRouteIndex(vehicle_id) + 1:]
    return not remaining and bool(current_edge) and not current_edge.startswith(":")


def stop_vehicle(vehicle_id: str, state: ControllerState) -> None:
    if vehicle_id not in traci.vehicle.getIDList():
        return
    if vehicle_id in state.stopped_vehicles:
        return
    traci.vehicle.setSpeedMode(vehicle_id, SPEED_MODE_DEFAULT)
    traci.vehicle.setSpeed(vehicle_id, 0.0)
    state.stopped_vehicles.add(vehicle_id)


def release_vehicle(vehicle_id: str, state: ControllerState) -> None:
    if vehicle_id not in traci.vehicle.getIDList():
        return
    traci.vehicle.setSpeedMode(vehicle_id, SPEED_MODE_DEFAULT)
    traci.vehicle.setSpeed(vehicle_id, -1.0)
    state.stopped_vehicles.discard(vehicle_id)
    state.released_vehicles.add(vehicle_id)


def _reservation_for_current_edge(sumo_id: str, schedule: VehicleSchedule):
    current_edge = traci.vehicle.getRoadID(sumo_id)
    for res in schedule.reservations:
        if res.edge_id == current_edge:
            return res
    return None


def _apply_reservation(sumo_id: str, reservation, sim_time: float, state: ControllerState) -> None:
    if sim_time >= reservation.enter_time:
        release_vehicle(sumo_id, state)
        return
    lane_id = traci.vehicle.getLaneID(sumo_id)
    if not lane_id:
        return
    dist_to_edge_end = traci.lane.getLength(lane_id) - traci.vehicle.getLanePosition(sumo_id)
    if dist_to_edge_end <= STOP_DISTANCE_M:
        stop_vehicle(sumo_id, state)


def step_control(
    schedules: Dict[int, VehicleSchedule],
    sim_time: float,
    state: ControllerState,
) -> None:
    for vid, schedule in schedules.items():
        sumo_id = str(vid)
        if sumo_id not in traci.vehicle.getIDList() or sumo_id in state.cleared_vehicles:
            continue
        if vehicle_has_cleared(sumo_id):
            state.cleared_vehicles.add(sumo_id)
            continue

        reservation = _reservation_for_current_edge(sumo_id, schedule)
        if reservation is not None:
            _apply_reservation(sumo_id, reservation, sim_time, state)


