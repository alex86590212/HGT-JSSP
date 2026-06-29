"""TraCI vehicle observation helpers."""

from __future__ import annotations

from src.sumo_interface.traci_runner import (
    ControllerState,
    LiveVehicleRecord,
    edge_from_lane,
    estimate_arrival_time,
    get_approaching_vehicles,
    source_direction_from_route,
    source_lane_from_route,
)


VehicleRecord = LiveVehicleRecord


class VehicleObserver:
    """Read approaching vehicles from TraCI.

    This class is intentionally thin around the current helper function. It
    gives the controller a stable dependency that can later be replaced by a
    mock observer in tests or by richer TraCI state extraction for PPO-GNN.
    """

    def __init__(self, detection_distance: float, min_eta_speed: float) -> None:
        self.detection_distance = detection_distance
        self.min_eta_speed = min_eta_speed

    def get_approaching_vehicles(self, sim_time: float, state: ControllerState):
        return get_approaching_vehicles(
            sim_time=sim_time,
            state=state,
            detection_distance=self.detection_distance,
            min_eta_speed=self.min_eta_speed,
        )


__all__ = [
    "ControllerState",
    "LiveVehicleRecord",
    "VehicleObserver",
    "VehicleRecord",
    "edge_from_lane",
    "estimate_arrival_time",
    "get_approaching_vehicles",
    "source_direction_from_route",
    "source_lane_from_route",
]
