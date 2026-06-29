"""SUMO vehicle control helpers."""

from __future__ import annotations

from src.sumo_interface.traci_runner import (
    ControllerState,
    apply_sumo_control,
    clear_scheduled_stop,
    release_vehicle,
    restore_default_vehicle_control,
    stop_vehicle_if_needed,
    vehicle_has_cleared_intersection,
)


class VehicleController:
    """Apply scheduler output to SUMO stop/release commands."""

    def __init__(
        self,
        stop_distance: float,
        hold_distance: float,
        release_lookahead: float,
    ) -> None:
        self.stop_distance = stop_distance
        self.hold_distance = hold_distance
        self.release_lookahead = release_lookahead

    def apply(self, vehicle_records, schedule, state: ControllerState, sim_time: float):
        return apply_sumo_control(
            vehicle_records=vehicle_records,
            schedule=schedule,
            state=state,
            stop_distance=self.stop_distance,
            hold_distance=self.hold_distance,
            sim_time=sim_time,
            release_lookahead=self.release_lookahead,
        )


__all__ = [
    "ControllerState",
    "VehicleController",
    "apply_sumo_control",
    "clear_scheduled_stop",
    "release_vehicle",
    "restore_default_vehicle_control",
    "stop_vehicle_if_needed",
    "vehicle_has_cleared_intersection",
]
