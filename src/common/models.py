"""Shared model aliases for the current FCFS controller implementation.

The canonical tie-break rule is:
estimated_arrival_time, first_detected_time, source_lane, vehicle_id.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.sumo_interface.traci_runner import (
    ControllerState,
    EdgeRecord,
    FCFSReservationSchedule,
    LiveVehicleRecord,
    OperationRecord,
    OperationReservation,
    VehicleReservation,
)


VehicleRecord = LiveVehicleRecord
ScheduleResult = FCFSReservationSchedule


@dataclass(frozen=True)
class ControlResult:
    allowed_vehicle_id: Optional[str]
    allowed_vehicle_ids: List[str]
    active_vehicle_ids: List[str]
    stopped_now: List[str]
    released_now: List[str]
    currently_stopped: List[str]
    cleared: List[str]


@dataclass(frozen=True)
class SafetyViolation:
    violation_type: str
    conflict_zone: str
    vehicle_ids: Tuple[str, ...]
    start_time: float
    end_time: float
    details: str


@dataclass(frozen=True)
class EpisodeMetrics:
    episode_id: str
    total_vehicles_completed: int
    throughput: float
    average_delay: float
    total_delay: float
    average_waiting_time: float
    maximum_waiting_time: float
    number_of_stops: int
    scheduler_runtime_avg_ms: float
    safety_violations: int
    sumo_collisions: int
    deadlock_or_timeout_events: int
    class_delay: Dict[str, float]


__all__ = [
    "ControllerState",
    "ControlResult",
    "EdgeRecord",
    "EpisodeMetrics",
    "OperationRecord",
    "OperationReservation",
    "SafetyViolation",
    "ScheduleResult",
    "VehicleRecord",
    "VehicleReservation",
]
