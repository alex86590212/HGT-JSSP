"""Conflict-zone-aware FCFS scheduler."""

from __future__ import annotations

from typing import Any, Iterable

from src.schedulers.base import BaseScheduler
from src.sumo_interface.traci_runner import (
    DEFAULT_CONFLICT_ZONE_CLEARANCE_SECONDS,
    DEFAULT_SAME_LANE_CLEARANCE_SECONDS,
    fcfs_schedule,
    fcfs_vehicle_order,
    select_releasable_vehicle_ids,
)


class FCFSScheduler(BaseScheduler):
    name = "fcfs"

    def __init__(
        self,
        conflict_zone_clearance: float = DEFAULT_CONFLICT_ZONE_CLEARANCE_SECONDS,
        same_lane_clearance: float = DEFAULT_SAME_LANE_CLEARANCE_SECONDS,
    ) -> None:
        self.conflict_zone_clearance = conflict_zone_clearance
        self.same_lane_clearance = same_lane_clearance

    def schedule(self, vehicle_records: Iterable[Any], jssp_graph: Any, current_time: float):
        return fcfs_schedule(
            vehicle_records,
            conflict_zone_clearance=self.conflict_zone_clearance,
            same_lane_clearance=self.same_lane_clearance,
            current_time=current_time,
        )


__all__ = ["FCFSScheduler", "fcfs_schedule", "fcfs_vehicle_order", "select_releasable_vehicle_ids"]
