"""Geometry and interval based safety validation for conflict zones."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from src.common.models import SafetyViolation


@dataclass(frozen=True)
class ZoneOccupancyInterval:
    vehicle_id: str
    route_id: str
    conflict_zone: str
    entry_time: float
    exit_time: float


class SafetyValidator:
    """Track actual conflict-zone occupancy and report overlaps.

    The scheduler reserves zones using predicted times. This validator is meant
    to be fed actual vehicle positions from SUMO and is therefore the source of
    truth for safety checks before a PPO-GNN scheduler is introduced.
    """

    def __init__(
        self,
        zone_half_size_m: float = 8.0,
        overlap_tolerance_s: float = 0.0,
    ) -> None:
        self.zone_half_size_m = zone_half_size_m
        self.overlap_tolerance_s = overlap_tolerance_s
        self.intervals: List[ZoneOccupancyInterval] = []
        self._open: Dict[str, Tuple[str, str, float]] = {}
        self._external_violations: List[SafetyViolation] = []

    def zone_for_position(
        self,
        x: float,
        y: float,
        center: Tuple[float, float] = (0.0, 0.0),
    ) -> Optional[str]:
        cx, cy = center
        half = self.zone_half_size_m
        if not (cx - half <= x <= cx + half and cy - half <= y <= cy + half):
            return None
        if x >= cx and y >= cy:
            return "z1"
        if x < cx and y >= cy:
            return "z2"
        if x < cx and y < cy:
            return "z3"
        return "z4"

    def update_vehicle_position(
        self,
        vehicle_id: str,
        route_id: str,
        sim_time: float,
        position: Tuple[float, float],
        center: Tuple[float, float] = (0.0, 0.0),
    ) -> None:
        zone = self.zone_for_position(position[0], position[1], center=center)
        previous = self._open.get(vehicle_id)
        previous_zone = previous[0] if previous else None

        if previous_zone == zone:
            return

        if previous is not None:
            old_zone, old_route, entry_time = previous
            self.record_interval(
                vehicle_id=vehicle_id,
                route_id=old_route,
                conflict_zone=old_zone,
                entry_time=entry_time,
                exit_time=sim_time,
            )
            self._open.pop(vehicle_id, None)

        if zone is not None:
            self._open[vehicle_id] = (zone, route_id, sim_time)

    def close_vehicle(self, vehicle_id: str, sim_time: float) -> None:
        previous = self._open.pop(vehicle_id, None)
        if previous is None:
            return
        zone, route_id, entry_time = previous
        self.record_interval(
            vehicle_id=vehicle_id,
            route_id=route_id,
            conflict_zone=zone,
            entry_time=entry_time,
            exit_time=sim_time,
        )

    def close_open_intervals(self, final_time: float) -> None:
        for vehicle_id in list(self._open):
            self.close_vehicle(vehicle_id, final_time)

    def record_interval(
        self,
        vehicle_id: str,
        route_id: str,
        conflict_zone: str,
        entry_time: float,
        exit_time: float,
    ) -> None:
        if exit_time < entry_time:
            raise ValueError("zone occupancy exit_time must be >= entry_time")
        self.intervals.append(
            ZoneOccupancyInterval(
                vehicle_id=vehicle_id,
                route_id=route_id,
                conflict_zone=conflict_zone,
                entry_time=entry_time,
                exit_time=exit_time,
            )
        )

    def record_collision(
        self,
        vehicle_ids: Sequence[str],
        sim_time: float,
        details: str = "SUMO collision event",
    ) -> None:
        self._external_violations.append(
            SafetyViolation(
                violation_type="sumo_collision",
                conflict_zone="SUMO",
                vehicle_ids=tuple(vehicle_ids),
                start_time=sim_time,
                end_time=sim_time,
                details=details,
            )
        )

    def validate_overlaps(self) -> List[SafetyViolation]:
        violations: List[SafetyViolation] = []
        by_zone: Dict[str, List[ZoneOccupancyInterval]] = {}
        for interval in self.intervals:
            by_zone.setdefault(interval.conflict_zone, []).append(interval)

        for zone, intervals in by_zone.items():
            ordered = sorted(intervals, key=lambda item: (item.entry_time, item.exit_time, item.vehicle_id))
            for index, first in enumerate(ordered):
                for second in ordered[index + 1 :]:
                    if second.entry_time >= first.exit_time:
                        break
                    if first.vehicle_id == second.vehicle_id:
                        continue
                    overlap_start = max(first.entry_time, second.entry_time)
                    overlap_end = min(first.exit_time, second.exit_time)
                    if overlap_end - overlap_start <= self.overlap_tolerance_s:
                        continue
                    violations.append(
                        SafetyViolation(
                            violation_type="conflict_zone_overlap",
                            conflict_zone=zone,
                            vehicle_ids=(first.vehicle_id, second.vehicle_id),
                            start_time=overlap_start,
                            end_time=overlap_end,
                            details=(
                                f"{first.vehicle_id} and {second.vehicle_id} overlap "
                                f"in {zone} for {overlap_end - overlap_start:.3f}s"
                            ),
                        )
                    )
        return violations

    def safety_violations(self) -> List[SafetyViolation]:
        return self.validate_overlaps() + list(self._external_violations)

    def write_zone_occupancy_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "vehicle_id",
                    "route_id",
                    "conflict_zone",
                    "entry_time",
                    "exit_time",
                ],
            )
            writer.writeheader()
            for interval in self.intervals:
                writer.writerow(asdict(interval))

    def write_violations_json(
        self,
        path: Path,
        violations: Optional[Iterable[SafetyViolation]] = None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            asdict(violation)
            for violation in (self.safety_violations() if violations is None else violations)
        ]
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


__all__ = ["SafetyValidator", "ZoneOccupancyInterval"]
