"""Conflict-zone definitions for the single-intersection network."""

from __future__ import annotations

from typing import Dict, List

from src.config.load_config import load_vehicle_types


CONFLICT_ZONES: Dict[str, str] = {
    "z1": "northeast conflict zone",
    "z2": "northwest conflict zone",
    "z3": "southwest conflict zone",
    "z4": "southeast conflict zone",
}


ROUTE_TO_CONFLICT_ZONES: Dict[str, List[str]] = {
    "E_to_W": ["z1", "z2"],
    "W_to_E": ["z3", "z4"],
    "N_to_S": ["z2", "z3"],
    "S_to_N": ["z4", "z1"],
    "E_to_N": ["z1", "z4"],
    "N_to_W": ["z2", "z1"],
    "W_to_S": ["z3", "z2"],
    "S_to_E": ["z4", "z3"],
    "E_to_S": ["z1", "z2", "z3"],
    "N_to_E": ["z2", "z3", "z4"],
    "W_to_N": ["z3", "z4", "z1"],
    "S_to_W": ["z4", "z1", "z2"],
}


DEFAULT_PROCESSING_TIME_BY_TYPE: Dict[str, float] = {
    "passenger": 1.8,
    "delivery": 2.2,
    "truck": 2.8,
    "bus": 2.8,
}


def load_processing_time_by_type() -> Dict[str, float]:
    """Load nominal per-zone processing times from vehicle_types.json.

    These times are scheduler/JSSP model parameters, not values computed by
    SUMO. Keeping the JSON config authoritative prevents experiments from
    silently using a different set of constants than the documented config.
    """

    vehicle_types = load_vehicle_types()
    raw_times = vehicle_types.get("processing_time_by_type")
    if not isinstance(raw_times, dict):
        return dict(DEFAULT_PROCESSING_TIME_BY_TYPE)

    processing_times: Dict[str, float] = {}
    for vehicle_type, value in raw_times.items():
        try:
            processing_time = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid processing time for vehicle type {vehicle_type!r}: {value!r}"
            ) from exc
        if processing_time <= 0:
            raise ValueError(
                f"Processing time for vehicle type {vehicle_type!r} must be positive."
            )
        processing_times[str(vehicle_type)] = processing_time

    return processing_times or dict(DEFAULT_PROCESSING_TIME_BY_TYPE)


PROCESSING_TIME_BY_TYPE: Dict[str, float] = load_processing_time_by_type()
