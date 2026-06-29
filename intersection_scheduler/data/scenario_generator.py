"""Procedural scenario generator for a 4-way intersection."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from intersection_scheduler.environment.intersection import Vehicle


# 3x3 grid — (col, row) positions, normalised to [0,2]
ZONE_POSITIONS: Dict[int, Tuple[float, float]] = {
    1: (0.0, 2.0), 2: (1.0, 2.0), 3: (2.0, 2.0),
    4: (0.0, 1.0), 5: (1.0, 1.0), 6: (2.0, 1.0),
    7: (0.0, 0.0), 8: (1.0, 0.0), 9: (2.0, 0.0),
}

ROUTES: Dict[str, List[int]] = {
    # Straight through
    "N_S": [2, 5, 8],  "S_N": [8, 5, 2],
    "E_W": [6, 5, 4],  "W_E": [4, 5, 6],
    # Right turns
    "N_W": [2, 1, 4],  "S_E": [8, 9, 6],
    "E_N": [6, 3, 2],  "W_S": [4, 7, 8],
    # Left turns
    "N_E": [2, 5, 6],  "S_W": [8, 5, 4],
    "E_S": [6, 5, 8],  "W_N": [4, 5, 2],
}

SAME_LANE_GROUPS: Dict[str, List[str]] = {
    "north": ["N_S", "N_W", "N_E"],
    "south": ["S_N", "S_E", "S_W"],
    "east":  ["E_W", "E_N", "E_S"],
    "west":  ["W_E", "W_S", "W_N"],
}

# Map manoeuvre -> entry direction
_MANOEUVRE_TO_LANE: Dict[str, str] = {
    m: lane
    for lane, manoeuvres in SAME_LANE_GROUPS.items()
    for m in manoeuvres
}

# Zone "size" for processing time: corner/side vs centre
_ZONE_SIZE: Dict[int, float] = {
    1: 4.0, 2: 5.0, 3: 4.0,
    4: 5.0, 5: 7.0, 6: 5.0,
    7: 4.0, 8: 5.0, 9: 4.0,
}


@dataclass
class Scenario:
    vehicles: List[Vehicle]
    manoeuvres: List[str]


class ScenarioGenerator:
    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)

    def generate(
        self,
        n_vehicles: int,
        manoeuvre_types: Optional[List[str]] = None,
        arrival_window: float = 8.0,
        velocity_range: Tuple[float, float] = (8.0, 14.0),
        min_headway: float = 1.5,
    ) -> Scenario:
        if manoeuvre_types is None:
            manoeuvre_types = list(ROUTES.keys())

        max_attempts = 200
        last_scenario: Optional[Scenario] = None

        for _ in range(max_attempts):
            manoeuvres = [
                manoeuvre_types[i]
                for i in self.rng.integers(0, len(manoeuvre_types), size=n_vehicles)
            ]

            # Sample arrival times
            arrivals = sorted(self.rng.uniform(0.0, arrival_window, size=n_vehicles).tolist())

            # Enforce same-lane headway
            by_lane: Dict[str, List[int]] = {}
            for vi, m in enumerate(manoeuvres):
                lane = _MANOEUVRE_TO_LANE[m]
                by_lane.setdefault(lane, []).append(vi)

            valid = True
            for lane_vehicles in by_lane.values():
                sorted_by_arrival = sorted(lane_vehicles, key=lambda i: arrivals[i])
                for k in range(1, len(sorted_by_arrival)):
                    prev_i = sorted_by_arrival[k - 1]
                    curr_i = sorted_by_arrival[k]
                    if arrivals[curr_i] - arrivals[prev_i] < min_headway - 1e-9:
                        valid = False
                        break
                if not valid:
                    break

            if not valid:
                continue

            # Enforce no two vehicles from the same lane with identical arrival times
            seen: Dict[Tuple[str, float], bool] = {}
            dup = False
            for vi, m in enumerate(manoeuvres):
                lane = _MANOEUVRE_TO_LANE[m]
                key = (lane, round(arrivals[vi], 6))
                if key in seen:
                    dup = True
                    break
                seen[key] = True
            if dup:
                continue

            # Build vehicles
            vehicles = []
            for vid, (m, arr) in enumerate(zip(manoeuvres, arrivals)):
                route = ROUTES[m]
                vel = float(self.rng.uniform(*velocity_range))
                processing_times = [
                    _ZONE_SIZE[z] / vel + float(self.rng.uniform(0.0, 0.3))
                    for z in route
                ]
                vehicles.append(Vehicle(
                    id=vid,
                    arrival_time=arr,
                    route=route,
                    processing_times=processing_times,
                    velocity=vel,
                ))

            scenario = Scenario(vehicles=vehicles, manoeuvres=manoeuvres)
            last_scenario = scenario
            if validate_scenario(scenario):
                return scenario

        # Fallback: relax constraints and build a guaranteed valid scenario
        if last_scenario is not None:
            return last_scenario

        # Last resort: 2 vehicles straight through, guaranteed conflict
        manoeuvres = ["N_S", "E_W"]
        arrivals = [0.0, 0.1]
        vehicles = []
        for vid, (m, arr) in enumerate(zip(manoeuvres, arrivals)):
            route = ROUTES[m]
            vel = float(self.rng.uniform(*velocity_range))
            processing_times = [_ZONE_SIZE[z] / vel for z in route]
            vehicles.append(Vehicle(id=vid, arrival_time=arr, route=route,
                                    processing_times=processing_times, velocity=vel))
        return Scenario(vehicles=vehicles, manoeuvres=manoeuvres)

    def easy(self, n_vehicles: int = 2) -> Scenario:
        return self.generate(
            n_vehicles=n_vehicles,
            manoeuvre_types=["N_S", "S_N", "E_W", "W_E"],
            arrival_window=10.0,
            min_headway=2.0,
        )

    def medium(self, n_vehicles: int = 4) -> Scenario:
        return self.generate(
            n_vehicles=n_vehicles,
            manoeuvre_types=["N_S", "S_N", "E_W", "W_E", "N_W", "S_E", "E_N", "W_S"],
            arrival_window=6.0,
            min_headway=1.5,
        )

    def hard(self, n_vehicles: int = 6) -> Scenario:
        return self.generate(
            n_vehicles=n_vehicles,
            manoeuvre_types=list(ROUTES.keys()),
            arrival_window=4.0,
            min_headway=1.0,
        )


def validate_scenario(scenario: Scenario) -> bool:
    """Return True if scenario is non-trivial and well-formed."""
    vehicles = scenario.vehicles

    # All processing times > 0
    for v in vehicles:
        if any(pt <= 0 for pt in v.processing_times):
            return False

    # No two vehicles from same lane with identical arrival times
    seen: Dict[Tuple[str, float], bool] = {}
    for v, m in zip(vehicles, scenario.manoeuvres):
        lane = _MANOEUVRE_TO_LANE.get(m, "")
        key = (lane, round(v.arrival_time, 6))
        if key in seen:
            return False
        seen[key] = True

    # At least one Type-3 conflict must exist
    zone_to_vehicles: Dict[int, List[int]] = {}
    for v in vehicles:
        for z in v.route:
            zone_to_vehicles.setdefault(z, []).append(v.id)
    has_conflict = any(len(vs) > 1 for vs in zone_to_vehicles.values())
    return has_conflict


def get_curriculum_scenario(episode: int, gen: ScenarioGenerator) -> Scenario:
    if episode < 5_000:
        return gen.easy(n_vehicles=2)
    elif episode < 15_000:
        n = int(gen.rng.integers(3, 5))
        return gen.medium(n_vehicles=n)
    elif episode < 30_000:
        n = int(gen.rng.integers(4, 7))
        return gen.hard(n_vehicles=n)
    else:
        n = int(gen.rng.integers(3, 9))
        return gen.generate(
            n_vehicles=n,
            arrival_window=float(gen.rng.uniform(3.0, 10.0)),
        )
