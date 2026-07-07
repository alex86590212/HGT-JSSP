"""Procedural scenario generator for a 4-way, 2-lane-per-approach intersection.

Modelled as a 4x4 grid of 16 zones. Each approach (N/S/E/W) has a left lane
(straight or left-turn only) and a right lane (straight or right-turn only),
matching standard lane discipline. Straight-through manoeuvres travel a full
row/column; right turns are a short 2-zone hop into the adjacent core zone;
left turns travel straight through 2-3 core zones before sweeping perpendicular
to their far exit.

Zone grid:
     1   2   3   4
     5   6   7   8
     9  10  11  12
    13  14  15  16

Entry zones: 1,2 (N) / 15,16 (S) / 4,8 (E) / 9,13 (W)
Exit zones:  3,4 (N) / 13,14 (S) / 1,5 (E) / 12,16 (W)
Core zones (shared crossing points): 6, 7, 10, 11
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from intersection_scheduler.data.scenario_generator import Scenario
from intersection_scheduler.environment.intersection import Vehicle

# 4x4 grid — (col, row) positions, normalised to [0,3]
ZONE_POSITIONS: Dict[int, Tuple[float, float]] = {
    1: (0.0, 3.0), 2: (1.0, 3.0), 3: (2.0, 3.0), 4: (3.0, 3.0),
    5: (0.0, 2.0), 6: (1.0, 2.0), 7: (2.0, 2.0), 8: (3.0, 2.0),
    9: (0.0, 1.0), 10: (1.0, 1.0), 11: (2.0, 1.0), 12: (3.0, 1.0),
    13: (0.0, 0.0), 14: (1.0, 0.0), 15: (2.0, 0.0), 16: (3.0, 0.0),
}

ROUTES: Dict[str, List[int]] = {
    # Straight through — 4 zones, full row/column
    "N_S_R": [1, 5, 9, 13],
    "N_S_L": [2, 6, 10, 14],
    "S_N_R": [16, 12, 8, 4],
    "S_N_L": [15, 11, 7, 3],
    "E_W_L": [8, 7, 6, 5],
    "E_W_R": [4, 3, 2, 1],
    "W_E_L": [9, 10, 11, 12],
    "W_E_R": [13, 14, 15, 16],
    # Right turns — 2 zones, immediate peel into adjacent core zone
    "N_W": [1, 5],
    "S_E": [16, 12],
    "E_N": [4, 3],
    "W_S": [13, 14],
    # Left turns — 5 zones, straight through core then sweep to far exit
    "N_E": [2, 6, 10, 11, 12],
    "S_W": [15, 11, 7, 6, 5],
    "E_S": [8, 7, 6, 10, 14],
    "W_N": [9, 10, 11, 7, 3],
}

# Same-lane FIFO groups: one queue per direction per physical lane (8 total).
# Left-lane and right-lane vehicles from the same direction don't block each
# other — only vehicles sharing a physical lane queue in arrival order.
SAME_LANE_GROUPS: Dict[str, List[str]] = {
    "north_left":  ["N_S_L", "N_E"],
    "north_right": ["N_S_R", "N_W"],
    "south_left":  ["S_N_L", "S_W"],
    "south_right": ["S_N_R", "S_E"],
    "east_left":   ["E_W_L", "E_S"],
    "east_right":  ["E_W_R", "E_N"],
    "west_left":   ["W_E_L", "W_N"],
    "west_right":  ["W_E_R", "W_S"],
}

_MANOEUVRE_TO_LANE: Dict[str, str] = {
    m: lane
    for lane, manoeuvres in SAME_LANE_GROUPS.items()
    for m in manoeuvres
}

# Zone size for processing time: core (crossing) zones are larger than the
# 12 entry/exit zones, mirroring the 3x3 model's corner-vs-centre distinction.
_CORE_ZONES = {6, 7, 10, 11}
_ZONE_SIZE: Dict[int, float] = {
    zid: (7.0 if zid in _CORE_ZONES else 5.0)
    for zid in range(1, 17)
}

_STRAIGHT_MANOEUVRES = ["N_S_R", "N_S_L", "S_N_R", "S_N_L", "E_W_L", "E_W_R", "W_E_L", "W_E_R"]
_RIGHT_TURNS = ["N_W", "S_E", "E_N", "W_S"]
_LEFT_TURNS = ["N_E", "S_W", "E_S", "W_N"]


def validate_scenario(scenario: Scenario) -> bool:
    """Return True if scenario is non-trivial and well-formed."""
    vehicles = scenario.vehicles

    for v in vehicles:
        if any(pt <= 0 for pt in v.processing_times):
            return False

    seen: Dict[Tuple[str, float], bool] = {}
    for v, m in zip(vehicles, scenario.manoeuvres):
        lane = _MANOEUVRE_TO_LANE.get(m, "")
        key = (lane, round(v.arrival_time, 6))
        if key in seen:
            return False
        seen[key] = True

    zone_to_vehicles: Dict[int, List[int]] = {}
    for v in vehicles:
        for z in v.route:
            zone_to_vehicles.setdefault(z, []).append(v.id)
    return any(len(vs) > 1 for vs in zone_to_vehicles.values())


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

            arrivals = sorted(self.rng.uniform(0.0, arrival_window, size=n_vehicles).tolist())

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

        if last_scenario is not None:
            return last_scenario

        manoeuvres = ["N_S_L", "E_W_L"]
        arrivals = [0.0, 0.1]
        vehicles = []
        for vid, (m, arr) in enumerate(zip(manoeuvres, arrivals)):
            route = ROUTES[m]
            vel = float(self.rng.uniform(*velocity_range))
            processing_times = [_ZONE_SIZE[z] / vel for z in route]
            vehicles.append(Vehicle(id=vid, arrival_time=arr, route=route,
                                    processing_times=processing_times, velocity=vel))
        return Scenario(vehicles=vehicles, manoeuvres=manoeuvres)

    def easy(self, n_vehicles: int = 4) -> Scenario:
        return self.generate(
            n_vehicles=n_vehicles,
            manoeuvre_types=list(_STRAIGHT_MANOEUVRES),
            arrival_window=10.0,
            min_headway=2.0,
        )

    def medium(self, n_vehicles: int = 8) -> Scenario:
        return self.generate(
            n_vehicles=n_vehicles,
            manoeuvre_types=list(_STRAIGHT_MANOEUVRES) + list(_RIGHT_TURNS),
            arrival_window=6.0,
            min_headway=1.5,
        )

    def hard(self, n_vehicles: int = 12) -> Scenario:
        return self.generate(
            n_vehicles=n_vehicles,
            manoeuvre_types=list(ROUTES.keys()),
            arrival_window=4.0,
            min_headway=1.0,
        )


def get_curriculum_scenario(episode: int, gen: ScenarioGenerator) -> Scenario:
    """Curriculum phase boundaries, stretched relative to the 3x3 model.

    The 4x4 intersection has ~2-3x the combinatorial complexity (16 zones vs 9,
    16 manoeuvres vs 12, 8 same-lane queues vs 4, up to 5-zone left turns vs 3,
    2x the vehicle counts per tier). The 3x3 curriculum gave its hard phase
    20k episodes (ep 30k-50k of a 50k run); this stretches each phase
    proportionally for a ~100k-episode run.
    """
    if episode < 8_000:
        return gen.easy(n_vehicles=4)
    elif episode < 25_000:
        n = int(gen.rng.integers(5, 9))
        return gen.medium(n_vehicles=n)
    elif episode < 50_000:
        n = int(gen.rng.integers(8, 13))
        return gen.hard(n_vehicles=n)
    else:
        n = int(gen.rng.integers(4, 16))
        return gen.generate(
            n_vehicles=n,
            arrival_window=float(gen.rng.uniform(3.0, 10.0)),
        )
