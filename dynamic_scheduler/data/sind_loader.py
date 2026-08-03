"""Load SinD departure-demand scenarios as DynamicVehicle arrival streams.

SinD (see dynamic_scheduler/data/sind_dataset.md) supplies real-world
departure demand at signalized intersections — WHO enters, WHEN, and toward
WHICH exit approach. It does NOT supply a zone-level path through our 4x4
grid, per-zone processing times, or velocity: those are stripped by design
(the source dataset's raw trajectories/timings are explicitly excluded from
the prepared format). So this loader only replaces the ARRIVAL PROCESS
(what TrafficGenerator.generate_episode_arrivals produces from a Poisson
sample) with real timings and real route choices; velocity and processing
times are still synthesized exactly as TrafficGenerator does it — nothing
about the environment, graph, or policy changes.

Route mapping: SinD's route_id is a cardinal pair, e.g. "E_to_W" — no
lane-level left/right distinction (SinD has no notion of our zone grid).
Our ROUTES table splits each straight-through cardinal pair into two zone
paths (e.g. E_W_L / E_W_R, opposite entry lanes of the same approach). Since
SinD doesn't tell us which lane, each such departure is randomly assigned
one of the two variants (uniform, per departure) rather than picking one
arbitrarily and silently biasing which zones ever see SinD traffic.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dynamic_scheduler.environment.dynamic_intersection import DynamicVehicle
from intersection_scheduler.data.scenario_generator_4x4 import ROUTES, _ZONE_SIZE

_VELOCITY_RANGE: Tuple[float, float] = (8.0, 14.0)

# SinD cardinal pair -> our named route(s). Straight-through pairs have two
# zone-path variants (opposite entry lanes); turns have exactly one.
_ROUTE_ID_MAP: Dict[str, Tuple[str, ...]] = {
    "N_to_S": ("N_S_R", "N_S_L"),
    "S_to_N": ("S_N_R", "S_N_L"),
    "E_to_W": ("E_W_L", "E_W_R"),
    "W_to_E": ("W_E_L", "W_E_R"),
    "N_to_W": ("N_W",),
    "S_to_E": ("S_E",),
    "E_to_N": ("E_N",),
    "W_to_S": ("W_S",),
    "N_to_E": ("N_E",),
    "S_to_W": ("S_W",),
    "E_to_S": ("E_S",),
    "W_to_N": ("W_N",),
}


class SinDCatalog:
    """Index over the prepared SinD catalog: lists scenario files, loads them
    into DynamicVehicle arrival streams on demand."""

    def __init__(self, catalog_path: str = "dynamic_scheduler/data/processed/sind/catalog.json") -> None:
        self.catalog_path = Path(catalog_path)
        with open(self.catalog_path) as f:
            self.catalog = json.load(f)
        self.root = self.catalog_path.parent

        # Recording directory names on disk are a normalized form of the
        # catalog's recording_id (e.g. "Chongqing/6_22_NR_1" ->
        # "chongqing_6_22_nr_1") with no documented exact rule, so discover
        # converted recordings directly from the filesystem under each
        # cataloged intersection rather than reconstruct the path.
        self.scenario_paths: List[Path] = []
        for inter in self.catalog.get("intersections", []):
            if inter.get("scenario_count", 0) == 0:
                continue
            intersection_dir = self.root / "intersections" / inter["intersection_id"]
            recordings_dir = intersection_dir / "recordings"
            if not recordings_dir.is_dir():
                continue
            for recording_dir in sorted(recordings_dir.iterdir()):
                if recording_dir.is_dir():
                    self.scenario_paths.extend(sorted(recording_dir.glob("scenario_*.json")))

    def __len__(self) -> int:
        return len(self.scenario_paths)

    def load_scenario(
        self, path: Path, rng: Optional[random.Random] = None,
    ) -> Tuple[List[DynamicVehicle], float, dict]:
        """Load one scenario file into a (arrivals, episode_duration, meta) tuple.

        arrivals: DynamicVehicle list, sorted by arrival_time, ready for
        DynamicIntersectionEnv.reset — same shape TrafficGenerator produces.
        meta: the scenario's own metadata (intersection_id, recording_id,
        congestion label, ...) for logging/eval breakdowns.
        """
        rng = rng or random
        with open(path) as f:
            scenario = json.load(f)

        arrivals: List[DynamicVehicle] = []
        for i, dep in enumerate(scenario["departures"]):
            variants = _ROUTE_ID_MAP.get(dep["route_id"])
            if variants is None:
                continue  # unmapped cardinal pair (shouldn't occur; skip defensively)
            route_name = variants[0] if len(variants) == 1 else rng.choice(variants)
            route = ROUTES[route_name]

            vel = float(rng.uniform(*_VELOCITY_RANGE))
            processing_times = [
                _ZONE_SIZE[z] / vel + float(rng.uniform(0.0, 0.3))
                for z in route
            ]
            arrivals.append(DynamicVehicle(
                id=i,
                arrival_time=float(dep["departure_time_s"]),
                route=route,
                processing_times=processing_times,
                velocity=vel,
                manoeuvre=route_name,
            ))

        arrivals.sort(key=lambda v: v.arrival_time)
        for new_id, v in enumerate(arrivals):
            v.id = new_id

        meta = {
            "scenario_id": scenario.get("scenario_id"),
            "intersection_id": scenario.get("intersection_id"),
            "recording_id": scenario.get("recording_id"),
            "congestion": scenario.get("congestion"),
        }
        episode_duration = float(scenario.get("departure_horizon_s", 30.0))
        return arrivals, episode_duration, meta

    def sample(self, rng: Optional[random.Random] = None) -> Tuple[List[DynamicVehicle], float, dict]:
        """Pick a random scenario and load it — for mixing into a curriculum."""
        rng = rng or random
        path = rng.choice(self.scenario_paths)
        return self.load_scenario(path, rng=rng)

    def all_scenarios(
        self, rng: Optional[random.Random] = None,
    ) -> List[Tuple[List[DynamicVehicle], float, dict]]:
        """Load every scenario — for a fixed held-out eval tier."""
        rng = rng or random
        return [self.load_scenario(p, rng=rng) for p in self.scenario_paths]
