"""Poisson-arrival traffic generator for the online 4x4 dynamic scheduler.

Unlike intersection_scheduler.data.scenario_generator_4x4.ScenarioGenerator
(samples a fixed N-vehicle batch), this generates a continuous arrival stream
for a fixed wall-clock episode duration: vehicles arrive via a Poisson
process with rate `arrival_rate` (vehicles/sec), each assigned a random
manoeuvre/velocity exactly as the offline 4x4 generator does.

Curriculum difficulty is controlled by arrival_rate (vehicles/sec) and the
manoeuvre pool, mirroring the offline model's easy/medium/hard tiers but as
a rate axis instead of a fixed-count axis (see design discussion — this
replaces "N vehicles in a scenario" with "vehicles/sec of continuous
traffic").
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from dynamic_scheduler.environment.dynamic_intersection import DynamicVehicle
from intersection_scheduler.data.scenario_generator_4x4 import (
    ROUTES,
    _ZONE_SIZE,
)

_VELOCITY_RANGE: Tuple[float, float] = (8.0, 14.0)
_MIN_HEADWAY = 1.0  # minimum gap (s) between arrivals in the same lane group


class TrafficGenerator:
    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)
        self._next_id = 0

    def generate_episode_arrivals(
        self,
        duration: float,
        arrival_rate: float,
        manoeuvre_types: Optional[List[str]] = None,
    ) -> List[DynamicVehicle]:
        """Sample a Poisson arrival stream over [0, duration].

        arrival_rate: mean vehicles/sec (constant within this episode).
        Returns vehicles sorted by arrival_time, each with a fresh unique id.
        """
        if manoeuvre_types is None:
            manoeuvre_types = list(ROUTES.keys())

        arrivals: List[DynamicVehicle] = []
        t = 0.0
        while True:
            # Exponential inter-arrival time for a Poisson process.
            gap = self.rng.exponential(1.0 / arrival_rate) if arrival_rate > 0 else float("inf")
            t += gap
            if t >= duration:
                break

            manoeuvre = manoeuvre_types[self.rng.integers(0, len(manoeuvre_types))]
            route = ROUTES[manoeuvre]
            vel = float(self.rng.uniform(*_VELOCITY_RANGE))
            processing_times = [
                _ZONE_SIZE[z] / vel + float(self.rng.uniform(0.0, 0.3))
                for z in route
            ]
            arrivals.append(DynamicVehicle(
                id=self._next_id,
                arrival_time=t,
                route=route,
                processing_times=processing_times,
                velocity=vel,
                manoeuvre=manoeuvre,
            ))
            self._next_id += 1

        return arrivals

    def easy(self, duration: float) -> List[DynamicVehicle]:
        # All manoeuvre types (including left turns) at every tier — only
        # arrival rate varies across the curriculum, since this experiment
        # validates online scheduling mechanics under full realistic traffic
        # rather than re-teaching manoeuvre-type difficulty from scratch.
        #
        # Rates chosen from an iGreedy saturation sweep on the 4x4 intersection
        # (16 zones, high parallel capacity): rates below ~1/s produce almost
        # no contention (iGreedy already ~0s waiting, nothing to learn), so the
        # tiers span mild contention -> genuine saturation where scheduling
        # quality actually matters:
        #   easy  1.5/s (~90 veh/60s, iGreedy ~0.8s)
        #   medium 2.5/s (~150 veh, iGreedy ~1.1s)
        #   hard  4.0/s (~240 veh, iGreedy ~10.6s, ~69% completion — saturated)
        return self.generate_episode_arrivals(
            duration, arrival_rate=1.5, manoeuvre_types=list(ROUTES.keys()),
        )

    def medium(self, duration: float) -> List[DynamicVehicle]:
        return self.generate_episode_arrivals(
            duration, arrival_rate=2.5, manoeuvre_types=list(ROUTES.keys()),
        )

    def hard(self, duration: float) -> List[DynamicVehicle]:
        return self.generate_episode_arrivals(
            duration, arrival_rate=4.0, manoeuvre_types=list(ROUTES.keys()),
        )


def get_curriculum_arrivals(episode: int, gen: TrafficGenerator, duration: float) -> List[DynamicVehicle]:
    """Curriculum phases as arrival-rate tiers, mirroring
    scenario_generator_4x4.get_curriculum_scenario's episode boundaries."""
    if episode < 8_000:
        return gen.easy(duration)
    elif episode < 25_000:
        return gen.medium(duration)
    elif episode < 50_000:
        return gen.hard(duration)
    else:
        # Mixed regime: sample across the full meaningful contention range,
        # from mild (1.0/s) through saturation (5.0/s), so the policy
        # generalises across traffic densities rather than overfitting one.
        rate = float(gen.rng.uniform(1.0, 5.0))
        return gen.generate_episode_arrivals(duration, arrival_rate=rate)
