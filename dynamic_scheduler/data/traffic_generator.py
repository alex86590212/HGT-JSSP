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
        # Rates calibrated against REAL arrival rates measured directly from
        # the 78 SinD scenarios (dynamic_scheduler/data/processed/sind/),
        # total departures/intersection across all approaches:
        #   min=0.067  p25=0.100  median=0.133  mean=0.183  p75=0.200  max=0.833
        # An earlier 1.5/2.5/4.0 retune was calibrated against the FIFO
        # baseline's waiting time instead, which produced tiers 5-20x busier
        # than anything observed in real intersection data (our "easy" was
        # ~8x the real median) — good for finding a regime where FIFO pays a
        # price, useless for training a policy meant to run on real traffic.
        # Each tier takes its real-world reference rate with a ~15-20%
        # margin (not the raw historical value) so training sees moderately
        # busier traffic than what was observed, not just a replay of it:
        #   easy   0.20/s  (~p75 of real data,          ~12 veh/60s)
        #   medium 0.35/s  (~1.7x real mean,             ~21 veh/60s)
        #   hard   1.00/s  (~1.2x real max ever observed, ~60 veh/60s)
        return self.generate_episode_arrivals(
            duration, arrival_rate=0.20, manoeuvre_types=list(ROUTES.keys()),
        )

    def medium(self, duration: float) -> List[DynamicVehicle]:
        return self.generate_episode_arrivals(
            duration, arrival_rate=0.35, manoeuvre_types=list(ROUTES.keys()),
        )

    def hard(self, duration: float) -> List[DynamicVehicle]:
        return self.generate_episode_arrivals(
            duration, arrival_rate=1.00, manoeuvre_types=list(ROUTES.keys()),
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
        # Mixed regime: sample across the full realistic contention range,
        # from below-median real traffic (0.1/s) through past the busiest
        # observed real window (1.2/s), so the policy generalises across
        # traffic densities actually seen at real intersections rather than
        # overfitting one.
        rate = float(gen.rng.uniform(0.1, 1.2))
        return gen.generate_episode_arrivals(duration, arrival_rate=rate)
