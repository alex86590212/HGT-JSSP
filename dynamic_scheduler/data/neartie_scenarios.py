from __future__ import annotations

import random
from typing import List

from dynamic_scheduler.data.traffic_generator import ROUTES, _ZONE_SIZE
from dynamic_scheduler.environment.dynamic_intersection import DynamicVehicle

DEFAULT_VELOCITY = 10.0

# Same construction as the redesigned pile-up scenario: 3 routes reaching a
# SHARED_ZONE at different route positions (hence different zone-local
# deadlines and different downstream backlog potential), arriving in a
# tight overlapping window so 3+ candidates genuinely contend at once.
# The original single-route-vs-disjoint-route near-tie design always tied
# iGreedy/Backpressure/EDF (confirmed empirically -- a binary "insert here
# or not" decision collapses to the same ranking under all three rules
# whenever no real backlog differential exists yet), so it could not
# separate EDF from the other baselines and has been replaced.
ZONE_ROUTE_KEYS = ("W_E_L", "N_S_L", "N_E")
SHARED_ZONE = 10


def _proc_times(route: List[int], velocity: float) -> List[float]:
    return [_ZONE_SIZE[z] / velocity for z in route]


def neartie_scenario(
    seed: int = 0,
    n_queue: int = 2,
    queue_gap: float = 0.05,
) -> List[DynamicVehicle]:
    """3-way contention at SHARED_ZONE with a near-tie margin: n_queue
    vehicles on each of 3 routes (W_E_L, N_S_L, N_E) arrive in a tight,
    overlapping window (queue_gap between successive arrivals within and
    across groups), so zone-local deadlines land close together across
    routes while each route's differing position/length gives iGreedy,
    Backpressure, and EDF different bases to break the near-tie.
    """
    rng = random.Random(seed)
    routes = {k: ROUTES[k] for k in ZONE_ROUTE_KEYS}
    procs = {k: _proc_times(r, DEFAULT_VELOCITY) for k, r in routes.items()}

    vehicles: List[DynamicVehicle] = []
    vid = 0
    for group_idx, key in enumerate(routes):
        route = routes[key]
        proc = procs[key]
        for i in range(n_queue):
            base = group_idx * queue_gap * 0.5
            arrival = max(0.0, base + i * queue_gap + rng.uniform(-0.01, 0.01))
            vehicles.append(DynamicVehicle(
                id=vid, arrival_time=arrival, route=list(route),
                processing_times=list(proc), velocity=DEFAULT_VELOCITY,
                manoeuvre=key,
            ))
            vid += 1

    return vehicles


def neartie_batch(n_scenarios: int, **kwargs) -> List[List[DynamicVehicle]]:
    return [neartie_scenario(seed=s, **kwargs) for s in range(n_scenarios)]
