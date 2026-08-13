from __future__ import annotations

import random
from typing import List

from dynamic_scheduler.environment.dynamic_intersection import DynamicVehicle
from dynamic_scheduler.data.traffic_generator import ROUTES, _ZONE_SIZE

DEFAULT_VELOCITY = 10.0

# Four routes, each reaching SHARED_ZONE at a DIFFERENT route position, so
# each has a different "urgency profile" relative to raw arrival_time:
#   ZONE_ROUTE_KEYS[k] reaches SHARED_ZONE at route position k
# This gives iGreedy (soonest achievable start), Backpressure (zone
# backlog), and EDF (zone-local deadline = arrival + free-flow offset to
# that position) genuinely different bases to rank a THREE-OR-MORE-WAY
# contention on the same zone, instead of a single binary insert decision
# (which collapses to the same ranking under all three rules almost always).
ZONE_ROUTE_KEYS = ["W_E_L", "N_S_L", "S_W", "N_E"]
#   W_E_L: [9, 10, 11, 12]      -- zone 10 at position 1
#   N_S_L: [2, 6, 10, 14]       -- zone 10 at position 2
#   S_W:   [15, 11, 7, 6, 5]    -- (no zone 10; replaced below)
#   N_E:   [2, 6, 10, 11, 12]   -- zone 10 at position 2 (5-zone route, more
#                                  downstream congestion potential)
SHARED_ZONE = 10


def _proc_times(route: List[int], velocity: float) -> List[float]:
    return [_ZONE_SIZE[z] / velocity for z in route]


def pileup_scenario(
    seed: int = 0,
    n_per_group: int = 2,
    group_gap: float = 0.15,
) -> List[DynamicVehicle]:
    """Three-or-more-way contention at SHARED_ZONE: n_per_group vehicles on
    EACH of three routes that reach zone 10 at different route positions
    (W_E_L at position 1, N_S_L and N_E at position 2), all arriving in a
    tight, overlapping window so the zone-10 decision genuinely has 3+
    simultaneously-feasible candidates from DIFFERENT routes -- not just
    "insert one crosser into one queue" (a binary choice that collapses to
    the same ranking under iGreedy/Backpressure/EDF almost always).

    With 3+ simultaneous candidates from routes of different route-position
    (hence different zone-local deadlines) and different downstream route
    length (hence different backlog-accumulation potential), iGreedy
    (soonest achievable start), Backpressure (largest current zone
    backlog), and EDF (earliest zone-local deadline) can each produce a
    DIFFERENT total order at the zone-10 decision point.
    """
    rng = random.Random(seed)
    routes = {k: ROUTES[k] for k in ("W_E_L", "N_S_L", "N_E")}
    procs = {k: _proc_times(r, DEFAULT_VELOCITY) for k, r in routes.items()}

    vehicles: List[DynamicVehicle] = []
    vid = 0
    for group_idx, key in enumerate(routes):
        route = routes[key]
        proc = procs[key]
        for i in range(n_per_group):
            base = group_idx * group_gap * 0.5
            arrival = max(0.0, base + i * group_gap + rng.uniform(-0.03, 0.03))
            vehicles.append(DynamicVehicle(
                id=vid, arrival_time=arrival, route=list(route),
                processing_times=list(proc), velocity=DEFAULT_VELOCITY,
                manoeuvre=key,
            ))
            vid += 1

    return vehicles


def pileup_batch(n_scenarios: int, **kwargs) -> List[List[DynamicVehicle]]:
    return [pileup_scenario(seed=s, **kwargs) for s in range(n_scenarios)]
