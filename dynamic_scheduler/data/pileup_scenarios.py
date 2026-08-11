from __future__ import annotations

import random
from typing import List

from dynamic_scheduler.environment.dynamic_intersection import DynamicVehicle
from dynamic_scheduler.data.traffic_generator import ROUTES, _ZONE_SIZE

DEFAULT_VELOCITY = 10.0


def _proc_times(route: List[int], velocity: float) -> List[float]:
    return [_ZONE_SIZE[z] / velocity for z in route]


def pileup_scenario(
    seed: int = 0,
    n_trailing: int = 3,
    # Must be well under a zone's processing_time (~0.5-0.7s here at
    # DEFAULT_VELOCITY) or trailing vehicles clear zone 9 before the next
    # arrives and no real contention ever forms -- confirmed empirically:
    # trailing_gap=1.0 produced ~zero waiting time for every vehicle.
    trailing_gap: float = 0.15,
    contested_route_key: str = "W_E_L",
    clear_route_key: str = "N_W",
    urgent_lead: float = 0.5,
) -> List[DynamicVehicle]:
    """One EDF-urgent vehicle on a route with heavy downstream contention,
    vs. one slightly-less-urgent vehicle on a route with none.

    contested_route_key's route must pass through a zone that n_trailing
    other vehicles on the SAME route will also need shortly after -- the
    pile-up EDF can't see coming. clear_route_key's route shares no zone
    with the contested route, so it never competes for anything.
    """
    rng = random.Random(seed)
    contested_route = ROUTES[contested_route_key]
    clear_route = ROUTES[clear_route_key]
    # Small per-scenario jitter so a multi-scenario batch isn't n identical
    # copies of the same instance -- keeps the qualitative pattern (one
    # urgent + contended convoy vs. one clear-path vehicle) while varying
    # exact timing enough to average over.
    jitter = lambda scale: rng.uniform(-scale, scale)  # noqa: E731

    vehicles: List[DynamicVehicle] = []
    vid = 0

    # The urgent vehicle: earliest EDF deadline (arrival_time=0), first in
    # line on the contested route.
    vehicles.append(DynamicVehicle(
        id=vid, arrival_time=0.0, route=list(contested_route),
        processing_times=_proc_times(contested_route, DEFAULT_VELOCITY),
        velocity=DEFAULT_VELOCITY, manoeuvre=contested_route_key,
    ))
    vid += 1

    # The trailing vehicles that will pile up behind the urgent one on the
    # SAME route -- arriving shortly after, each further back.
    for i in range(n_trailing):
        gap = max(0.02, trailing_gap + jitter(trailing_gap * 0.2))
        arrival = (i + 1) * gap
        vehicles.append(DynamicVehicle(
            id=vid, arrival_time=arrival, route=list(contested_route),
            processing_times=_proc_times(contested_route, DEFAULT_VELOCITY),
            velocity=DEFAULT_VELOCITY, manoeuvre=contested_route_key,
        ))
        vid += 1

    # The clear-path vehicle: deadline slightly LATER than the urgent
    # vehicle's (so EDF always picks the urgent one first), but its route
    # shares no zone with the contested route -- no downstream pile-up ever
    # forms behind it regardless of when it's served.
    vehicles.append(DynamicVehicle(
        id=vid, arrival_time=max(0.01, urgent_lead + jitter(urgent_lead * 0.2)),
        route=list(clear_route),
        processing_times=_proc_times(clear_route, DEFAULT_VELOCITY),
        velocity=DEFAULT_VELOCITY, manoeuvre=clear_route_key,
    ))
    vid += 1

    return vehicles


def pileup_batch(n_scenarios: int, **kwargs) -> List[List[DynamicVehicle]]:
    return [pileup_scenario(seed=s, **kwargs) for s in range(n_scenarios)]
