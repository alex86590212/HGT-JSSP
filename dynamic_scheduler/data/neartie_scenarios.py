from __future__ import annotations

import random
from typing import List

from dynamic_scheduler.data.traffic_generator import ROUTES, _ZONE_SIZE
from dynamic_scheduler.environment.dynamic_intersection import DynamicVehicle

DEFAULT_VELOCITY = 10.0
CONGESTED_ROUTE_KEY = "N_S_L"   # [2, 6, 10, 14]
CLEAR_ROUTE_KEY = "N_W"          # [1, 5] -- disjoint from CONGESTED_ROUTE_KEY


def _proc_times(route: List[int], velocity: float) -> List[float]:
    return [_ZONE_SIZE[z] / velocity for z in route]


def neartie_scenario(
    seed: int = 0,
    n_queue: int = 2,
    queue_gap: float = 0.15,
    deadline_tie_margin: float = 0.02,
) -> List[DynamicVehicle]:
    """A vehicle (the 'queue-front' candidate) that's about to be marginally
    preferred by EDF over a clear-path vehicle -- but the queue-front
    candidate's zone already has n_queue OTHER vehicles ahead of it in the
    SAME zone's queue (arrived slightly earlier, same route), so serving it
    first doesn't reduce ITS OWN wait at all (it's still stuck behind its
    own queue). The clear-path vehicle's zone has zero contention -- serving
    it is strictly free. EDF's deadline tie-break has no notion of "does
    serving this candidate now actually help" -- it only compares deadlines.
    """
    rng = random.Random(seed)
    congested_route = ROUTES[CONGESTED_ROUTE_KEY]
    congested_proc = _proc_times(congested_route, DEFAULT_VELOCITY)
    clear_route = ROUTES[CLEAR_ROUTE_KEY]
    clear_proc = _proc_times(clear_route, DEFAULT_VELOCITY)

    vehicles: List[DynamicVehicle] = []
    vid = 0

    # n_queue vehicles queued ahead, all on the congested route, arriving
    # first so they're detected/queued/tentative before the tie-break pair.
    queue_arrivals = [0.0 + i * queue_gap + rng.uniform(-0.01, 0.01) for i in range(n_queue)]
    for arrival in queue_arrivals:
        vehicles.append(DynamicVehicle(
            id=vid, arrival_time=arrival, route=list(congested_route),
            processing_times=list(congested_proc), velocity=DEFAULT_VELOCITY,
            manoeuvre=CONGESTED_ROUTE_KEY,
        ))
        vid += 1

    last_queue_arrival = queue_arrivals[-1]

    # Queue-front candidate: right behind the queue, same route -- its OWN
    # deadline is close to the clear vehicle's, but it's the (n_queue+1)-th
    # vehicle on a zone that already has n_queue ahead of it.
    congested_candidate_arrival = last_queue_arrival + queue_gap + rng.uniform(-0.01, 0.01)
    congested_id = vid
    vehicles.append(DynamicVehicle(
        id=vid, arrival_time=congested_candidate_arrival, route=list(congested_route),
        processing_times=list(congested_proc), velocity=DEFAULT_VELOCITY,
        manoeuvre=CONGESTED_ROUTE_KEY,
    ))
    vid += 1

    # Clear-path candidate: deadline marginally LATER than the congested
    # candidate's (so EDF's tie-break prefers the congested one first), but
    # its route shares no zone with the congested route -- zero contention.
    clear_arrival = congested_candidate_arrival + deadline_tie_margin
    clear_id = vid
    vehicles.append(DynamicVehicle(
        id=vid, arrival_time=clear_arrival, route=list(clear_route),
        processing_times=list(clear_proc), velocity=DEFAULT_VELOCITY,
        manoeuvre=CLEAR_ROUTE_KEY,
    ))
    vid += 1

    assert congested_id < clear_id, "tie-break relies on congested candidate having the lower vehicle_id"
    return vehicles


def neartie_batch(n_scenarios: int, **kwargs) -> List[List[DynamicVehicle]]:
    return [neartie_scenario(seed=s, **kwargs) for s in range(n_scenarios)]
