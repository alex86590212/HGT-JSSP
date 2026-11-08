from __future__ import annotations

import random
from typing import List

from dynamic_scheduler.data.traffic_generator import ROUTES, _ZONE_SIZE
from dynamic_scheduler.environment.dynamic_intersection import DynamicVehicle

DEFAULT_VELOCITY = 10.0

# Both routes pass through zone 10, at DIFFERENT route positions:
#   BASE_ROUTE_KEY zone 10 is at position 2 (deadline = arrival + 2 zones'
#     free-flow offset -- "less urgent" relative to raw arrival_time)
#   DISRUPTOR_ROUTE_KEY zone 10 is at position 1 (deadline = arrival + 1
#     zone's offset -- "more urgent" relative to raw arrival_time)
# This asymmetry is the mechanism: it lets the disruptor be DETECTED later
# in absolute terms (arrival_time - detection_window is later) while still
# having a LOWER zone-10 deadline than the base vehicles once detected --
# something impossible if both vehicles hit zone 10 at the same route
# position, since deadline and detection time are then both monotonic in
# arrival_time in the same direction.
BASE_ROUTE_KEY = "N_S_L"       # [2, 6, 10, 14] -- zone 10 at position 2
DISRUPTOR_ROUTE_KEY = "W_E_L"  # [9, 10, 11, 12] -- zone 10 at position 1
CONTESTED_ZONE = 10


def _proc_times(route: List[int], velocity: float) -> List[float]:
    return [_ZONE_SIZE[z] / velocity for z in route]


def _deadline_for_zone(route: List[int], proc: List[float], zone: int, arrival: float) -> float:
    pos = route.index(zone)
    return arrival + sum(proc[:pos])


def churn_scenario(
    seed: int = 0,
    n_base: int = 3,
    base_gap: float = 0.6,
    detection_window: float = 10.0,
    # Available slack is (base_offset - disruptor_offset) = 1.2 - 0.5 = 0.7s
    # (see module docstring), split between how much the disruptor's
    # deadline undercuts the last base vehicle's (deadline_margin) and how
    # long after the base queue settles it must still be detected
    # (detect_margin) -- deadline_margin + detect_margin must stay under 0.7
    # or the two constraints become infeasible (see the asserts below).
    disruptor_deadline_margin: float = 0.2,
    disruptor_detect_margin: float = 0.3,
) -> List[DynamicVehicle]:
    """n_base vehicles on BASE_ROUTE_KEY queue for the shared zone
    (CONTESTED_ZONE), detected and settled into TENTATIVE plans first. A
    disruptor on DISRUPTOR_ROUTE_KEY is then detected -- strictly after the
    base queue has settled -- with a zone-10 deadline that undercuts the
    last base vehicle's, forcing a real choice: bump the disruptor ahead
    (paying requeue churn on the base vehicles it displaces) or leave the
    settled order alone and accept the disruptor's own delay. EDF only
    compares deadlines and has no notion of the churn cost either way.
    """
    rng = random.Random(seed)
    base_route = ROUTES[BASE_ROUTE_KEY]
    base_proc = _proc_times(base_route, DEFAULT_VELOCITY)
    disruptor_route = ROUTES[DISRUPTOR_ROUTE_KEY]
    disruptor_proc = _proc_times(disruptor_route, DEFAULT_VELOCITY)

    vehicles: List[DynamicVehicle] = []
    vid = 0

    first_base_arrival = detection_window + 3.0
    base_arrivals = [first_base_arrival + i * base_gap + rng.uniform(-0.03, 0.03)
                      for i in range(n_base)]
    for arrival in base_arrivals:
        vehicles.append(DynamicVehicle(
            id=vid, arrival_time=arrival, route=list(base_route),
            processing_times=list(base_proc), velocity=DEFAULT_VELOCITY,
            manoeuvre=BASE_ROUTE_KEY,
        ))
        vid += 1

    last_base_arrival = base_arrivals[-1]
    last_base_deadline = _deadline_for_zone(base_route, base_proc, CONTESTED_ZONE, last_base_arrival)
    latest_base_detect = last_base_arrival - detection_window

    # Solve for disruptor_arrival directly: its zone-10 deadline is
    # (arrival + disruptor_offset), want that == last_base_deadline - margin.
    disruptor_pos = disruptor_route.index(CONTESTED_ZONE)
    disruptor_offset = sum(disruptor_proc[:disruptor_pos])
    disruptor_arrival = (last_base_deadline - disruptor_deadline_margin) - disruptor_offset
    disruptor_detect = disruptor_arrival - detection_window

    assert disruptor_detect > latest_base_detect + disruptor_detect_margin, (
        f"disruptor detected at {disruptor_detect:.2f} must be after the last "
        f"base vehicle's detection ({latest_base_detect:.2f}) plus margin -- "
        f"reduce disruptor_deadline_margin or n_base*base_gap"
    )
    disruptor_deadline = disruptor_arrival + disruptor_offset
    assert disruptor_deadline < last_base_deadline, "disruptor must undercut the last base vehicle's deadline"

    vehicles.append(DynamicVehicle(
        id=vid, arrival_time=disruptor_arrival, route=list(disruptor_route),
        processing_times=list(disruptor_proc), velocity=DEFAULT_VELOCITY,
        manoeuvre=DISRUPTOR_ROUTE_KEY,
    ))
    vid += 1

    return vehicles


def churn_batch(n_scenarios: int, **kwargs) -> List[List[DynamicVehicle]]:
    return [churn_scenario(seed=s, **kwargs) for s in range(n_scenarios)]
