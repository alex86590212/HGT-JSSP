from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from sumo_validation.network.zone_mapping import zone_incoming_edge


@dataclass(frozen=True)
class ZoneReservation:
    zone_id: int
    edge_id: str
    enter_time: float
    exit_time: float


@dataclass(frozen=True)
class VehicleSchedule:
    vehicle_id: int
    route_zones: Tuple[int, ...]
    reservations: Tuple[ZoneReservation, ...]


def translate(
    op_timeline: Dict[Tuple[int, int], dict],
    arrivals,
) -> Dict[int, VehicleSchedule]:
    routes_by_vid = {v.id: v.route for v in arrivals}
    entries_by_vid: Dict[int, List[Tuple[int, dict]]] = {}
    for (vid, route_pos), entry in op_timeline.items():
        entries_by_vid.setdefault(vid, []).append((route_pos, entry))

    result: Dict[int, VehicleSchedule] = {}
    for vid, entries in entries_by_vid.items():
        entries.sort(key=lambda t: t[0])
        route = routes_by_vid[vid]
        reservations = []
        for route_pos, entry in entries:
            if route_pos == 0:
                # route[0] has no SUMO edge of its own -- route_generator.py
                # builds one edge per (route[i-1], route[i]) pair, so the
                # vehicle's first SUMO edge corresponds to route_pos == 1.
                # A vehicle occupying only route[0] (never reaching route[1])
                # has no SUMO-side reservation to enforce.
                continue
            zid = entry["zone_id"]
            prev_zid = route[route_pos - 1]
            edge_id = zone_incoming_edge(zid, prev_zid)
            reservations.append(
                ZoneReservation(
                    zone_id=zid,
                    edge_id=edge_id,
                    enter_time=entry["start"],
                    exit_time=entry["finish"],
                )
            )
        result[vid] = VehicleSchedule(
            vehicle_id=vid,
            route_zones=tuple(route),
            reservations=tuple(reservations),
        )
    return result
