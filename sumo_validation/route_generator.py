from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List

from sumo_validation.network.zone_mapping import zone_incoming_edge

VEHICLE_LENGTH_M = 5.0
MIN_GAP_M = 2.5


def build_route_file(arrivals, out_path: Path) -> None:
    root = ET.Element("routes")
    ET.SubElement(
        root, "vType",
        id="car", length=str(VEHICLE_LENGTH_M), minGap=str(MIN_GAP_M),
        maxSpeed="15", accel="2.6", decel="4.5",
    )

    for v in sorted(arrivals, key=lambda v: v.arrival_time):
        edges = _route_edges(v.route)
        veh = ET.SubElement(
            root, "vehicle",
            id=str(v.id), type="car", depart=f"{v.arrival_time:.2f}",
            departSpeed=f"{v.velocity:.2f}",
        )
        ET.SubElement(veh, "route", edges=" ".join(edges))

    ET.ElementTree(root).write(out_path)


def _route_edges(zones: List[int]) -> List[str]:
    # Every route in dynamic_scheduler.data.traffic_generator.ROUTES has >=2
    # zones, so this always yields >=1 edge (confirmed against ROUTES directly).
    edges = []
    for i in range(1, len(zones)):
        edges.append(zone_incoming_edge(zones[i], zones[i - 1]))
    return edges
