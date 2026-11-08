from __future__ import annotations

from intersection_scheduler.data.scenario_generator_4x4 import ZONE_POSITIONS

_ROWS = "ABCD"


def _node_name(x: float, y: float) -> str:
    col = int(round(x))
    row = int(round(y))
    return f"{_ROWS[3 - row]}{col}"


ZONE_TO_NODE = {zid: _node_name(x, y) for zid, (x, y) in ZONE_POSITIONS.items()}
NODE_TO_ZONE = {node: zid for zid, node in ZONE_TO_NODE.items()}


def zone_incoming_edge(zid: int, from_zid: int) -> str:
    return f"{ZONE_TO_NODE[from_zid]}{ZONE_TO_NODE[zid]}"


def zone_node(zid: int) -> str:
    return ZONE_TO_NODE[zid]
