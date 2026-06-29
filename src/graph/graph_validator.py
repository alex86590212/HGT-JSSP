"""Validation helpers for JSSP timing-conflict graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from src.graph.conflict_zones import CONFLICT_ZONES, ROUTE_TO_CONFLICT_ZONES


REQUIRED_NODE_FIELDS = {
    "vehicle_id",
    "route_id",
    "source_lane",
    "conflict_zone",
    "operation_index",
    "processing_time",
}


@dataclass(frozen=True)
class GraphValidationResult:
    valid: bool
    errors: List[str]


def validate_route_mapping(
    route_to_zones: Dict[str, List[str]] = ROUTE_TO_CONFLICT_ZONES,
    conflict_zones: Dict[str, str] = CONFLICT_ZONES,
) -> GraphValidationResult:
    errors: List[str] = []
    for route_id, zones in route_to_zones.items():
        if not zones:
            errors.append(f"Route {route_id} has no conflict-zone sequence.")
        for zone in zones:
            if zone not in conflict_zones:
                errors.append(f"Route {route_id} uses unknown zone {zone}.")
    return GraphValidationResult(valid=not errors, errors=errors)


def validate_jssp_graph(graph: Any) -> GraphValidationResult:
    errors: List[str] = []
    seen_nodes = set()

    for node_id, attrs in graph.nodes(data=True):
        if node_id in seen_nodes:
            errors.append(f"Duplicate node {node_id}.")
        seen_nodes.add(node_id)

        missing = sorted(REQUIRED_NODE_FIELDS - set(attrs))
        if missing:
            errors.append(f"Node {node_id} missing fields: {', '.join(missing)}.")
        zone = attrs.get("conflict_zone")
        if zone is not None and zone not in CONFLICT_ZONES:
            errors.append(f"Node {node_id} uses unknown zone {zone}.")

    for source, target, attrs in graph.edges(data=True):
        if source not in graph:
            errors.append(f"Edge source {source} does not exist.")
        if target not in graph:
            errors.append(f"Edge target {target} does not exist.")
        edge_type = attrs.get("edge_type")
        if not edge_type:
            errors.append(f"Edge {source}->{target} missing edge_type.")
        if edge_type == "type3_conflict_candidate":
            source_attrs = graph.nodes[source]
            target_attrs = graph.nodes[target]
            if source_attrs.get("source_lane") == target_attrs.get("source_lane"):
                errors.append(f"Invalid Type-3 same-lane conflict {source}->{target}.")
            if source_attrs.get("conflict_zone") != target_attrs.get("conflict_zone"):
                errors.append(f"Invalid Type-3 different-zone conflict {source}->{target}.")

    return GraphValidationResult(valid=not errors, errors=errors)

