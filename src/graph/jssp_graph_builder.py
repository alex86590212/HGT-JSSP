"""NetworkX JSSP graph builder for live and deterministic vehicle records."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

from src.sumo_interface.traci_runner import (
    EdgeRecord,
    LiveVehicleRecord,
    OperationRecord,
    add_type1_edges,
    add_type2_edges,
    add_type3_edges,
    annotate_graph_with_schedule,
    build_jssp_graph,
    build_operations,
    edge_type_counts,
    operation_id,
)


VehicleRecord = LiveVehicleRecord


__all__ = [
    "Any",
    "Dict",
    "EdgeRecord",
    "Iterable",
    "List",
    "OperationRecord",
    "Tuple",
    "VehicleRecord",
    "add_type1_edges",
    "add_type2_edges",
    "add_type3_edges",
    "annotate_graph_with_schedule",
    "build_jssp_graph",
    "build_operations",
    "edge_type_counts",
    "operation_id",
]

