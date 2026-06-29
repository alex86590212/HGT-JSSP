"""Build an example JSSP graph for the single-intersection SUMO setup.

This script does not talk to SUMO. It shows the data transformation that a
future TraCI bridge would perform:

* vehicles are jobs
* conflict zones are machines/resources
* one vehicle-zone occupation is one operation/node
* route sequences define within-job precedence edges
* same-lane arrivals and shared zones define scheduling constraints
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from src.graph.conflict_zones import (
    CONFLICT_ZONES,
    PROCESSING_TIME_BY_TYPE,
    ROUTE_TO_CONFLICT_ZONES,
)

try:
    import networkx as nx
except ModuleNotFoundError:
    nx = None

try:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
except ModuleNotFoundError:
    plt = None
    Line2D = None


DEFAULT_OUTPUT_DIR = Path("outputs") / "jssp_graph"


@dataclass(frozen=True)
class Vehicle:
    """Manual vehicle state that would later come from TraCI."""

    vehicle_id: str
    vehicle_type: str
    route_id: str
    arrival_time: float

    @property
    def source_lane(self) -> str:
        source_approach = self.route_id.split("_to_", maxsplit=1)[0]
        return f"{source_approach}_in"


@dataclass(frozen=True)
class Operation:
    """One vehicle occupying one conflict zone."""

    node_id: str
    vehicle_id: str
    vehicle_type: str
    route_id: str
    source_lane: str
    conflict_zone: str
    operation_index: int
    arrival_time: float
    processing_time: float


@dataclass(frozen=True)
class EdgeRecord:
    """Human-readable edge record for printing and JSON export."""

    edge_type: str
    source: str
    target: str
    description: str
    conflict_zone: str | None = None
    source_lane: str | None = None
    candidate_pair_id: str | None = None


def example_vehicles() -> List[Vehicle]:
    """Return a small deterministic traffic snapshot.

    There are two vehicles per source lane so Type-2 same-lane order edges are
    visible. The route mix also makes several Type-3 conflict candidates.
    """

    return [
        Vehicle("veh1", "passenger", "E_to_W", 3.0),
        Vehicle("veh2", "delivery", "E_to_N", 7.0),
        Vehicle("veh3", "passenger", "N_to_S", 4.0),
        Vehicle("veh4", "truck", "N_to_W", 8.5),
        Vehicle("veh5", "delivery", "W_to_E", 5.5),
        Vehicle("veh6", "passenger", "W_to_S", 9.0),
        Vehicle("veh7", "passenger", "S_to_N", 6.5),
        Vehicle("veh8", "truck", "S_to_E", 10.5),
    ]


def require_networkx() -> None:
    if nx is None:
        raise SystemExit(
            "Missing dependency: networkx. Install dependencies with "
            "`python -m pip install -r requirements.txt`."
        )


def operation_id(vehicle: Vehicle, conflict_zone: str, index: int) -> str:
    return f"{vehicle.vehicle_id}__{conflict_zone}__op{index}"


def build_operations(vehicles: Iterable[Vehicle]) -> Tuple[List[Operation], Dict[str, List[Operation]]]:
    operations: List[Operation] = []
    by_vehicle: Dict[str, List[Operation]] = {}

    for vehicle in vehicles:
        if vehicle.route_id not in ROUTE_TO_CONFLICT_ZONES:
            raise ValueError(f"No conflict-zone mapping defined for route {vehicle.route_id!r}")
        if vehicle.vehicle_type not in PROCESSING_TIME_BY_TYPE:
            raise ValueError(f"No processing time defined for vehicle type {vehicle.vehicle_type!r}")

        route_operations: List[Operation] = []
        processing_time = PROCESSING_TIME_BY_TYPE[vehicle.vehicle_type]
        for index, conflict_zone in enumerate(ROUTE_TO_CONFLICT_ZONES[vehicle.route_id]):
            operation = Operation(
                node_id=operation_id(vehicle, conflict_zone, index),
                vehicle_id=vehicle.vehicle_id,
                vehicle_type=vehicle.vehicle_type,
                route_id=vehicle.route_id,
                source_lane=vehicle.source_lane,
                conflict_zone=conflict_zone,
                operation_index=index,
                arrival_time=vehicle.arrival_time,
                processing_time=processing_time,
            )
            operations.append(operation)
            route_operations.append(operation)
        by_vehicle[vehicle.vehicle_id] = route_operations

    return operations, by_vehicle


def add_operation_nodes(graph: Any, operations: Iterable[Operation]) -> None:
    for operation in operations:
        graph.add_node(
            operation.node_id,
            vehicle_id=operation.vehicle_id,
            vehicle_type=operation.vehicle_type,
            route_id=operation.route_id,
            source_lane=operation.source_lane,
            conflict_zone=operation.conflict_zone,
            conflict_zone_name=CONFLICT_ZONES[operation.conflict_zone],
            operation_index=operation.operation_index,
            arrival_time=operation.arrival_time,
            processing_time=operation.processing_time,
        )


def add_type1_precedence_edges(
    graph: Any,
    operations_by_vehicle: Dict[str, List[Operation]],
) -> List[EdgeRecord]:
    """Type-1: route precedence for consecutive operations of the same job."""

    edges: List[EdgeRecord] = []
    for vehicle_id, route_operations in operations_by_vehicle.items():
        for previous, current in zip(route_operations, route_operations[1:]):
            graph.add_edge(
                previous.node_id,
                current.node_id,
                edge_type="type1_precedence",
                label="T1",
                vehicle_id=vehicle_id,
            )
            edges.append(
                EdgeRecord(
                    edge_type="type1_precedence",
                    source=previous.node_id,
                    target=current.node_id,
                    description=f"{vehicle_id} route order",
                )
            )
    return edges


def add_type2_same_lane_edges(
    graph: Any,
    operations_by_vehicle: Dict[str, List[Operation]],
) -> List[EdgeRecord]:
    """Type-2: preserve FIFO order for vehicles detected on the same source lane.

    The edge links the first operation of the earlier-arriving vehicle to the
    first operation of the later-arriving vehicle. A stricter model could also
    add lane-order constraints between later route operations if needed.
    """

    first_operations = [ops[0] for ops in operations_by_vehicle.values()]
    by_lane: Dict[str, List[Operation]] = {}
    for operation in first_operations:
        by_lane.setdefault(operation.source_lane, []).append(operation)

    edges: List[EdgeRecord] = []
    for source_lane, lane_operations in by_lane.items():
        lane_operations.sort(key=lambda op: (op.arrival_time, op.vehicle_id))
        for leader, follower in zip(lane_operations, lane_operations[1:]):
            graph.add_edge(
                leader.node_id,
                follower.node_id,
                edge_type="type2_same_lane_order",
                label="T2",
                source_lane=source_lane,
            )
            edges.append(
                EdgeRecord(
                    edge_type="type2_same_lane_order",
                    source=leader.node_id,
                    target=follower.node_id,
                    description=f"{source_lane} arrival order",
                    source_lane=source_lane,
                )
            )
    return edges


def add_type3_conflict_edges(graph: Any, operations: Iterable[Operation]) -> List[EdgeRecord]:
    """Type-3: paired candidate edges for operations sharing a conflict zone.

    These are disjunctive scheduling choices. The scheduler should choose one
    direction for each pair, e.g. op_a before op_b or op_b before op_a.
    """

    by_zone: Dict[str, List[Operation]] = {}
    for operation in operations:
        by_zone.setdefault(operation.conflict_zone, []).append(operation)

    edges: List[EdgeRecord] = []
    pair_index = 0
    for conflict_zone, zone_operations in by_zone.items():
        zone_operations.sort(key=lambda op: (op.arrival_time, op.vehicle_id))
        for first, second in combinations(zone_operations, 2):
            if first.vehicle_id == second.vehicle_id:
                continue
            if first.source_lane == second.source_lane:
                continue

            pair_id = f"conflict_pair_{pair_index:03d}"
            pair_index += 1

            for source, target in [(first, second), (second, first)]:
                graph.add_edge(
                    source.node_id,
                    target.node_id,
                    edge_type="type3_conflict_candidate",
                    label="T3",
                    conflict_zone=conflict_zone,
                    candidate_pair_id=pair_id,
                )

            edges.append(
                EdgeRecord(
                    edge_type="type3_conflict_candidate_pair",
                    source=first.node_id,
                    target=second.node_id,
                    description=f"shared {conflict_zone}; choose one ordering",
                    conflict_zone=conflict_zone,
                    candidate_pair_id=pair_id,
                )
            )
    return edges


def build_jssp_graph(vehicles: Iterable[Vehicle] | None = None) -> Tuple[Any, Dict[str, Any]]:
    """Build and return the NetworkX graph plus printable/exportable records."""

    require_networkx()
    vehicle_list = list(vehicles if vehicles is not None else example_vehicles())
    operations, operations_by_vehicle = build_operations(vehicle_list)

    graph = nx.DiGraph()
    graph.graph["name"] = "single_intersection_jssp_example"
    graph.graph["conflict_zones"] = CONFLICT_ZONES
    graph.graph["route_to_conflict_zones"] = ROUTE_TO_CONFLICT_ZONES

    add_operation_nodes(graph, operations)
    type1_edges = add_type1_precedence_edges(graph, operations_by_vehicle)
    type2_edges = add_type2_same_lane_edges(graph, operations_by_vehicle)
    type3_edges = add_type3_conflict_edges(graph, operations)

    records = {
        "vehicles": vehicle_list,
        "operations": operations,
        "type1_edges": type1_edges,
        "type2_edges": type2_edges,
        "type3_edges": type3_edges,
    }
    return graph, records


def print_operations(operations: Iterable[Operation]) -> None:
    print("\nNodes: vehicle-conflict-zone operations")
    print(
        "node_id                 vehicle  type       zone  route   source  arrival  proc_time"
    )
    print("-" * 88)
    for operation in operations:
        print(
            f"{operation.node_id:<23} "
            f"{operation.vehicle_id:<8} "
            f"{operation.vehicle_type:<10} "
            f"{operation.conflict_zone:<5} "
            f"{operation.route_id:<7} "
            f"{operation.source_lane:<7} "
            f"{operation.arrival_time:>7.1f} "
            f"{operation.processing_time:>9.1f}"
        )


def print_edges(title: str, edges: Iterable[EdgeRecord], paired: bool = False) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for edge in edges:
        connector = " <-> " if paired else " -> "
        detail_parts = [edge.description]
        if edge.source_lane:
            detail_parts.append(f"lane={edge.source_lane}")
        if edge.conflict_zone:
            detail_parts.append(f"zone={edge.conflict_zone}")
        if edge.candidate_pair_id:
            detail_parts.append(f"pair={edge.candidate_pair_id}")
        detail = ", ".join(detail_parts)
        print(f"{edge.source}{connector}{edge.target}  ({detail})")


def graph_to_jsonable(graph: Any, records: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "graph_name": graph.graph["name"],
        "conflict_zones": graph.graph["conflict_zones"],
        "route_to_conflict_zones": graph.graph["route_to_conflict_zones"],
        "vehicles": [
            {
                "vehicle_id": vehicle.vehicle_id,
                "vehicle_type": vehicle.vehicle_type,
                "route_id": vehicle.route_id,
                "source_lane": vehicle.source_lane,
                "arrival_time": vehicle.arrival_time,
            }
            for vehicle in records["vehicles"]
        ],
        "nodes": [
            {"node_id": node_id, **attributes}
            for node_id, attributes in graph.nodes(data=True)
        ],
        "directed_edges": [
            {"source": source, "target": target, **attributes}
            for source, target, attributes in graph.edges(data=True)
        ],
        "type3_candidate_pairs": [
            {
                "candidate_pair_id": edge.candidate_pair_id,
                "source": edge.source,
                "target": edge.target,
                "conflict_zone": edge.conflict_zone,
                "description": edge.description,
            }
            for edge in records["type3_edges"]
        ],
    }


def export_json(path: Path, graph: Any, records: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(graph_to_jsonable(graph, records), handle, indent=2)
        handle.write("\n")


def operation_positions(records: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
    positions: Dict[str, Tuple[float, float]] = {}
    vehicles = sorted(records["vehicles"], key=lambda veh: (veh.arrival_time, veh.vehicle_id))
    vehicle_y = {vehicle.vehicle_id: -index for index, vehicle in enumerate(vehicles)}

    for operation in records["operations"]:
        positions[operation.node_id] = (
            float(operation.operation_index),
            float(vehicle_y[operation.vehicle_id]),
        )
    return positions


def visualize_graph(path: Path, graph: Any, records: Dict[str, Any]) -> None:
    if plt is None or Line2D is None:
        raise SystemExit(
            "Missing dependency: matplotlib. Install dependencies with "
            "`python -m pip install -r requirements.txt`."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    positions = operation_positions(records)
    node_labels = {
        node_id: f"{attributes['vehicle_id']}\n{attributes['conflict_zone']}"
        for node_id, attributes in graph.nodes(data=True)
    }
    zone_colors = {
        "z1": "#4c78a8",
        "z2": "#f58518",
        "z3": "#54a24b",
        "z4": "#b279a2",
    }
    node_colors = [
        zone_colors[attributes["conflict_zone"]]
        for _, attributes in graph.nodes(data=True)
    ]

    type1_edges = [
        (source, target)
        for source, target, attrs in graph.edges(data=True)
        if attrs["edge_type"] == "type1_precedence"
    ]
    type2_edges = [
        (source, target)
        for source, target, attrs in graph.edges(data=True)
        if attrs["edge_type"] == "type2_same_lane_order"
    ]
    type3_pairs = [(edge.source, edge.target) for edge in records["type3_edges"]]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_title("Example JSSP Graph for One Unsignalized Intersection")

    nx.draw_networkx_nodes(
        graph,
        positions,
        node_color=node_colors,
        node_size=1450,
        edgecolors="#2f2f2f",
        linewidths=1.0,
        ax=ax,
    )
    nx.draw_networkx_labels(
        graph,
        positions,
        labels=node_labels,
        font_size=8,
        font_color="white",
        ax=ax,
    )
    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=type1_edges,
        edge_color="#2f2f2f",
        arrows=True,
        arrowstyle="-|>",
        width=1.8,
        connectionstyle="arc3,rad=0.05",
        ax=ax,
    )
    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=type2_edges,
        edge_color="#1f77b4",
        arrows=True,
        arrowstyle="-|>",
        style="dashed",
        width=1.6,
        connectionstyle="arc3,rad=0.18",
        ax=ax,
    )
    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=type3_pairs,
        edge_color="#d62728",
        arrows=False,
        style="dotted",
        width=1.2,
        alpha=0.7,
        ax=ax,
    )

    legend_items = [
        Line2D([0], [0], color="#2f2f2f", lw=2, label="Type-1 route precedence"),
        Line2D([0], [0], color="#1f77b4", lw=2, ls="--", label="Type-2 same-lane order"),
        Line2D([0], [0], color="#d62728", lw=2, ls=":", label="Type-3 conflict candidate"),
    ]
    ax.legend(
        handles=legend_items,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=True,
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an example JSSP conflict graph for the SUMO intersection."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for optional JSON and PNG outputs (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Do not export the graph JSON file.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not write the matplotlib graph visualization.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph, records = build_jssp_graph()

    print("Conflict zones")
    for zone_id, description in CONFLICT_ZONES.items():
        print(f"{zone_id}: {description}")

    print("\nRoute to conflict-zone mapping")
    for route_id, zones in ROUTE_TO_CONFLICT_ZONES.items():
        print(f"{route_id}: {' -> '.join(zones)}")

    print_operations(records["operations"])
    print_edges("Type-1 edges: same-vehicle route precedence", records["type1_edges"])
    print_edges("Type-2 edges: same-lane arrival order", records["type2_edges"])
    print_edges(
        "Type-3 edges: paired conflict candidates",
        records["type3_edges"],
        paired=True,
    )

    print(
        f"\nNetworkX graph object: {graph.__class__.__name__} "
        f"with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} directed edges"
    )

    if not args.no_json:
        json_path = args.output_dir / "intersection_jssp_graph.json"
        export_json(json_path, graph, records)
        print(f"Exported JSON: {json_path}")

    if not args.no_plot:
        png_path = args.output_dir / "intersection_jssp_graph.png"
        visualize_graph(png_path, graph, records)
        print(f"Exported visualization: {png_path}")


if __name__ == "__main__":
    main()
