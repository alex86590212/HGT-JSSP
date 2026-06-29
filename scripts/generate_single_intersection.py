"""Generate a simple SUMO network for one unsignalized four-way intersection.

The generated files are intended as the first simulation component for
intersection-scheduling experiments. The network has one inbound and one
outbound lane on each cardinal approach, no traffic light, simple route IDs,
and deterministic random demand.
"""

from __future__ import annotations

import argparse
import random
import shutil
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree as ET


DESIRED_INCOMING_LANE_LENGTH_M = 200.0
DESIRED_OUTGOING_LANE_LENGTH_M = 100.0
# SUMO shortens visible edge lanes near the internal junction. Extending the
# plain edge geometry by this small allowance keeps the compiled lane lengths
# close to the desired values above.
JUNCTION_TRIM_ALLOWANCE_M = 10.0
INCOMING_ARM_M = DESIRED_INCOMING_LANE_LENGTH_M + JUNCTION_TRIM_ALLOWANCE_M
OUTGOING_ARM_M = DESIRED_OUTGOING_LANE_LENGTH_M + JUNCTION_TRIM_ALLOWANCE_M
DETECTOR_DISTANCE_FROM_START_M = DESIRED_INCOMING_LANE_LENGTH_M - 50.0
LANE_OFFSET_M = 3.2
EDGE_SPEED_MPS = 13.89  # 50 km/h

DEFAULT_OUTPUT_DIR = Path("sumo") / "single_intersection"
DEFAULT_HARD_VEHICLE_COUNT = 160

# Node coordinates keep inbound and outbound lanes on the same arm separated.
# All lanes meet at J0, which is the only intersection node. J0 is unregulated:
# signal-free, with no built-in SUMO right-of-way. The external scheduler is
# responsible for deciding which vehicle may enter.
NODES = [
    ("J0", 0.0, 0.0, "unregulated"),
    ("N_src", -LANE_OFFSET_M, INCOMING_ARM_M, None),
    ("N_sink", LANE_OFFSET_M, OUTGOING_ARM_M, None),
    ("S_src", LANE_OFFSET_M, -INCOMING_ARM_M, None),
    ("S_sink", -LANE_OFFSET_M, -OUTGOING_ARM_M, None),
    ("E_src", INCOMING_ARM_M, LANE_OFFSET_M, None),
    ("E_sink", OUTGOING_ARM_M, -LANE_OFFSET_M, None),
    ("W_src", -INCOMING_ARM_M, -LANE_OFFSET_M, None),
    ("W_sink", -OUTGOING_ARM_M, LANE_OFFSET_M, None),
]

EDGES = [
    {
        "id": "N_in",
        "from": "N_src",
        "to": "J0",
        "shape": f"{-LANE_OFFSET_M},{INCOMING_ARM_M} {-LANE_OFFSET_M},8 0,0",
    },
    {
        "id": "N_out",
        "from": "J0",
        "to": "N_sink",
        "shape": f"0,0 {LANE_OFFSET_M},8 {LANE_OFFSET_M},{OUTGOING_ARM_M}",
    },
    {
        "id": "S_in",
        "from": "S_src",
        "to": "J0",
        "shape": f"{LANE_OFFSET_M},{-INCOMING_ARM_M} {LANE_OFFSET_M},-8 0,0",
    },
    {
        "id": "S_out",
        "from": "J0",
        "to": "S_sink",
        "shape": f"0,0 {-LANE_OFFSET_M},-8 {-LANE_OFFSET_M},{-OUTGOING_ARM_M}",
    },
    {
        "id": "E_in",
        "from": "E_src",
        "to": "J0",
        "shape": f"{INCOMING_ARM_M},{LANE_OFFSET_M} 8,{LANE_OFFSET_M} 0,0",
    },
    {
        "id": "E_out",
        "from": "J0",
        "to": "E_sink",
        "shape": f"0,0 8,{-LANE_OFFSET_M} {OUTGOING_ARM_M},{-LANE_OFFSET_M}",
    },
    {
        "id": "W_in",
        "from": "W_src",
        "to": "J0",
        "shape": f"{-INCOMING_ARM_M},{-LANE_OFFSET_M} -8,{-LANE_OFFSET_M} 0,0",
    },
    {
        "id": "W_out",
        "from": "J0",
        "to": "W_sink",
        "shape": f"0,0 -8,{LANE_OFFSET_M} {-OUTGOING_ARM_M},{LANE_OFFSET_M}",
    },
]

# Route IDs use the convention <origin approach>_to_<destination approach>.
# The route ID deliberately does not encode "left", "right", or "straight";
# those movement classes can be derived from the origin/destination pair.
ROUTES = [
    ("N_to_S", "N_in S_out", "straight"),
    ("N_to_E", "N_in E_out", "left"),
    ("N_to_W", "N_in W_out", "right"),
    ("S_to_N", "S_in N_out", "straight"),
    ("S_to_W", "S_in W_out", "left"),
    ("S_to_E", "S_in E_out", "right"),
    ("E_to_W", "E_in W_out", "straight"),
    ("E_to_S", "E_in S_out", "left"),
    ("E_to_N", "E_in N_out", "right"),
    ("W_to_E", "W_in E_out", "straight"),
    ("W_to_N", "W_in N_out", "left"),
    ("W_to_S", "W_in S_out", "right"),
]

HARD_ROUTE_BURSTS = [
    # Four vehicles arrive close together from different approaches. These
    # bursts create many same-zone conflicts for the external scheduler.
    ["E_to_W", "N_to_S", "W_to_E", "S_to_N"],
    ["E_to_S", "N_to_E", "W_to_N", "S_to_W"],
    ["E_to_N", "N_to_W", "W_to_S", "S_to_E"],
    ["E_to_W", "N_to_E", "S_to_N", "W_to_S"],
    ["N_to_S", "E_to_S", "W_to_E", "S_to_W"],
]

VEHICLE_TYPES = [
    {
        "id": "passenger",
        "vClass": "passenger",
        "accel": "2.6",
        "decel": "4.5",
        "sigma": "0.5",
        "length": "5.0",
        "minGap": "2.5",
        "maxSpeed": f"{EDGE_SPEED_MPS}",
        "probability_weight": 0.70,
    },
    {
        "id": "delivery",
        "vClass": "delivery",
        "accel": "2.0",
        "decel": "4.0",
        "sigma": "0.5",
        "length": "7.0",
        "minGap": "3.0",
        "maxSpeed": "12.50",
        "probability_weight": 0.20,
    },
    {
        "id": "truck",
        "vClass": "truck",
        "accel": "1.3",
        "decel": "3.5",
        "sigma": "0.5",
        "length": "12.0",
        "minGap": "3.5",
        "maxSpeed": "11.11",
        "probability_weight": 0.10,
    },
]


def add_comment(parent: ET.Element, text: str) -> None:
    parent.append(ET.Comment(f" {text} "))


def indent_xml(element: ET.Element, level: int = 0) -> None:
    indent = "\n" + ("    " * level)
    child_indent = "\n" + ("    " * (level + 1))

    if len(element):
        if not element.text or not element.text.strip():
            element.text = child_indent
        for child in element:
            indent_xml(child, level + 1)
        if not element[-1].tail or not element[-1].tail.strip():
            element[-1].tail = indent
    if level and (not element.tail or not element.tail.strip()):
        element.tail = indent


def write_xml(root: ET.Element, path: Path) -> None:
    indent_xml(root)
    tree = ET.ElementTree(root)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def write_nodes(path: Path) -> None:
    root = ET.Element("nodes")
    add_comment(
        root,
        "J0 is signal-free and unregulated; source/sink nodes define the road arms.",
    )
    for node_id, x_coord, y_coord, node_type in NODES:
        attrs = {"id": node_id, "x": f"{x_coord:.1f}", "y": f"{y_coord:.1f}"}
        if node_type:
            attrs["type"] = node_type
        ET.SubElement(root, "node", attrs)
    write_xml(root, path)


def write_edges(path: Path) -> None:
    root = ET.Element("edges")
    add_comment(
        root,
        "Each approach has one inbound edge (*_in, about 200 m) and one outbound edge (*_out, about 100 m).",
    )
    for edge in EDGES:
        attrs = {
            "id": edge["id"],
            "from": edge["from"],
            "to": edge["to"],
            "numLanes": "1",
            "speed": f"{EDGE_SPEED_MPS:.2f}",
            "priority": "1",
            "shape": edge["shape"],
        }
        ET.SubElement(root, "edge", attrs)
    write_xml(root, path)


def write_connections(path: Path) -> None:
    root = ET.Element("connections")
    add_comment(
        root,
        "Only straight/left/right movements are allowed. U-turns are intentionally omitted.",
    )
    for route_id, edge_sequence, movement in ROUTES:
        from_edge, to_edge = edge_sequence.split()
        ET.SubElement(
            root,
            "connection",
            {
                "from": from_edge,
                "to": to_edge,
                "fromLane": "0",
                "toLane": "0",
                "dir": movement[0],
            },
        )
    write_xml(root, path)


def write_routes(path: Path) -> None:
    root = ET.Element("routes")
    add_comment(
        root,
        "Vehicle types and static routes. Demand is kept in demand.rou.xml.",
    )
    add_comment(
        root,
        "Route IDs follow <origin>_to_<destination>, e.g. E_to_W means enter from east and leave west.",
    )
    add_comment(
        root,
        "Straight: N_to_S, S_to_N, E_to_W, W_to_E. Left: N_to_E, S_to_W, E_to_S, W_to_N. Right: N_to_W, S_to_E, E_to_N, W_to_S.",
    )

    for vehicle_type in VEHICLE_TYPES:
        attrs = {
            key: value
            for key, value in vehicle_type.items()
            if key != "probability_weight"
        }
        ET.SubElement(root, "vType", attrs)

    for route_id, edge_sequence, movement in ROUTES:
        ET.SubElement(
            root,
            "route",
            {
                "id": route_id,
                "edges": edge_sequence,
                "description": movement,
            },
        )
    write_xml(root, path)


def write_demand(
    path: Path,
    *,
    seed: int,
    vehicle_count: int,
    begin: float,
    end: float,
) -> None:
    rng = random.Random(seed)
    root = ET.Element("routes")
    add_comment(
        root,
        f"Random deterministic demand generated with seed={seed}, vehicles={vehicle_count}, begin={begin}, end={end}.",
    )
    add_comment(
        root,
        "Vehicles reference route IDs from routes.rou.xml so TraCI can map route_id -> conflict-zone sequence.",
    )

    route_ids = [route_id for route_id, _, _ in ROUTES]
    type_ids = [vehicle_type["id"] for vehicle_type in VEHICLE_TYPES]
    type_weights = [
        vehicle_type["probability_weight"] for vehicle_type in VEHICLE_TYPES
    ]
    departures = sorted(rng.uniform(begin, end) for _ in range(vehicle_count))

    for index, depart in enumerate(departures):
        route_id = rng.choice(route_ids)
        vehicle_type = rng.choices(type_ids, weights=type_weights, k=1)[0]
        ET.SubElement(
            root,
            "vehicle",
            {
                "id": f"veh_{index:04d}",
                "type": vehicle_type,
                "route": route_id,
                "depart": f"{depart:.1f}",
                "departLane": "best",
                "departSpeed": "max",
            },
        )
    write_xml(root, path)


def write_hard_demand(
    path: Path,
    *,
    seed: int,
    vehicle_count: int,
    begin: float,
    end: float,
) -> None:
    rng = random.Random(seed)
    root = ET.Element("routes")
    add_comment(
        root,
        f"Hard deterministic demand generated with seed={seed}, vehicles={vehicle_count}.",
    )
    add_comment(
        root,
        "Bursty four-approach arrivals create dense queues and many shared conflict-zone candidates.",
    )
    add_comment(
        root,
        "Use with single_intersection_hard.sumocfg for scheduler stress tests.",
    )

    type_ids = [vehicle_type["id"] for vehicle_type in VEHICLE_TYPES]
    # Hard scenario uses more large/slow vehicles than the baseline.
    type_weights = [0.50, 0.30, 0.20]
    burst_gap = 6.0
    intra_burst_gap = 0.9
    depart_base = begin + 5.0

    for index in range(vehicle_count):
        burst_index = index // 4
        offset_index = index % 4
        burst_routes = HARD_ROUTE_BURSTS[burst_index % len(HARD_ROUTE_BURSTS)]
        route_id = burst_routes[offset_index]
        depart = depart_base + (burst_index * burst_gap) + (offset_index * intra_burst_gap)
        depart += rng.uniform(-0.25, 0.25)
        depart = max(begin, min(end - 1.0, depart))
        vehicle_type = rng.choices(type_ids, weights=type_weights, k=1)[0]
        ET.SubElement(
            root,
            "vehicle",
            {
                "id": f"hard_veh_{index:04d}",
                "type": vehicle_type,
                "route": route_id,
                "depart": f"{depart:.1f}",
                "departLane": "best",
                "departSpeed": "max",
            },
        )
    write_xml(root, path)


def write_additional(path: Path) -> None:
    root = ET.Element("additional")
    add_comment(
        root,
        "Optional E1 detectors on inbound lanes, placed about 50 m before the junction.",
    )
    for approach in ["N", "S", "E", "W"]:
        ET.SubElement(
            root,
            "inductionLoop",
            {
                "id": f"det_{approach}_in_50m",
                "lane": f"{approach}_in_0",
                "pos": f"{DETECTOR_DISTANCE_FROM_START_M:.1f}",
                "period": "1.0",
                "file": "detectors.out.xml",
                "friendlyPos": "true",
            },
        )
    write_xml(root, path)


def write_sumocfg(
    path: Path,
    *,
    begin: float,
    end: float,
    route_files: str = "routes.rou.xml,demand.rou.xml",
) -> None:
    root = ET.Element("configuration")

    input_group = ET.SubElement(root, "input")
    ET.SubElement(input_group, "net-file", {"value": "single_intersection.net.xml"})
    ET.SubElement(
        input_group,
        "route-files",
        {"value": route_files},
    )
    ET.SubElement(input_group, "additional-files", {"value": "additional.add.xml"})

    time_group = ET.SubElement(root, "time")
    ET.SubElement(time_group, "begin", {"value": f"{begin:.1f}"})
    ET.SubElement(time_group, "end", {"value": f"{end:.1f}"})
    ET.SubElement(time_group, "step-length", {"value": "0.1"})

    report_group = ET.SubElement(root, "report")
    ET.SubElement(report_group, "no-step-log", {"value": "true"})

    processing_group = ET.SubElement(root, "processing")
    ET.SubElement(processing_group, "collision.action", {"value": "warn"})

    write_xml(root, path)


def write_readme(
    path: Path,
    *,
    seed: int,
    vehicle_count: int,
    hard_vehicle_count: int,
) -> None:
    readme = f"""# SUMO Single Unsignalized Intersection

This folder contains a simple four-way, signal-free SUMO network for later
mapping of vehicle routes to manually defined conflict-zone sequences.

## Network Structure

- `J0` is the only intersection node and has no traffic light.
- `J0` is `unregulated`, so SUMO does not impose a right-of-way rule after the
  external scheduler releases a vehicle.
- Each cardinal approach has one inbound edge and one outbound edge:
  `N_in`/`N_out`, `S_in`/`S_out`, `E_in`/`E_out`, `W_in`/`W_out`.
- Inbound roads are about `{DESIRED_INCOMING_LANE_LENGTH_M:.0f}` m long.
- Outbound roads are about `{DESIRED_OUTGOING_LANE_LENGTH_M:.0f}` m long.
- `additional.add.xml` adds optional E1 detectors on inbound lanes about 50 m
  before the intersection.

## Route Naming

Route IDs use `<origin>_to_<destination>`:

- Straight: `N_to_S`, `S_to_N`, `E_to_W`, `W_to_E`
- Left turns: `N_to_E`, `S_to_W`, `E_to_S`, `W_to_N`
- Right turns: `N_to_W`, `S_to_E`, `E_to_N`, `W_to_S`

This naming is intentionally simple so an external scheduler can map a SUMO
route ID directly to a conflict-zone sequence.

## Files

- `single_intersection.nod.xml`: plain node definitions
- `single_intersection.edg.xml`: plain edge definitions
- `single_intersection.con.xml`: allowed turning connections
- `single_intersection.net.xml`: compiled SUMO network
- `routes.rou.xml`: vehicle types and static route definitions
- `demand.rou.xml`: random deterministic demand (`seed={seed}`,
  `vehicles={vehicle_count}`)
- `hard_demand.rou.xml`: bursty stress-test demand (`vehicles={hard_vehicle_count}`)
- `additional.add.xml`: optional inbound lane detectors
- `single_intersection.sumocfg`: runnable SUMO configuration
- `single_intersection_hard.sumocfg`: denser hard scenario for scheduler tests

## Controller Stop Line

The TraCI controllers stop held vehicles close to the intersection by default,
about 8 m before the end of each inbound lane. The hold command is still issued
farther upstream so vehicles have enough room to brake. Because vehicles are
held very near the junction, the controller includes a small lane-end fallback:
if a released vehicle is clipped exactly at the inbound lane end, it is moved a
few centimeters onto the open internal link that SUMO reports for its route.

## Hard Scenario

The hard scenario sends repeated four-approach bursts through the intersection,
with more delivery vehicles and trucks than the baseline. It is intended to
produce queues and many Type-2/Type-3 scheduling constraints for testing FCFS
and later PPO-GNN schedulers.

## Run

From this repository root:

```powershell
sumo-gui -c sumo\\single_intersection\\single_intersection.sumocfg
```

Hard scenario:

```powershell
sumo-gui -c sumo\\single_intersection\\single_intersection_hard.sumocfg
```

Hard scenario with the live FCFS/JSSP debugger:

```powershell
.venv\\Scripts\\python.exe scripts\\run_visual_debug_controller.py --config sumo\\single_intersection\\single_intersection_hard.sumocfg
```

Or from this folder:

```powershell
sumo-gui -c single_intersection.sumocfg
```

Regenerate the files with a different seed or demand size:

```powershell
python scripts\\generate_single_intersection.py --seed 7 --vehicle-count 200
```

Regenerate a larger hard scenario:

```powershell
python scripts\\generate_single_intersection.py --hard-vehicle-count 240
```

## TraCI Start Example

```python
import traci

traci.start([
    "sumo-gui",
    "-c",
    "sumo/single_intersection/single_intersection.sumocfg",
])
```
"""
    path.write_text(readme, encoding="utf-8")


def run_netconvert(output_dir: Path) -> None:
    netconvert = shutil.which("netconvert")
    if not netconvert:
        print(
            "netconvert was not found on PATH; plain XML files were generated, "
            "but single_intersection.net.xml was not compiled.",
            file=sys.stderr,
        )
        return

    command = [
        netconvert,
        "--node-files",
        str(output_dir / "single_intersection.nod.xml"),
        "--edge-files",
        str(output_dir / "single_intersection.edg.xml"),
        "--connection-files",
        str(output_dir / "single_intersection.con.xml"),
        "--output-file",
        str(output_dir / "single_intersection.net.xml"),
        "--no-turnarounds",
        "true",
    ]
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a SUMO four-way unsignalized intersection example."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for generated SUMO files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic demand generation.",
    )
    parser.add_argument(
        "--vehicle-count",
        type=int,
        default=80,
        help="Number of vehicles in demand.rou.xml.",
    )
    parser.add_argument(
        "--hard-vehicle-count",
        type=int,
        default=DEFAULT_HARD_VEHICLE_COUNT,
        help="Number of vehicles in hard_demand.rou.xml.",
    )
    parser.add_argument(
        "--begin",
        type=float,
        default=0.0,
        help="Simulation begin time in seconds.",
    )
    parser.add_argument(
        "--end",
        type=float,
        default=600.0,
        help="Simulation end time in seconds.",
    )
    parser.add_argument(
        "--skip-netconvert",
        action="store_true",
        help="Generate plain XML files without compiling the .net.xml.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.vehicle_count < 0:
        raise ValueError("--vehicle-count must be non-negative")
    if args.hard_vehicle_count < 0:
        raise ValueError("--hard-vehicle-count must be non-negative")
    if args.end <= args.begin:
        raise ValueError("--end must be greater than --begin")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    write_nodes(output_dir / "single_intersection.nod.xml")
    write_edges(output_dir / "single_intersection.edg.xml")
    write_connections(output_dir / "single_intersection.con.xml")
    write_routes(output_dir / "routes.rou.xml")
    write_demand(
        output_dir / "demand.rou.xml",
        seed=args.seed,
        vehicle_count=args.vehicle_count,
        begin=args.begin,
        end=args.end,
    )
    write_hard_demand(
        output_dir / "hard_demand.rou.xml",
        seed=args.seed + 1000,
        vehicle_count=args.hard_vehicle_count,
        begin=args.begin,
        end=args.end,
    )
    write_additional(output_dir / "additional.add.xml")
    write_sumocfg(output_dir / "single_intersection.sumocfg", begin=args.begin, end=args.end)
    write_sumocfg(
        output_dir / "single_intersection_hard.sumocfg",
        begin=args.begin,
        end=args.end,
        route_files="routes.rou.xml,hard_demand.rou.xml",
    )
    write_readme(
        output_dir / "README.md",
        seed=args.seed,
        vehicle_count=args.vehicle_count,
        hard_vehicle_count=args.hard_vehicle_count,
    )

    if not args.skip_netconvert:
        run_netconvert(output_dir)

    print(f"Generated SUMO intersection files in {output_dir}")


if __name__ == "__main__":
    main()
