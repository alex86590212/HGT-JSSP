from __future__ import annotations

import argparse
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

GRID_SIZE = 4
EDGE_LENGTH_M = 40.0
# Must be >= dynamic_scheduler.data.traffic_generator._VELOCITY_RANGE's upper
# bound (14.0 m/s) -- SUMO refuses a vehicle's departSpeed above the edge's
# speed limit ("unpriorised junction too close" at depart), so a lower cap
# here silently breaks departure for the faster fraction of vehicles.
SPEED_MPS = 15.0

HERE = Path(__file__).resolve().parent


def run_netgenerate(out_net: Path) -> None:
    if shutil.which("netgenerate") is None:
        raise RuntimeError("netgenerate not found on PATH. Install SUMO first (brew install sumo).")
    cmd = [
        "netgenerate",
        "--grid",
        "--grid.number", str(GRID_SIZE),
        "--grid.length", str(EDGE_LENGTH_M),
        "--default.speed", str(SPEED_MPS),
        "--no-turnarounds",
        "--tls.guess", "false",
        "--output-file", str(out_net),
    ]
    subprocess.run(cmd, check=True)


def strip_traffic_lights(net_path: Path) -> None:
    tree = ET.parse(net_path)
    root = tree.getroot()
    for tl in list(root.findall("tlLogic")):
        root.remove(tl)
    for junction in root.findall("junction"):
        if junction.get("type") == "traffic_light":
            junction.set("type", "priority")
        if "tl" in junction.attrib:
            del junction.attrib["tl"]
    for connection in root.findall("connection"):
        for attr in ("tl", "linkIndex"):
            if attr in connection.attrib:
                del connection.attrib[attr]
    tree.write(net_path)


def write_sumocfg(net_path: Path, route_path: Path, out_cfg: Path) -> None:
    cfg = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="{net_path.name}"/>
        <route-files value="{route_path.name}"/>
    </input>
    <time>
        <begin value="0"/>
    </time>
</configuration>
"""
    out_cfg.write_text(cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(HERE))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    net_path = out_dir / "grid_4x4.net.xml"

    run_netgenerate(net_path)
    strip_traffic_lights(net_path)

    route_path = out_dir / "grid_4x4.rou.xml"
    if not route_path.exists():
        route_path.write_text('<?xml version="1.0" encoding="UTF-8"?>\n<routes/>\n')

    write_sumocfg(net_path, route_path, out_dir / "grid_4x4.sumocfg")
    print(f"wrote {net_path}")


if __name__ == "__main__":
    main()
