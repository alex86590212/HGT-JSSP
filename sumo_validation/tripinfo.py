from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict


def read_time_loss(tripinfo_path: Path) -> Dict[int, float]:
    result: Dict[int, float] = {}
    if not tripinfo_path.exists():
        return result
    root = ET.parse(tripinfo_path).getroot()
    for trip in root.findall("tripinfo"):
        try:
            vid = int(trip.get("id"))
        except (TypeError, ValueError):
            continue
        time_loss = trip.get("timeLoss")
        if time_loss is None:
            continue
        result[vid] = float(time_loss)
    return result
