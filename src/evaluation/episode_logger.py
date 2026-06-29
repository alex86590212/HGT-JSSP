"""Per-episode artifact writer."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from src.common.models import EpisodeMetrics
from src.evaluation.metrics import write_metrics_csv, write_metrics_json


class EpisodeLogger:
    """Create and write the standard output files for one episode."""

    def __init__(self, output_root: Path, episode_id: str) -> None:
        self.output_root = output_root
        self.episode_id = episode_id
        self.logs_dir = output_root / "logs" / episode_id
        self.metrics_dir = output_root / "metrics" / episode_id
        self.debug_graphs_dir = output_root / "debug_graphs" / episode_id
        for directory in (self.logs_dir, self.metrics_dir, self.debug_graphs_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def path(self, name: str, category: str = "logs") -> Path:
        if category == "metrics":
            return self.metrics_dir / name
        if category == "debug_graphs":
            return self.debug_graphs_dir / name
        return self.logs_dir / name

    def write_csv(self, name: str, rows: Iterable[Mapping[str, Any]]) -> None:
        rows = list(rows)
        fieldnames: List[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        path = self.logs_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row))

    def write_json(self, name: str, payload: Any, category: str = "logs") -> None:
        path = self.path(name, category=category)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def write_metrics(self, metrics: EpisodeMetrics) -> None:
        write_metrics_json(self.metrics_dir / "metrics.json", metrics)
        write_metrics_csv(self.metrics_dir / "metrics.csv", [metrics])

    def write_reservations(self, schedule: Any) -> None:
        reservations = getattr(schedule, "operation_reservations", [])
        rows = [asdict(reservation) for reservation in reservations]
        self.write_csv("reservations.csv", rows)

    def write_empty_standard_logs(self) -> None:
        for filename in ("vehicle_events.csv", "reservations.csv", "control_actions.csv"):
            path = self.logs_dir / filename
            if not path.exists():
                with path.open("w", newline="", encoding="utf-8") as handle:
                    handle.write("")


__all__ = ["EpisodeLogger"]
