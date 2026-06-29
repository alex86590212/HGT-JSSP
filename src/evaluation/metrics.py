"""Metrics helpers for FCFS intersection experiments."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from src.common.models import EpisodeMetrics


def metrics_to_dict(metrics: EpisodeMetrics) -> Dict[str, Any]:
    payload = asdict(metrics)
    for class_name, delay in metrics.class_delay.items():
        payload[f"{class_name}_delay"] = delay
    return payload


def compute_episode_metrics(
    *,
    episode_id: str,
    sim_time: float,
    total_vehicles_completed: int,
    number_of_stops: int,
    scheduler_runtime_avg_ms: float,
    safety_violations: int,
    sumo_collisions: int,
    deadlock_or_timeout_events: int = 0,
    class_delay: Optional[Mapping[str, float]] = None,
    total_delay: float = 0.0,
    total_waiting_time: float = 0.0,
    maximum_waiting_time: float = 0.0,
) -> EpisodeMetrics:
    horizon = max(sim_time, 1e-9)
    throughput = total_vehicles_completed / horizon
    average_delay = total_delay / total_vehicles_completed if total_vehicles_completed else 0.0
    average_waiting_time = (
        total_waiting_time / total_vehicles_completed if total_vehicles_completed else 0.0
    )
    return EpisodeMetrics(
        episode_id=episode_id,
        total_vehicles_completed=total_vehicles_completed,
        throughput=throughput,
        average_delay=average_delay,
        total_delay=total_delay,
        average_waiting_time=average_waiting_time,
        maximum_waiting_time=maximum_waiting_time,
        number_of_stops=number_of_stops,
        scheduler_runtime_avg_ms=scheduler_runtime_avg_ms,
        safety_violations=safety_violations,
        sumo_collisions=sumo_collisions,
        deadlock_or_timeout_events=deadlock_or_timeout_events,
        class_delay=dict(class_delay or {"car": 0.0, "truck": 0.0, "bus": 0.0}),
    )


def write_metrics_json(path: Path, metrics: EpisodeMetrics) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metrics_to_dict(metrics), handle, indent=2)


def write_metrics_csv(path: Path, metrics_rows: Iterable[EpisodeMetrics]) -> None:
    rows = [metrics_to_dict(metrics) for metrics in metrics_rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_metrics_summary(path_json: Path, path_csv: Path, metrics_rows: Iterable[EpisodeMetrics]) -> None:
    rows = list(metrics_rows)
    path_json.parent.mkdir(parents=True, exist_ok=True)
    with path_json.open("w", encoding="utf-8") as handle:
        json.dump([metrics_to_dict(metrics) for metrics in rows], handle, indent=2)
    write_metrics_csv(path_csv, rows)


__all__ = [
    "compute_episode_metrics",
    "metrics_to_dict",
    "write_metrics_csv",
    "write_metrics_json",
    "write_metrics_summary",
]
