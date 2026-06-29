"""Scheduler interface used by the SUMO controller."""

from __future__ import annotations

from typing import Any, Iterable


class BaseScheduler:
    name = "base"

    def schedule(self, vehicle_records: Iterable[Any], jssp_graph: Any, current_time: float):
        raise NotImplementedError

