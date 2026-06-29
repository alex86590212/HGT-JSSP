"""Greedy scheduler placeholder.

This project is intentionally FCFS-only for now. The class exists so the CLI and
controller can be wired through a scheduler interface before learning policies
are added.
"""

from __future__ import annotations

from src.schedulers.base import BaseScheduler


class GreedyScheduler(BaseScheduler):
    name = "greedy"

    def schedule(self, vehicle_records, jssp_graph, current_time):
        raise NotImplementedError("Greedy scheduler is not implemented yet.")

