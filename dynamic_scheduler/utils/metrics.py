"""Metrics for the online dynamic 4x4 scheduler.

episode_waiting_time mirrors intersection_scheduler.utils.metrics'
definition (finish - arrival - sum(processing_times), averaged per vehicle)
but adapted for the online env: only vehicles that reached a LOCKED final
operation by episode end have a well-defined waiting time — a vehicle still
mid-route (or not yet detected) at the episode cutoff has no finish time yet
and is excluded, matching how a real deployment would only report completed
trips. Waiting time is captured at vehicle-removal time in
DynamicIntersectionEnv.completed_log (see _remove_completed_vehicles), since
the vehicle/its operations are gone from the env afterward.
"""

from __future__ import annotations

from typing import List


def episode_waiting_time(completed_log: List[dict]) -> float:
    """Mean waiting time across vehicles that completed during the episode.

    completed_log: env.completed_log after a full episode — list of dicts
    with a "waiting_time" key (see DynamicIntersectionEnv._remove_completed_vehicles).
    Returns 0.0 if no vehicle completed (matches the offline convention of
    returning 0.0 for an empty vehicle set).
    """
    if not completed_log:
        return 0.0
    return sum(entry["waiting_time"] for entry in completed_log) / len(completed_log)


def episode_makespan(completed_log: List[dict]) -> float:
    """Latest finish time among completed vehicles, or 0.0 if none completed."""
    if not completed_log:
        return 0.0
    return max(entry["finish_time"] for entry in completed_log)
