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
    """Mean waiting time across vehicles that COMPLETED during the episode.

    NOTE: this excludes vehicles still in flight at the episode cutoff, which
    biases the number downward (the most-delayed vehicles are often the ones
    that didn't finish). For an unbiased number use
    episode_waiting_time_all() below. Kept for completeness / comparison.
    """
    if not completed_log:
        return 0.0
    return sum(entry["waiting_time"] for entry in completed_log) / len(completed_log)


def episode_waiting_time_all(completed_log: List[dict], inflight_times: List[float]) -> float:
    """Mean waiting time over ALL vehicles seen — completed plus still-in-flight.

    inflight_times: env.inflight_waiting_times() captured at episode end.
    Including in-flight vehicles removes the survivorship bias of
    episode_waiting_time(): a vehicle stuck in congestion at the cutoff
    counts its accumulated delay rather than being dropped. This is the
    metric to report and to select checkpoints on.
    """
    all_times = [e["waiting_time"] for e in completed_log] + list(inflight_times)
    if not all_times:
        return 0.0
    return sum(all_times) / len(all_times)


def completion_rate(n_completed: int, n_seen: int) -> float:
    """Fraction of detected vehicles that completed within the episode.

    A low completion rate means many vehicles never cleared — the scheduler
    is falling behind the arrival rate. Reported alongside waiting time so a
    low waiting time on few completions isn't mistaken for good performance.
    """
    return n_completed / n_seen if n_seen > 0 else 0.0


def episode_makespan(completed_log: List[dict]) -> float:
    """Latest finish time among completed vehicles, or 0.0 if none completed."""
    if not completed_log:
        return 0.0
    return max(entry["finish_time"] for entry in completed_log)
