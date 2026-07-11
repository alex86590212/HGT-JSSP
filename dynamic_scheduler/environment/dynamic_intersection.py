"""DynamicIntersectionEnv: online receding-horizon MDP for the 4x4 intersection.

Unlike intersection_scheduler.environment.IntersectionEnv (fixed vehicle batch,
one-shot irrevocable scheduling), this environment models continuous vehicle
arrivals and revisable planning:

  - Vehicles arrive via a Poisson process over a fixed wall-clock episode
    duration, not a known-upfront fixed batch.
  - A vehicle's operations only enter the graph once detected (within
    `detection_window` seconds of its arrival_time) — matches a realistic
    roadside sensor/V2X range.
  - Each operation is UNSCHEDULED -> TENTATIVE -> LOCKED, instead of the
    offline model's binary scheduled/unscheduled:
      * UNSCHEDULED: not yet detected, or detected but not yet planned.
      * TENTATIVE: has a planned start/finish time, still revisable.
      * LOCKED: vehicle is within `commit_window` seconds of arrival_time —
        no longer revisable (mirrors "too close to safely replan").
  - Replanning is event-driven: a replan pass happens whenever a new vehicle
    is detected. All TENTATIVE operations may be revised in that pass; LOCKED
    operations are frozen.
  - Vehicles whose last operation is LOCKED and finished are removed from the
    graph entirely (decision-irrelevant once complete).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch


class OpState(Enum):
    UNSCHEDULED = 0
    TENTATIVE = 1
    LOCKED = 2


@dataclass
class DynamicVehicle:
    id: int
    arrival_time: float        # ground-truth time it reaches the intersection
    route: List[int]
    processing_times: List[float]
    velocity: float
    manoeuvre: str = ""
    detected: bool = False     # has entered the detection window


@dataclass
class DynamicOperation:
    vehicle_id: int
    zone_id: int
    route_position: int
    route_length: int
    processing_time: float
    state: OpState = OpState.UNSCHEDULED
    start_time: float = 0.0
    earliest_finish: float = 0.0
    # Timing at the moment this op was last tentatively planned — used to
    # compute the rescheduling penalty when a replan changes start_time.
    prev_start_time: Optional[float] = None


@dataclass
class DynamicZone:
    id: int
    x: float
    y: float
    time_free: float = 0.0
    n_competing: int = 0


class DynamicIntersectionEnv:
    """Online receding-horizon scheduling environment.

    Time advances via `advance_to(t)` (jump to the next event: a new vehicle
    detection, a lock transition, or episode end) rather than one op-schedule
    per step as in the offline env. A "step" in this environment is a full
    replan pass over all currently open (TENTATIVE + newly UNSCHEDULED but
    detected) operations, driven externally by the training loop choosing one
    operation per policy call within a replan pass — mirroring the offline
    env's action-per-call semantics so run_episode-style loops still work.
    """

    def __init__(
        self,
        detection_window: float = 10.0,
        commit_window: float = 2.5,
        zone_positions: Optional[Dict[int, Tuple[float, float]]] = None,
        penalty_coef: float = 0.1,
        max_proximity_weight: float = 2.0,
    ) -> None:
        self.detection_window = detection_window
        self.commit_window = commit_window
        self._zone_positions = zone_positions
        # Rescheduling-penalty tuning. penalty_coef scales the whole penalty
        # term relative to the completion-time (waiting) objective, so the
        # penalty doesn't drown the primary signal. max_proximity_weight caps
        # the 1/time_to_arrival weight, which is otherwise unbounded and
        # explodes for near-arrival vehicles (was producing -6+ single-step
        # penalties that dominated the reward).
        self.penalty_coef = penalty_coef
        self.max_proximity_weight = max_proximity_weight

        self.vehicles: Dict[int, DynamicVehicle] = {}
        self.operations: List[DynamicOperation] = []   # flat list, index = action
        self.zones: Dict[int, DynamicZone] = {}
        self.current_time: float = 0.0
        self.episode_duration: float = 0.0

        # Pending arrivals not yet detected: list of DynamicVehicle sorted by
        # arrival_time, populated at reset from the traffic generator.
        self._pending_arrivals: List[DynamicVehicle] = []
        self._next_vehicle_id: int = 0

        # Type-3 conflict tracking, same semantics as the offline env but
        # rebuilt whenever the active operation set changes (new detection or
        # a vehicle's completion), since the vehicle set is not fixed here.
        self.conflict_edges: List[Tuple[int, int]] = []
        self.active_conflict_edges: List[Tuple[int, int]] = []

        # Normalisation stats — recomputed each replan since the active set
        # of vehicles/ops changes over the episode (no fixed max at reset).
        self.norm_max_p: float = 1.0
        self.norm_max_c: float = 1.0
        self.norm_max_r: float = 1.0

        self._prev_completion_times: Dict[int, float] = {}

        # Log of vehicles removed after completing their route — captured at
        # removal time since env.vehicles/operations no longer hold them
        # afterward. Each entry: {vehicle_id, arrival_time, finish_time,
        # waiting_time}. Cleared at reset; read by dynamic_scheduler.utils.metrics.
        self.completed_log: List[dict] = []

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(
        self,
        arrivals: List[DynamicVehicle],
        episode_duration: float,
    ) -> "DynamicIntersectionEnv":
        """Start a new episode.

        arrivals: full list of vehicles that will arrive during this episode,
            pre-generated by the traffic generator (Poisson process). This is
            the ground truth used to reveal vehicles as time passes — the
            POLICY never sees vehicles before they're detected, but the env
            needs the full schedule to know what to reveal and when.
        episode_duration: wall-clock length of the episode in seconds.
        """
        self.vehicles = {}
        self.operations = []
        self.zones = {}
        self.current_time = 0.0
        self.episode_duration = episode_duration
        self.conflict_edges = []
        self.active_conflict_edges = []
        self._prev_completion_times = {}
        self.completed_log = []

        self._pending_arrivals = sorted(arrivals, key=lambda v: v.arrival_time)
        self._next_vehicle_id = (max((v.id for v in arrivals), default=-1) + 1)

        # Populate zone table up front (zone geometry is static across the
        # whole episode, unlike vehicles) using 4x4 positions if given.
        if self._zone_positions is not None:
            for zid, (x, y) in self._zone_positions.items():
                self.zones[zid] = DynamicZone(id=zid, x=x / 3.0, y=y / 3.0)

        # Reveal any vehicles that are already within the detection window
        # at t=0 (arrival_time <= detection_window).
        self._detect_new_vehicles()

        return self

    # ------------------------------------------------------------------
    # Detection / lifecycle
    # ------------------------------------------------------------------

    def _detect_new_vehicles(self) -> List[int]:
        """Move any pending vehicle within detection_window into self.vehicles.

        Returns the list of newly detected vehicle ids (triggers a replan).
        """
        newly_detected: List[int] = []
        still_pending: List[DynamicVehicle] = []

        for v in self._pending_arrivals:
            if v.arrival_time - self.current_time <= self.detection_window + 1e-9:
                v.detected = True
                self.vehicles[v.id] = v
                for j, (zone_id, pt) in enumerate(zip(v.route, v.processing_times)):
                    self.operations.append(DynamicOperation(
                        vehicle_id=v.id,
                        zone_id=zone_id,
                        route_position=j,
                        route_length=len(v.route),
                        processing_time=pt,
                    ))
                    if zone_id not in self.zones:
                        # Fallback zone if zone_positions wasn't supplied.
                        self.zones[zone_id] = DynamicZone(id=zone_id, x=0.0, y=0.0)
                newly_detected.append(v.id)
            else:
                still_pending.append(v)

        self._pending_arrivals = still_pending
        if newly_detected:
            self._rebuild_conflict_edges()
            self._recompute_norm_stats()
        return newly_detected

    def _lock_near_vehicles(self) -> None:
        """Transition TENTATIVE ops to LOCKED once within commit_window."""
        for op in self.operations:
            if op.state != OpState.TENTATIVE:
                continue
            vehicle = self.vehicles.get(op.vehicle_id)
            if vehicle is None:
                continue
            time_to_arrival = vehicle.arrival_time - self.current_time
            if time_to_arrival <= self.commit_window + 1e-9:
                op.state = OpState.LOCKED

    def _remove_completed_vehicles(self) -> None:
        """Drop vehicles whose last operation is LOCKED and has finished.

        Before removal, logs each vehicle's waiting time (finish - arrival -
        sum(processing_times), same convention as
        intersection_scheduler.utils.metrics.episode_waiting_time) into
        self.completed_log — this is the only point where that information
        is captured, since the vehicle/its ops are gone afterward.
        """
        completed_vids = set()
        finish_by_vid: Dict[int, float] = {}
        for vid, vehicle in self.vehicles.items():
            ops = [o for o in self.operations if o.vehicle_id == vid]
            if not ops:
                continue
            last_op = max(ops, key=lambda o: o.route_position)
            if last_op.state == OpState.LOCKED and last_op.earliest_finish <= self.current_time + 1e-9:
                completed_vids.add(vid)
                finish_by_vid[vid] = last_op.earliest_finish

        if not completed_vids:
            return

        for vid in completed_vids:
            vehicle = self.vehicles[vid]
            finish = finish_by_vid[vid]
            min_finish = vehicle.arrival_time + sum(vehicle.processing_times)
            waiting_time = max(0.0, finish - min_finish)
            self.completed_log.append({
                "vehicle_id": vid,
                "arrival_time": vehicle.arrival_time,
                "finish_time": finish,
                "waiting_time": waiting_time,
            })

        self.operations = [o for o in self.operations if o.vehicle_id not in completed_vids]
        for vid in completed_vids:
            del self.vehicles[vid]
        self._rebuild_conflict_edges()
        self._recompute_norm_stats()

    # ------------------------------------------------------------------
    # Conflict edges / norm stats — rebuilt whenever the active op set
    # changes (new detection or vehicle removal), unlike the offline env
    # where these are computed once at reset.
    # ------------------------------------------------------------------

    def _rebuild_conflict_edges(self) -> None:
        self.conflict_edges = []
        n = len(self.operations)
        for i in range(n):
            for j in range(i + 1, n):
                oi, oj = self.operations[i], self.operations[j]
                if oi.zone_id == oj.zone_id and oi.vehicle_id != oj.vehicle_id:
                    self.conflict_edges.append((i, j))
        # Active = both ops not LOCKED (a LOCKED op's ordering is already
        # resolved and irreversible; only genuinely open pairs remain "active").
        self.active_conflict_edges = [
            (a, b) for (a, b) in self.conflict_edges
            if self.operations[a].state != OpState.LOCKED
            and self.operations[b].state != OpState.LOCKED
        ]

    def _recompute_norm_stats(self) -> None:
        all_p = [op.processing_time for op in self.operations] or [1.0]
        all_c = [op.earliest_finish for op in self.operations] or [1.0]
        all_r = [v.arrival_time for v in self.vehicles.values()] or [1.0]
        self.norm_max_p = max(all_p) or 1.0
        self.norm_max_c = max(all_c) or 1.0
        self.norm_max_r = max(all_r) or 1.0
        for z in self.zones.values():
            z.n_competing = sum(
                1 for o in self.operations
                if o.zone_id == z.id and o.state != OpState.LOCKED
            )

    # ------------------------------------------------------------------
    # Replanning
    # ------------------------------------------------------------------

    def plan_operation(self, action: int) -> Tuple["DynamicIntersectionEnv", float, bool]:
        """Tentatively (re)plan the operation at index `action`.

        Mirrors IntersectionEnv.step's timing computation, but:
          - may be called on an op that already has a TENTATIVE plan (a
            revision), in which case prev_start_time is preserved for the
            rescheduling-penalty term.
          - never marks the op as irrevocably done — it becomes TENTATIVE,
            not LOCKED (locking happens separately via time proximity).
        """
        op = self.operations[action]
        assert op.state != OpState.LOCKED, f"Operation {action} is locked, cannot replan"

        self._prev_completion_times = self._last_finish_per_vehicle()

        vehicle = self.vehicles[op.vehicle_id]
        start = max(self.current_time, vehicle.arrival_time)

        if op.route_position > 0:
            pred = self._predecessor_op(op)
            if pred is not None and pred.state != OpState.UNSCHEDULED:
                start = max(start, pred.earliest_finish)

        zone = self.zones[op.zone_id]
        start = max(start, self._zone_free_time(op))

        finish = start + op.processing_time

        # Preserve the previously tentative start time (if any) before
        # overwriting, so the caller/reward function can measure the change.
        if op.state == OpState.TENTATIVE:
            op.prev_start_time = op.start_time
        else:
            op.prev_start_time = None

        op.start_time = start
        op.earliest_finish = finish
        op.state = OpState.TENTATIVE

        self._propagate_finish_times(op)

        reward = self._compute_reward(op)

        done = self.current_time >= self.episode_duration
        return self, reward, done

    def _zone_free_time(self, op: DynamicOperation) -> float:
        """Earliest time op's zone is free, considering only LOCKED occupants
        (TENTATIVE plans in the same zone are not yet binding)."""
        zone = self.zones[op.zone_id]
        locked_finishes = [
            o.earliest_finish for o in self.operations
            if o.zone_id == op.zone_id and o.state == OpState.LOCKED and o is not op
        ]
        return max([zone.time_free] + locked_finishes) if locked_finishes else zone.time_free

    def _predecessor_op(self, op: DynamicOperation) -> Optional[DynamicOperation]:
        for o in self.operations:
            if o.vehicle_id == op.vehicle_id and o.route_position == op.route_position - 1:
                return o
        return None

    def _propagate_finish_times(self, op: DynamicOperation) -> None:
        ops_sorted = sorted(
            [o for o in self.operations if o.vehicle_id == op.vehicle_id],
            key=lambda o: o.route_position,
        )
        start_idx = next(
            (i for i, o in enumerate(ops_sorted) if o.route_position == op.route_position),
            None,
        )
        if start_idx is None:
            return
        for i in range(start_idx, len(ops_sorted) - 1):
            cur = ops_sorted[i]
            nxt = ops_sorted[i + 1]
            if nxt.state == OpState.LOCKED:
                continue
            zone_avail = self._zone_free_time(nxt)
            new_finish = max(cur.earliest_finish, zone_avail) + nxt.processing_time
            if abs(new_finish - nxt.earliest_finish) < 1e-9:
                break
            nxt.earliest_finish = new_finish

    def _last_finish_per_vehicle(self) -> Dict[int, float]:
        result = {}
        for vid in self.vehicles:
            ops = [o for o in self.operations if o.vehicle_id == vid]
            if not ops:
                continue
            last = max(ops, key=lambda o: o.route_position)
            result[vid] = last.earliest_finish
        return result

    def inflight_waiting_times(self) -> List[float]:
        """Waiting time of vehicles still active (detected, not yet completed)
        at the current moment, using their current planned finish.

        Called at episode end so the eval metric can include vehicles that
        did not finish before the cutoff — otherwise the most-delayed
        vehicles (the ones still stuck in flight) are silently excluded,
        biasing the reported waiting time downward (survivorship bias).
        """
        times = []
        for vid, vehicle in self.vehicles.items():
            ops = [o for o in self.operations if o.vehicle_id == vid]
            if not ops:
                continue
            last = max(ops, key=lambda o: o.route_position)
            # Only vehicles that have at least a tentative plan have a
            # meaningful finish estimate; unplanned ops keep their init value.
            min_finish = vehicle.arrival_time + sum(vehicle.processing_times)
            times.append(max(0.0, last.earliest_finish - min_finish))
        return times

    # ------------------------------------------------------------------
    # Reward: completion-time term (normalised by currently-active vehicle
    # count) + rescheduling-penalty term (proximity-weighted timing change).
    # ------------------------------------------------------------------

    def _compute_reward(self, changed_op: DynamicOperation) -> float:
        current = self._last_finish_per_vehicle()
        completion_delta = 0.0
        for vid, c in current.items():
            p = self._prev_completion_times.get(vid, c)
            completion_delta += (c - p)
        n_active = max(len(self.vehicles), 1)
        completion_term = -completion_delta / n_active

        penalty_term = 0.0
        if changed_op.prev_start_time is not None:
            vehicle = self.vehicles.get(changed_op.vehicle_id)
            if vehicle is not None:
                time_to_arrival = max(vehicle.arrival_time - self.current_time, 1e-6)
                # Capped so a near-arrival vehicle can't produce an explosive
                # penalty that dominates the reward (see __init__).
                proximity_weight = min(1.0 / time_to_arrival, self.max_proximity_weight)
                timing_change = abs(changed_op.start_time - changed_op.prev_start_time)
                penalty_term = -self.penalty_coef * proximity_weight * timing_change

        return completion_term + penalty_term

    # ------------------------------------------------------------------
    # Time advancement
    # ------------------------------------------------------------------

    def advance_time(self) -> Tuple[bool, List[int]]:
        """Advance current_time to the next event (arrival detection, lock
        transition, or episode end); lock near vehicles and remove completed
        ones along the way.

        Returns (episode_done, newly_detected_vehicle_ids).
        """
        self._lock_near_vehicles()
        self._remove_completed_vehicles()

        if self.current_time >= self.episode_duration - 1e-9:
            return True, []

        next_arrival_detect_time = (
            self._pending_arrivals[0].arrival_time - self.detection_window
            if self._pending_arrivals else None
        )
        candidates = [self.episode_duration]
        if next_arrival_detect_time is not None:
            candidates.append(max(next_arrival_detect_time, self.current_time))
        # Also consider the next lock-transition time for any TENTATIVE op.
        for op in self.operations:
            if op.state == OpState.TENTATIVE:
                vehicle = self.vehicles.get(op.vehicle_id)
                if vehicle is not None:
                    lock_time = vehicle.arrival_time - self.commit_window
                    if lock_time > self.current_time:
                        candidates.append(lock_time)

        self.current_time = min(candidates)
        newly_detected = self._detect_new_vehicles()
        self._lock_near_vehicles()
        self._remove_completed_vehicles()

        done = self.current_time >= self.episode_duration - 1e-9
        return done, newly_detected

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def openable_op_indices(self) -> List[int]:
        """Indices of operations eligible for (re)planning this pass:
        UNSCHEDULED (detected, never planned) or TENTATIVE (revisable)."""
        return [
            i for i, op in enumerate(self.operations)
            if op.state != OpState.LOCKED
        ]

    def affected_op_indices(self, trigger_vehicle_ids: List[int]) -> List[int]:
        """Indices of operations that should be (re)planned in response to the
        given newly-detected vehicles — the churn-reduction filter.

        An operation is affected if:
          - it is UNSCHEDULED (it has never been planned; it must be planned
            regardless — this includes all ops of the trigger vehicles), OR
          - it is TENTATIVE and shares a conflict zone with any operation of a
            trigger vehicle (its optimal timing could genuinely have changed).

        TENTATIVE operations in zones untouched by the new arrivals are left
        as-is: not re-visited, no rescheduling penalty, no wasted transition.
        This cuts the ~37-replans-per-vehicle churn to only genuinely-affected
        re-plans, cleaning up the reward signal and speeding up episodes.

        If trigger_vehicle_ids is empty (e.g. a lock/removal-only event with no
        new detection), returns only UNSCHEDULED ops so nothing already-planned
        is needlessly disturbed.
        """
        trigger_zones = set()
        for vid in trigger_vehicle_ids:
            for op in self.operations:
                if op.vehicle_id == vid:
                    trigger_zones.add(op.zone_id)

        affected = []
        for i, op in enumerate(self.operations):
            if op.state == OpState.LOCKED:
                continue
            if op.state == OpState.UNSCHEDULED:
                affected.append(i)
            elif op.state == OpState.TENTATIVE and op.zone_id in trigger_zones:
                affected.append(i)
        return affected
