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
    is detected OR an operation locks. All affected TENTATIVE operations may
    be revised in that pass; LOCKED operations are frozen.
  - TENTATIVE plans are ZONE-BINDING among themselves: each zone keeps an
    explicit priority queue of TENTATIVE operations, in the order the policy
    planned them. Planning an operation appends it to its zone's queue;
    REPLANNING an operation moves it to the back of the queue (that is the
    revision semantics — you give up your slot). LOCKED ops leave the queue
    and become static occupancy windows that tentative ops gap-fill around.
    All operation times are derived from the queues + locked windows + route
    chains by a global recompute. This is what gives the policy control over
    the schedule: the order it plans operations IS the priority order through
    every conflict zone, exactly like the offline model — without it, timing
    would collapse to arrival-order FIFO and the policy would have no effect
    (verified: identical schedules across different random policies).
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


@dataclass(eq=False)  # identity semantics: ops live in zone queues (list.remove)
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

        # Realised delay of vehicles already removed by
        # _remove_completed_vehicles, and the last total returned by
        # _total_excess_delay. Both persist across removals so the reward
        # ledger covers every vehicle ever seen (see _compute_reward).
        self._completed_delay: float = 0.0
        self._prev_total_delay: float = 0.0

        # Zones that gained a LOCKED op during the most recent advance_time
        # call. Read by affected_op_indices so the next replan pass re-times
        # TENTATIVE ops in those zones against the now-binding occupancy.
        self._newly_locked_zones: set = set()

        # Per-zone priority queue of TENTATIVE operations, in policy-planning
        # order. The schedule is derived from these queues: each op starts no
        # earlier than its queue predecessor's finish. LOCKED ops are NOT in
        # the queues — locking converts an op into a static occupancy window
        # that tentative ops gap-fill around during the time recompute. (An
        # earlier design kept locked ops in the queues; that forces priority
        # requeues whenever a tentative op's chain shifts past a locked
        # successor, and those unchecked requeues created precedence cycles.)
        self._zone_queue: Dict[int, List[DynamicOperation]] = {}

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
        self._completed_delay = 0.0
        self._prev_total_delay = 0.0
        self._newly_locked_zones = set()
        self._zone_queue = {}
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
        """Transition TENTATIVE ops to LOCKED once within commit_window.

        A locking op leaves its zone's priority queue and becomes a static
        occupancy window (its times were consistent with the queue at the
        last recompute and are frozen from here on). Newly locked zones are
        recorded so the next replan pass lets the policy re-optimize
        (requeue) TENTATIVE plans around the now irrevocable occupancy."""
        for op in self.operations:
            if op.state != OpState.TENTATIVE:
                continue
            vehicle = self.vehicles.get(op.vehicle_id)
            if vehicle is None:
                continue
            if vehicle.arrival_time - self.current_time <= self.commit_window + 1e-9:
                op.state = OpState.LOCKED
                self._newly_locked_zones.add(op.zone_id)
                queue = self._zone_queue.get(op.zone_id)
                if queue is not None and op in queue:
                    queue.remove(op)

    def _remove_completed_vehicles(self) -> None:
        """Drop vehicles whose last operation is LOCKED and has finished.

        Before removal, logs each vehicle's waiting time (finish - arrival -
        sum(processing_times), same convention as
        intersection_scheduler.utils.metrics.episode_waiting_time) into
        self.completed_log — this is the only point where that information
        is captured, since the vehicle/its ops are gone afterward.
        """
        ops_by_vid = self._ops_by_vehicle()
        completed_vids = set()
        finish_by_vid: Dict[int, float] = {}
        for vid, vehicle in self.vehicles.items():
            ops = ops_by_vid.get(vid)
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
            # Carry the realised delay into the reward ledger before the
            # vehicle is deleted below, so _total_excess_delay stays a running
            # total over ALL vehicles rather than only the currently-active
            # ones (see _compute_reward).
            self._completed_delay += waiting_time

        self.operations = [o for o in self.operations if o.vehicle_id not in completed_vids]
        for zid, queue in self._zone_queue.items():
            self._zone_queue[zid] = [o for o in queue if o.vehicle_id not in completed_vids]
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
        # Group op indices by zone: only same-zone pairs can conflict, so this
        # is O(n + sum_z k_z^2) instead of the naive O(n^2) all-pairs scan.
        # Sorted to keep the exact lexicographic (i, j) order the naive scan
        # produced (edge tensors downstream stay byte-identical).
        by_zone: Dict[int, List[int]] = {}
        for i, op in enumerate(self.operations):
            by_zone.setdefault(op.zone_id, []).append(i)
        edges = []
        for indices in by_zone.values():
            for a_pos in range(len(indices)):
                i = indices[a_pos]
                oi = self.operations[i]
                for b_pos in range(a_pos + 1, len(indices)):
                    j = indices[b_pos]
                    if oi.vehicle_id != self.operations[j].vehicle_id:
                        edges.append((i, j))
        edges.sort()
        self.conflict_edges = edges
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

        Planning = claiming the next slot in the op's zone queue. A first
        plan appends the op to the queue; a REPLAN moves it to the back
        (revision gives up the held slot — that is the policy's requeue
        lever, and prev_start_time is preserved so the rescheduling penalty
        measures the change). All times are then re-derived from the queues
        and route chains by _recompute_times — planning ORDER is the schedule.
        """
        op = self.operations[action]
        assert op.state != OpState.LOCKED, f"Operation {action} is locked, cannot replan"

        self._prev_completion_times = self._last_finish_per_vehicle()
        # Re-baseline here, not at the end of the previous plan_operation:
        # advance_time detects vehicles and locks ops between actions, which
        # moves total delay for reasons THIS action is not responsible for.
        # Snapshotting immediately before the change credits each action with
        # exactly its own effect.
        self._prev_total_delay = self._total_excess_delay()

        queue = self._zone_queue.setdefault(op.zone_id, [])
        if op.state == OpState.TENTATIVE:
            # Preserve the previous start so the caller/reward function can
            # measure the timing change caused by this revision.
            op.prev_start_time = op.start_time
            queue.remove(op)
        else:
            op.prev_start_time = None
        queue.append(op)
        op.state = OpState.TENTATIVE

        self._recompute_times()

        reward = self._compute_reward(op)

        done = self.current_time >= self.episode_duration
        return self, reward, done

    def _recompute_times(self) -> None:
        """Derive all TENTATIVE ops' times from zone queues + route chains.

        Single Kahn pass over the tentative precedence graph (route edges
        between tentative ops + consecutive zone-queue edges). LOCKED ops
        contribute as constants: a locked route predecessor bounds its
        successor's earliest start, and locked ops' [start, finish) windows
        are static blockers that tentative ops gap-fill around (an op may
        run BEFORE a locked window if it fits — locking freezes one op's
        slot, not the whole zone). Raises on a precedence cycle: the
        feasibility deadlock check must prevent those from being planned."""
        tentative = [o for o in self.operations if o.state == OpState.TENTATIVE]
        if not tentative:
            return

        pos_index = {(o.vehicle_id, o.route_position): o for o in self.operations}
        locked_windows: Dict[int, List[Tuple[float, float]]] = {}
        for o in self.operations:
            if o.state == OpState.LOCKED:
                locked_windows.setdefault(o.zone_id, []).append(
                    (o.start_time, o.earliest_finish)
                )
        for windows in locked_windows.values():
            windows.sort()

        succs: Dict[int, List[DynamicOperation]] = {}
        n_preds: Dict[int, int] = {id(o): 0 for o in tentative}
        earliest: Dict[int, float] = {}

        for o in tentative:
            base = self.current_time
            if o.route_position == 0:
                vehicle = self.vehicles.get(o.vehicle_id)
                if vehicle is not None:
                    base = max(base, vehicle.arrival_time)
            else:
                pred = pos_index.get((o.vehicle_id, o.route_position - 1))
                if pred is not None and pred.state == OpState.LOCKED:
                    base = max(base, pred.earliest_finish)
            earliest[id(o)] = base
            route_succ = pos_index.get((o.vehicle_id, o.route_position + 1))
            if route_succ is not None and route_succ.state == OpState.TENTATIVE:
                succs.setdefault(id(o), []).append(route_succ)
                n_preds[id(route_succ)] += 1
        for queue in self._zone_queue.values():
            for a, b in zip(queue, queue[1:]):
                succs.setdefault(id(a), []).append(b)
                n_preds[id(b)] += 1

        ready = [o for o in tentative if n_preds[id(o)] == 0]
        processed = 0
        while ready:
            o = ready.pop()
            processed += 1
            start = earliest[id(o)]
            for w_start, w_end in locked_windows.get(o.zone_id, ()):
                if start + o.processing_time <= w_start + 1e-9:
                    break  # fits entirely before this (sorted) window
                if start < w_end - 1e-9:
                    start = w_end  # overlaps: bump past the window
            o.start_time = start
            o.earliest_finish = start + o.processing_time
            for nxt in succs.get(id(o), []):
                if earliest[id(nxt)] < o.earliest_finish:
                    earliest[id(nxt)] = o.earliest_finish
                n_preds[id(nxt)] -= 1
                if n_preds[id(nxt)] == 0:
                    ready.append(nxt)

        if processed != len(tentative):
            raise RuntimeError(
                "precedence cycle among tentative operations — the "
                "feasibility deadlock check failed to prevent a cyclic "
                "zone priority"
            )

    def _ops_by_vehicle(self) -> Dict[int, List[DynamicOperation]]:
        """Single-pass grouping of operations by vehicle_id — O(n) instead of
        an O(n) scan per vehicle for the helpers below."""
        grouped: Dict[int, List[DynamicOperation]] = {}
        for o in self.operations:
            grouped.setdefault(o.vehicle_id, []).append(o)
        return grouped

    def _last_finish_per_vehicle(self) -> Dict[int, float]:
        result = {}
        ops_by_vid = self._ops_by_vehicle()
        for vid in self.vehicles:
            ops = ops_by_vid.get(vid)
            if not ops:
                continue
            last = max(ops, key=lambda o: o.route_position)
            result[vid] = last.earliest_finish
        return result

    def _total_excess_delay(self) -> float:
        """Total excess-over-free-flow delay across every vehicle seen so far.

        Active vehicles are measured from their current planned finish;
        completed ones contribute their realised delay via _completed_delay
        (they are gone from self.vehicles by then). Same per-vehicle quantity
        as inflight_waiting_times()/completed_log, summed rather than averaged
        — see _compute_reward for why the sum, not the mean, is the right
        per-step signal.
        """
        total = self._completed_delay
        ops_by_vid = self._ops_by_vehicle()
        for vid, vehicle in self.vehicles.items():
            ops = ops_by_vid.get(vid)
            if not ops:
                continue
            last = max(ops, key=lambda o: o.route_position)
            min_finish = vehicle.arrival_time + sum(vehicle.processing_times)
            total += max(0.0, last.earliest_finish - min_finish)
        return total

    def inflight_waiting_times(self) -> List[float]:
        """Waiting time of vehicles still active (detected, not yet completed)
        at the current moment, using their current planned finish.

        Called at episode end so the eval metric can include vehicles that
        did not finish before the cutoff — otherwise the most-delayed
        vehicles (the ones still stuck in flight) are silently excluded,
        biasing the reported waiting time downward (survivorship bias).
        """
        times = []
        ops_by_vid = self._ops_by_vehicle()
        for vid, vehicle in self.vehicles.items():
            ops = ops_by_vid.get(vid)
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
        # Delay term: negative change in TOTAL excess delay over every vehicle
        # seen so far, completed ones included. Three properties this must have
        # to be a faithful proxy for the reported metric
        # (utils.metrics.episode_waiting_time_all):
        #
        #   1. Excess over free-flow, not raw finish time. Baseline is
        #      arrival_time + sum(processing_times), matching
        #      inflight_waiting_times()/_remove_completed_vehicles(). Raw
        #      earliest_finish charges a vehicle for its own unavoidable
        #      travel time, so long routes look "bad" however well scheduled.
        #   2. Completed vehicles keep contributing (via _completed_delay).
        #      _last_finish_per_vehicle() iterates self.vehicles, and
        #      _remove_completed_vehicles() deletes finished ones — so delay
        #      already caused used to silently leave the ledger, and shedding
        #      a delayed vehicle registered as a reward INCREASE.
        #   3. No moving denominator. Dividing each step's delta by the
        #      current vehicle count does not telescope when that count
        #      changes (it does, constantly: the population turns over
        #      ~2x per episode), so the episode's summed reward was not any
        #      fixed transform of total delay. Undivided, the per-step deltas
        #      telescope to -(total excess delay) over the episode, i.e. the
        #      reported mean-per-vehicle metric times a per-episode constant.
        #      Advantage normalisation in ppo_update absorbs that constant.
        total_delay = self._total_excess_delay()
        completion_term = -(total_delay - self._prev_total_delay)
        self._prev_total_delay = total_delay

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
        # Locks recorded here feed the NEXT replan pass (via
        # affected_op_indices), so clear the previous pass's set first.
        self._newly_locked_zones = set()
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
            trigger vehicle (its optimal timing could genuinely have changed), OR
          - it is TENTATIVE and its zone gained a LOCKED op in the advance_time
            call that led to this pass: the locked occupancy is now binding, so
            an overlapping tentative plan must be re-timed before it can lock
            overlapping it (zone-exclusivity would otherwise be violated).

        TENTATIVE operations in zones untouched by new arrivals or new locks
        are left as-is: not re-visited, no rescheduling penalty, no wasted
        transition. This cuts the ~37-replans-per-vehicle churn to only
        genuinely-affected re-plans, cleaning up the reward signal and
        speeding up episodes.
        """
        trigger_zones = set(self._newly_locked_zones)
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
