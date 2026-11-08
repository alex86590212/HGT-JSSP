"""Gap-free full-route admission environment.

A different admission policy from DynamicIntersectionEnv's blocking model: a
vehicle is not granted its first zone until its ENTIRE route can be
reserved, back-to-back with zero internal gaps, against every currently
committed reservation. If that is not possible right now, the vehicle waits
OUTSIDE the intersection (stays unadmitted, occupies nothing) and is
retried later.

Consequence: in-route blocking cannot occur here. A vehicle's waiting time
equals exactly the time it spent waiting OUTSIDE before admission; once it
starts moving, s_{i,k+1} == f_i,k for every k, so there is never any delay
after departure. This is a genuinely different constraint from
DynamicIntersectionEnv's blocking model (hold your CURRENT zone while
waiting for the next); this model instead never lets you occupy anything
until your whole path is already clear.

Admissions are TENTATIVE/LOCKED, same two-state idea as
DynamicIntersectionEnv, applied at whole-route granularity:
  - TENTATIVE: admitted, holds its reserved route, but still revisable.
  - LOCKED: irrevocable, once its committed departure is within
    commit_window of the current time -- permanent ground truth for
    everyone else from then on.
The default runner revises whenever a new vehicle is detected: every still-
TENTATIVE admission is revoked (its reservation wiped) and the combined
pool of pending + just-revoked vehicles is re-admitted from scratch, in
priority order, against only LOCKED occupancy. This mirrors exactly how
DynamicIntersectionEnv's TENTATIVE zone-queue revision already works (a
requeued op gives up its slot and is fully re-derived) -- no bespoke
pairwise-conflict detection needed, since it's the same admission_check
logic just re-run over a larger pool. Net effect: a vehicle admitted early
can get bumped later (pushed back to wait outside longer) if a higher-
priority vehicle is detected before it locks. The commitment-exchange runner
instead keeps the baseline and applies only feasible revisions that reduce
the complete detected-set planned departure sum, before each lock gate.

Same-lane FIFO is a HARD constraint this revision process must never
override: a vehicle cannot be admitted ahead of an earlier-arriving
same-lane vehicle regardless of priority tag (that's not a preference, real
vehicles can't overtake in their own lane) -- enforced by
_same_lane_leaders/_effective_start exactly as before, and additionally: a
vehicle may only LOCK once every same-lane leader is ALSO already LOCKED
(not merely tentative) -- otherwise a leader could still be revoked and
pushed back after a follower already became irrevocable, which would be an
inconsistent state no real vehicle could actually be in.
"""

from __future__ import annotations

import heapq
from copy import copy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from dynamic_scheduler.environment.dynamic_intersection import DynamicVehicle
from intersection_scheduler.data.scenario_generator_4x4 import _MANOEUVRE_TO_LANE

Selector = Callable[["AdmissionEnv", List[DynamicVehicle]], DynamicVehicle]


@dataclass
class _ZoneOccupancy:
    # (start, end, vehicle_id) intervals. LOCKED reservations are
    # permanent; TENTATIVE ones can be removed via remove_vehicle when
    # revoked during a revision pass.
    intervals: List[Tuple[float, float, int]] = field(default_factory=list)

    def blocking_interval(self, start: float, end: float) -> Optional[Tuple[float, float]]:
        for s, e, _ in self.intervals:
            if start < e - 1e-9 and s < end - 1e-9:
                return (s, e)
        return None

    def blocking_vehicle_ids(self, start: float, end: float) -> set:
        return {vid for s, e, vid in self.intervals if start < e - 1e-9 and s < end - 1e-9}

    def add(self, start: float, end: float, vehicle_id: int) -> None:
        self.intervals.append((start, end, vehicle_id))

    def remove_vehicle(self, vehicle_id: int) -> None:
        self.intervals = [iv for iv in self.intervals if iv[2] != vehicle_id]


class AdmissionEnv:
    """Tracks committed zone occupancy and admits vehicles whose entire
    route is currently reservable gap-free, starting as soon as possible.
    Admissions are TENTATIVE until commit_window before their departure,
    then LOCKED (irrevocable) -- see module docstring."""

    def __init__(self, detection_window: float = 10.0, commit_window: float = 2.5) -> None:
        self.detection_window = detection_window
        self.commit_window = commit_window
        self.zones: Dict[int, _ZoneOccupancy] = {}
        self.current_time: float = 0.0
        self.episode_duration: float = 0.0
        self.completed_log: List[dict] = []
        self.locked_ids: set = set()
        self.tentative_ids: set = set()
        self.committed_s1: Dict[int, float] = {}
        self._vehicles_by_id: Dict[int, DynamicVehicle] = {}
        self._lane_by_vehicle: Dict[int, Optional[str]] = {}

    @property
    def admitted_ids(self) -> set:
        return self.locked_ids | self.tentative_ids

    def reset(self, arrivals: List[DynamicVehicle], episode_duration: float) -> None:
        self.zones = {}
        self.current_time = 0.0
        self.episode_duration = episode_duration
        self.completed_log = []
        self.locked_ids = set()
        self.tentative_ids = set()
        self.committed_s1 = {}
        self._vehicles_by_id = {v.id: v for v in arrivals}
        self._lane_by_vehicle = {v.id: _MANOEUVRE_TO_LANE.get(v.manoeuvre) for v in arrivals}

    def _route_plan(self, vehicle: DynamicVehicle, s1: float) -> List[Tuple[int, float, float]]:
        plan = []
        t = s1
        for zone_id, p in zip(vehicle.route, vehicle.processing_times):
            plan.append((zone_id, t, t + p))
            t += p
        return plan

    def earliest_start(self, vehicle: DynamicVehicle) -> float:
        return max(self.current_time, vehicle.arrival_time)

    def _same_lane_leaders(self, vehicle: DynamicVehicle) -> List[DynamicVehicle]:
        """Same-lane vehicles that arrived strictly before this one (FIFO
        order, tie-broken by id) -- mirrors
        DynamicIntersectionEnv._same_lane_leaders_planned /
        gapfree_admission_solver's Type-2 constraint, so the greedy env
        enforces the identical no-overtaking rule the CP-SAT reference
        does."""
        lane = self._lane_by_vehicle.get(vehicle.id)
        if lane is None:
            return []
        leaders = []
        for other_id, other_lane in self._lane_by_vehicle.items():
            if other_lane != lane or other_id == vehicle.id:
                continue
            other = self._vehicles_by_id[other_id]
            if (other.arrival_time < vehicle.arrival_time - 1e-9
                    or (abs(other.arrival_time - vehicle.arrival_time) < 1e-9 and other_id < vehicle.id)):
                leaders.append(other)
        return leaders

    def _effective_start(self, vehicle: DynamicVehicle) -> Tuple[Optional[float], bool]:
        """Returns (s1, leaders_all_admitted). s1 is None if some same-lane
        leader hasn't been admitted yet at all -- vehicle cannot be
        admitted before it, regardless of zone occupancy."""
        leaders = self._same_lane_leaders(vehicle)
        same_lane_floor = 0.0
        for leader in leaders:
            if leader.id not in self.admitted_ids:
                return None, False
            same_lane_floor = max(
                same_lane_floor,
                self.committed_s1[leader.id] + leader.processing_times[0],
            )
        s1 = max(self.earliest_start(vehicle), same_lane_floor)
        return s1, True

    def admission_check(self, vehicle: DynamicVehicle) -> Tuple[bool, Optional[float]]:
        """Return (feasible, retry_time) for departing at the earliest
        start consistent with same-lane FIFO and zone occupancy.
        retry_time is the earliest moment anything could plausibly have
        changed for this candidate: the end of whichever committed zone
        interval blocks it soonest. If blocked purely by an unadmitted
        same-lane leader, retry_time is None -- that leader's own eventual
        admission event re-triggers a fresh check for every pending vehicle
        in the same run_admission_episode pass, so no separate retry point
        is needed here."""
        s1, leaders_ready = self._effective_start(vehicle)
        if not leaders_ready:
            return False, None
        earliest_retry = None
        for zone_id, s, e in self._route_plan(vehicle, s1):
            occ = self.zones.setdefault(zone_id, _ZoneOccupancy())
            block = occ.blocking_interval(s, e)
            if block is not None:
                _, block_end = block
                if earliest_retry is None or block_end < earliest_retry:
                    earliest_retry = block_end
        if earliest_retry is None:
            return True, None
        return False, earliest_retry

    def blockers_for(self, vehicle: DynamicVehicle) -> Tuple[Optional[float], set]:
        """Returns (s1, blocking_vehicle_ids) at the vehicle's effective
        start -- s1 is None if a same-lane leader isn't admitted yet
        (blocking_vehicle_ids is then empty; that case has no analytic
        retry point, same as admission_check). Used by the priority-
        inheritance admission variant to identify WHICH specific vehicles
        are blocking a candidate, not just whether one is."""
        s1, leaders_ready = self._effective_start(vehicle)
        if not leaders_ready:
            return None, set()
        blockers: set = set()
        for zone_id, s, e in self._route_plan(vehicle, s1):
            occ = self.zones.setdefault(zone_id, _ZoneOccupancy())
            blockers |= occ.blocking_vehicle_ids(s, e)
        return s1, blockers

    def admit(self, vehicle: DynamicVehicle) -> None:
        """Admit into TENTATIVE state -- reserves zone occupancy and
        records committed_s1, but does NOT log a final outcome yet, since a
        tentative admission can still be revoked before it locks."""
        s1, leaders_ready = self._effective_start(vehicle)
        assert leaders_ready and s1 is not None, "admit() called on an infeasible vehicle"
        plan = self._route_plan(vehicle, s1)
        for zone_id, s, e in plan:
            self.zones.setdefault(zone_id, _ZoneOccupancy()).add(s, e, vehicle.id)
        self.tentative_ids.add(vehicle.id)
        self.committed_s1[vehicle.id] = s1

    def revoke_tentative(self, vehicle_id: int) -> None:
        """Undo a TENTATIVE admission entirely: wipe its zone reservations
        and its committed departure, so it re-enters the pending pool as if
        never admitted. Never call on a LOCKED vehicle."""
        assert vehicle_id in self.tentative_ids, "revoke_tentative() called on a non-tentative vehicle"
        vehicle = self._vehicles_by_id[vehicle_id]
        for zone_id in set(vehicle.route):
            occ = self.zones.get(zone_id)
            if occ is not None:
                occ.remove_vehicle(vehicle_id)
        self.tentative_ids.discard(vehicle_id)
        del self.committed_s1[vehicle_id]

    def lock_near(self) -> None:
        """Promote each TENTATIVE vehicle within commit_window of its
        committed departure to LOCKED (irrevocable) and log its final
        outcome. A vehicle may only lock once every same-lane leader is
        ALSO already locked -- otherwise a leader could still be revoked
        and pushed back after this vehicle became irrevocable, which no
        real vehicle could actually be in. Vehicles blocked only by that
        leader-lock gate stay tentative and are simply re-checked next time
        lock_near runs (called every processed event)."""
        for vid in sorted(self.tentative_ids, key=lambda v: self.committed_s1[v]):
            s1 = self.committed_s1[vid]
            if s1 - self.current_time > self.commit_window + 1e-9:
                continue
            vehicle = self._vehicles_by_id[vid]
            leaders = self._same_lane_leaders(vehicle)
            if any(leader.id not in self.locked_ids for leader in leaders):
                continue
            self.tentative_ids.discard(vid)
            self.locked_ids.add(vid)
            plan = self._route_plan(vehicle, s1)
            finish = plan[-1][2]
            min_finish = vehicle.arrival_time + sum(vehicle.processing_times)
            waiting_time = max(0.0, finish - min_finish)
            self.completed_log.append({
                "vehicle_id": vid,
                "arrival_time": vehicle.arrival_time,
                "finish_time": finish,
                "waiting_time": waiting_time,
            })

    def finalize_tentative(self, vehicle_id: int) -> None:
        """End-of-episode only: log a still-TENTATIVE vehicle's current
        committed outcome as final, since there is no more revision
        opportunity once the episode ends."""
        vehicle = self._vehicles_by_id[vehicle_id]
        s1 = self.committed_s1[vehicle_id]
        plan = self._route_plan(vehicle, s1)
        finish = plan[-1][2]
        min_finish = vehicle.arrival_time + sum(vehicle.processing_times)
        waiting_time = max(0.0, finish - min_finish)
        self.completed_log.append({
            "vehicle_id": vehicle_id,
            "arrival_time": vehicle.arrival_time,
            "finish_time": finish,
            "waiting_time": waiting_time,
        })


def run_admission_episode(
    arrivals: List[DynamicVehicle],
    episode_duration: float,
    detection_window: float,
    selector: Selector,
    commit_window: float = 2.5,
) -> Tuple[float, float, int]:
    """One episode under gap-free admission with TENTATIVE/LOCKED
    revisability (see module docstring). Returns (mean_waiting_time,
    completion_rate, n_vehicles_seen), same convention as
    eval_dynamic.run_episode_with.

    Revision policy: WIPE-AND-RERANK -- whenever a new vehicle is
    detected, every still-TENTATIVE admission is revoked and the combined
    pending pool is re-admitted from scratch in priority order. See
    run_admission_episode_priority_inherit for the more surgical
    inheritance-aware eviction alternative.

    Never-admitted-by-cutoff vehicles contribute (episode_duration -
    arrival_time) as a floor on their eventual waiting time; still-
    TENTATIVE-at-cutoff vehicles are finalized at their current commitment
    -- consistent with never overstating completion (same spirit as
    DynamicIntersectionEnv.inflight_waiting_times)."""
    env = AdmissionEnv(detection_window=detection_window, commit_window=commit_window)
    env.reset(arrivals, episode_duration)

    by_id = {v.id: v for v in arrivals}
    arrivals_sorted = sorted(arrivals, key=lambda v: v.arrival_time)
    pending_ids: set = set()

    events: List[float] = []
    for v in arrivals_sorted:
        heapq.heappush(events, max(0.0, v.arrival_time - detection_window))
    heapq.heappush(events, episode_duration)

    processed: set = set()
    while events:
        t = heapq.heappop(events)
        if t in processed or t > episode_duration + 1e-9:
            continue
        processed.add(t)
        env.current_time = t

        # Lock BEFORE detecting/wiping: a vehicle already past its
        # commit_window threshold must not be swept up in the wipe below.
        env.lock_near()

        newly_detected = False
        for v in arrivals_sorted:
            if v.id in env.admitted_ids or v.id in pending_ids:
                continue
            if v.arrival_time - t <= detection_window + 1e-9:
                pending_ids.add(v.id)
                newly_detected = True

        if newly_detected:
            for vid in list(env.tentative_ids):
                env.revoke_tentative(vid)
                pending_ids.add(vid)

        while True:
            candidates = []
            for vid in list(pending_ids):
                v = by_id[vid]
                feasible, retry = env.admission_check(v)
                if feasible:
                    candidates.append(v)
                elif retry is not None and retry <= episode_duration + 1e-9:
                    heapq.heappush(events, retry)
            if not candidates:
                break
            chosen = selector(env, candidates)
            env.admit(chosen)
            pending_ids.discard(chosen.id)

        # Catch anything freshly (re-)admitted whose commitment already
        # qualifies for locking at this same instant.
        env.lock_near()

    for vid in list(env.tentative_ids):
        env.finalize_tentative(vid)

    n_seen = len(arrivals)
    for v in arrivals_sorted:
        if v.id not in env.admitted_ids:
            floor_wait = max(0.0, episode_duration - v.arrival_time)
            env.completed_log.append({
                "vehicle_id": v.id,
                "arrival_time": v.arrival_time,
                "finish_time": None,
                "waiting_time": floor_wait,
            })

    n_logged = len(env.completed_log)
    wt = sum(c["waiting_time"] for c in env.completed_log) / n_logged if n_logged else 0.0
    comp = len(env.admitted_ids) / n_seen if n_seen else 1.0
    return wt, comp, n_seen


# ---------------------------------------------------------------------
# Selectors: rank which currently-admissible vehicle gets admitted first
# when multiple are simultaneously eligible (admitting one can consume
# zone-time another needed, so order matters).
# ---------------------------------------------------------------------

def edf_admission_select(env: AdmissionEnv, candidates: List[DynamicVehicle]) -> DynamicVehicle:
    """iEDF's tag d_i^k = a_i + sum_{j<k} p_i^j evaluated at k=0 (the only
    operation ever independently rankable under whole-route admission,
    since offset_0 = 0): this reduces exactly to ranking by arrival_time.
    Kept as its own function (rather than inlined) to document that this IS
    the iEDF tag, just collapsed by k=0, not a redefinition."""
    return min(candidates, key=lambda v: (v.arrival_time, v.id))


def _edf_completion_tag(vehicle: DynamicVehicle) -> Tuple[float, int]:
    """iEDF's tag evaluated at k=n (one past the vehicle's last zone) instead
    of k=0: d_i = a_i + sum_j p_i^j, the vehicle's free-flow COMPLETION time
    rather than its free-flow start time. Still the same tag formula, still
    genuinely EDF -- k=0 degenerates to pure arrival_time here (nothing to
    sum before the first zone), but k=n does not, since whole-route
    admission makes exactly one decision per vehicle covering its ENTIRE
    route, and k=n is the free-flow instant that decision actually
    determines completion for. Rewards short routes similarly to SPT, but
    stays anchored to arrival_time as an additive term rather than
    discarding it outright -- a different ranking from pure SPT whenever
    two vehicles' (arrival_time + duration) sums cross differently than
    their durations alone would."""
    return (vehicle.arrival_time + sum(vehicle.processing_times), vehicle.id)


def edf_completion_admission_select(env: AdmissionEnv, candidates: List[DynamicVehicle]) -> DynamicVehicle:
    """Ranks by _edf_completion_tag -- see its docstring."""
    return min(candidates, key=_edf_completion_tag)


def igreedy_admission_select(env: AdmissionEnv, candidates: List[DynamicVehicle]) -> DynamicVehicle:
    """Earliest achievable start = the vehicle's EFFECTIVE earliest start
    (max of current_time, arrival_time, and any binding same-lane-leader
    floor) -- NOT just max(current_time, arrival_time). With same-lane
    FIFO enforced, this can diverge from arrival_time alone (a vehicle
    whose lane-leader committed to a late departure can have a later
    effective start than a same-arrival-time vehicle in a clear lane), so
    iGreedy and EDF are no longer guaranteed identical the way they are
    with same-lane FIFO ignored -- verify empirically, don't assume."""
    return min(candidates, key=lambda v: (env._effective_start(v)[0], v.arrival_time, v.id))


def lifo_admission_select(env: AdmissionEnv, candidates: List[DynamicVehicle]) -> DynamicVehicle:
    return max(candidates, key=lambda v: (v.arrival_time, v.id))


def backpressure_admission_select(env: AdmissionEnv, candidates: List[DynamicVehicle]) -> DynamicVehicle:
    """No literal 'zone backlog' exists in this model (nothing queues --
    a vehicle is either admitted or waiting outside with no reservation at
    all). Adapted analog: prefer admitting whichever candidate's route
    passes through the most zones ALSO wanted by other currently-pending
    candidates, i.e. clear the vehicle sitting at the most contested
    intersection of demand first. Ties broken by arrival_time."""
    candidate_ids = {v.id for v in candidates}
    zone_wanters: Dict[int, int] = {}
    for v in candidates:
        for z in set(v.route):
            zone_wanters[z] = zone_wanters.get(z, 0) + 1

    def contention(v: DynamicVehicle) -> int:
        return sum(zone_wanters.get(z, 0) - 1 for z in set(v.route))

    return max(candidates, key=lambda v: (contention(v), -v.arrival_time, -v.id))


# ---------------------------------------------------------------------
# Priority-inheritance admission: a more surgical alternative to
# run_admission_episode's blanket wipe-and-rerank, extending the
# priority-inheritance mechanism validated for the blocking model
# (edf_priority_inheritance_select in evaluation/eval_dynamic.py) to
# admission eviction instead of zone occupancy.
# ---------------------------------------------------------------------

def _edf_base_tag(vehicle: DynamicVehicle) -> Tuple[float, int]:
    """iEDF tag at k=0 (arrival_time, id) -- the base priority this
    mechanism is built around, same as edf_admission_select."""
    return (vehicle.arrival_time, vehicle.id)


def run_admission_episode_priority_inherit(
    arrivals: List[DynamicVehicle],
    episode_duration: float,
    detection_window: float,
    commit_window: float = 2.5,
    tag_fn: Callable[[DynamicVehicle], Tuple[float, int]] = _edf_base_tag,
) -> Tuple[float, float, int]:
    """One episode under gap-free admission with TENTATIVE/LOCKED
    revisability, using SELECTIVE inheritance-aware eviction instead of
    run_admission_episode's blanket wipe-and-rerank on every detection.

    tag_fn is the underlying priority tag inheritance is built on top of --
    defaults to _edf_base_tag (arrival_time), but _edf_completion_tag
    (arrival_time + total route duration, the free-flow COMPLETION time)
    is a genuine EDF-family alternative -- see that function's docstring.

    Motivation: if pending vehicle X (urgent, small tag) is blocked by
    TENTATIVE vehicle Y (large tag), don't just compare X's tag to Y's own
    tag -- compare X's tag to the tag Y would effectively have if credited
    with whatever ELSE it's currently blocking, so an eviction serves the
    single most deserving blocked vehicle rather than firing for any
    marginally-more-urgent newcomer.

    Concretely: let blocks(Y) be the set of PENDING vehicles currently
    blocked by TENTATIVE vehicle Y (via blockers_for). For pending X
    blocked by Y, define Y's effective tag AS SEEN BY X as
        min({base_tag(Y)} u {base_tag(Z) : Z in blocks(Y), Z != X})
    -- X's own claim is deliberately excluded from Y's credit (crediting Y
    with the very vehicle asking to evict it would make the eviction
    condition self-referential and mathematically unsatisfiable, since
    that credited minimum could never exceed X's own tag). X may evict Y
    only if base_tag(X) is STRICTLY smaller than this. Unlike the original
    zone-occupancy version, this reduces to a SINGLE-level (non-recursive)
    credit: a PENDING vehicle holds no reservation, so it can never itself
    be blocking anyone, and no transitive TENTATIVE-blocks-TENTATIVE chain
    exists in this admission model -- a genuine structural simplification
    versus the blocking-model original, not an oversight.

    Only TENTATIVE vehicles are ever evictable; LOCKED occupancy is always
    a hard constraint, same as run_admission_episode."""
    env = AdmissionEnv(detection_window=detection_window, commit_window=commit_window)
    env.reset(arrivals, episode_duration)

    by_id = {v.id: v for v in arrivals}
    arrivals_sorted = sorted(arrivals, key=lambda v: v.arrival_time)
    pending_ids: set = set()

    events: List[float] = []
    for v in arrivals_sorted:
        heapq.heappush(events, max(0.0, v.arrival_time - detection_window))
    heapq.heappush(events, episode_duration)

    processed: set = set()
    while events:
        t = heapq.heappop(events)
        if t in processed or t > episode_duration + 1e-9:
            continue
        processed.add(t)
        env.current_time = t
        env.lock_near()

        for v in arrivals_sorted:
            if v.id in env.admitted_ids or v.id in pending_ids:
                continue
            if v.arrival_time - t <= detection_window + 1e-9:
                pending_ids.add(v.id)

        while True:
            blocked_by: Dict[int, set] = {}
            feasible: List[DynamicVehicle] = []
            for vid in list(pending_ids):
                v = by_id[vid]
                s1, blockers = env.blockers_for(v)
                if s1 is None:
                    continue
                if not blockers:
                    feasible.append(v)
                else:
                    blocked_by[vid] = blockers

            if feasible:
                chosen = min(feasible, key=tag_fn)
                env.admit(chosen)
                pending_ids.discard(chosen.id)
                continue

            blocks_map: Dict[int, List[int]] = {}
            for x_id, blockers in blocked_by.items():
                for y_id in blockers:
                    blocks_map.setdefault(y_id, []).append(x_id)

            # Only evict on X's behalf if doing so would FULLY unblock X --
            # partial eviction (justified against one blocker while another,
            # unrelated blocker still stands) never actually helps X and
            # just wastes a cycle re-evicting the same vehicle forever, since
            # X can never benefit from it. A multi-zone route (e.g. a long
            # route sharing a zone with one short-route vehicle) can leave X
            # blocked by an entirely separate vehicle in a different zone;
            # discovered via a genuine infinite-oscillation bug where a
            # 5-zone vehicle kept "winning" an eviction against a 2-zone
            # vehicle blocking only ITS shared zone, while a SEPARATE
            # blocker elsewhere kept the request from ever landing.
            best_evict = None  # (x_tag, x_id, [y_ids to evict])
            for x_id, blockers in blocked_by.items():
                x = by_id[x_id]
                x_tag = tag_fn(x)
                # A same-lane LEADER of x can never be usefully evicted: x's
                # own admissibility is gated on that leader being admitted
                # (hard no-overtake FIFO precondition, see
                # _effective_start), so removing it doesn't relieve x -- it
                # makes x UNCHECKABLE (leaders_ready=False) until the leader
                # is re-admitted, and since the leader itself then has no
                # blockers of its own, it immediately wins the next
                # "feasible" admission and re-blocks x identically. Found
                # via a genuine infinite oscillation between two same-lane
                # vehicles evicting/re-admitting each other forever.
                leader_ids = {leader.id for leader in env._same_lane_leaders(x)}
                to_evict = []
                fully_unblockable = True
                for y_id in blockers:
                    if y_id in leader_ids:
                        fully_unblockable = False  # same-lane leader: never evictable
                        break
                    if y_id not in env.tentative_ids:
                        fully_unblockable = False  # LOCKED occupancy is never evictable
                        break
                    others = [z for z in blocks_map.get(y_id, []) if z != x_id]
                    y_effective_excl_x = min(
                        [tag_fn(by_id[y_id])] + [tag_fn(by_id[z]) for z in others]
                    )
                    if x_tag < y_effective_excl_x:
                        to_evict.append(y_id)
                    else:
                        fully_unblockable = False
                        break
                if fully_unblockable and to_evict and (best_evict is None or x_tag < best_evict[0]):
                    best_evict = (x_tag, x_id, to_evict)

            if best_evict is None:
                break

            _, _, evict_y_ids = best_evict
            for evict_y_id in evict_y_ids:
                env.revoke_tentative(evict_y_id)
                pending_ids.add(evict_y_id)

        for vid in list(pending_ids):
            v = by_id[vid]
            feasible, retry = env.admission_check(v)
            if (not feasible) and retry is not None and retry <= episode_duration + 1e-9:
                heapq.heappush(events, retry)

        env.lock_near()

    for vid in list(env.tentative_ids):
        env.finalize_tentative(vid)

    n_seen = len(arrivals)
    for v in arrivals_sorted:
        if v.id not in env.admitted_ids:
            floor_wait = max(0.0, episode_duration - v.arrival_time)
            env.completed_log.append({
                "vehicle_id": v.id,
                "arrival_time": v.arrival_time,
                "finish_time": None,
                "waiting_time": floor_wait,
            })

    n_logged = len(env.completed_log)
    wt = sum(c["waiting_time"] for c in env.completed_log) / n_logged if n_logged else 0.0
    comp = len(env.admitted_ids) / n_seen if n_seen else 1.0
    return wt, comp, n_seen


@dataclass
class _CommitmentExchange:
    target_id: int
    blocker_ids: frozenset
    deadline: float
    gain: float
    departures: Dict[int, float]
    current: AdmissionEnv


def _clone_admission_env(env: AdmissionEnv) -> AdmissionEnv:
    trial = copy(env)
    trial.zones = {z: _ZoneOccupancy(list(occ.intervals)) for z, occ in env.zones.items()}
    trial.locked_ids = set(env.locked_ids)
    trial.tentative_ids = set(env.tentative_ids)
    trial.committed_s1 = dict(env.committed_s1)
    trial.completed_log = [dict(entry) for entry in env.completed_log]
    return trial


def _fill_detected_igreedy(env: AdmissionEnv, detected_ids: set) -> None:
    remaining = detected_ids - env.admitted_ids
    while remaining:
        feasible = [env._vehicles_by_id[vid] for vid in sorted(remaining)
                    if env.admission_check(env._vehicles_by_id[vid])[0]]
        if not feasible:
            return
        chosen = igreedy_admission_select(env, feasible)
        env.admit(chosen)
        remaining.remove(chosen.id)


def _project_detected_departures(env: AdmissionEnv, detected_ids: set) -> Optional[Dict[int, float]]:
    """Replay actual admission retries on a clone, without future arrivals.

    The projection may extend beyond the episode cutoff. Its future
    admissions must never be copied back into the live environment.
    """
    trial = _clone_admission_env(env)
    while True:
        _fill_detected_igreedy(trial, detected_ids)
        remaining = detected_ids - trial.admitted_ids
        if not remaining:
            return {vid: trial.committed_s1[vid] for vid in sorted(detected_ids)}
        trial.lock_near()
        next_event = _next_admission_event(trial, detected_ids)
        if next_event is None:
            return None
        trial.current_time = next_event



def _commitment_exchange_proposals(
    env: AdmissionEnv, detected_ids: set, stats: Optional[Dict] = None,
) -> List[_CommitmentExchange]:
    """One deterministic, complete candidate batch against a fixed snapshot.

    Each nonlocked target gets one proposal: retain its admitted FIFO
    predecessors, revise other tentative admissions, admit the target now,
    and rebuild using iGreedy and the same retry rules as the baseline.
    """
    detected_ids = set(detected_ids)
    baseline = _project_detected_departures(env, detected_ids)
    if baseline is None:
        return []
    proposals = []
    for target_id in sorted(detected_ids - env.locked_ids):
        if stats is not None:
            stats["evaluated"] = stats.get("evaluated", 0) + 1
        target = env._vehicles_by_id[target_id]
        trial = _clone_admission_env(env)
        leaders = sorted(env._same_lane_leaders(target), key=lambda v: (v.arrival_time, v.id))
        preserve = {v.id for v in leaders} & env.admitted_ids
        for vid in sorted(trial.tentative_ids - preserve):
            trial.revoke_tentative(vid)
        feasible = True
        for candidate in leaders + [target]:
            if candidate.id in trial.admitted_ids:
                continue
            if candidate.id not in detected_ids or not trial.admission_check(candidate)[0]:
                feasible = False
                break
            trial.admit(candidate)
        if not feasible:
            continue
        if trial.committed_s1[target_id] >= baseline[target_id] - 1e-9:
            continue
        if stats is not None:
            stats["proposed"] = stats.get("proposed", 0) + 1
        _fill_detected_igreedy(trial, detected_ids)
        alternative = _project_detected_departures(trial, detected_ids)
        if alternative is None or alternative.keys() != baseline.keys():
            continue
        if any(alternative[vid] != env.committed_s1[vid] for vid in env.locked_ids):
            continue
        blockers = frozenset(vid for vid in env.tentative_ids - {target_id}
                             if abs(alternative[vid] - env.committed_s1[vid]) > 1e-9)
        gain = sum(baseline.values()) - sum(alternative.values())
        if not blockers or gain <= 1e-9:
            continue
        proposals.append(_CommitmentExchange(
            target_id, blockers,
            min(env.committed_s1[vid] - env.commit_window for vid in blockers),
            gain, alternative, trial,
        ))
        if stats is not None:
            stats["profitable"] = stats.get("profitable", 0) + 1
    return proposals


def _next_admission_event(env: AdmissionEnv, detected_ids: set) -> Optional[float]:
    """Recompute retry/commitment events; obsolete trial events are cancelled."""
    events = [retry for vid in sorted(detected_ids - env.admitted_ids)
              for _, retry in [env.admission_check(env._vehicles_by_id[vid])]
              if retry is not None and retry > env.current_time + 1e-9]
    events.extend(env.committed_s1[vid] - env.commit_window
                  for vid in env.tentative_ids
                  if env.committed_s1[vid] - env.commit_window > env.current_time + 1e-9)
    if env.episode_duration > env.current_time + 1e-9:
        events.append(env.episode_duration)
    return min(events) if events else None


def _exchange_order_key(proposal: _CommitmentExchange, order: str) -> tuple:
    arrival = proposal.current._vehicles_by_id[proposal.target_id].arrival_time
    if order == "deadline":
        return (proposal.deadline, -proposal.gain, arrival, proposal.target_id)
    if order == "arrival":
        return (arrival, -proposal.gain, proposal.target_id)
    if order == "gain":
        return (-proposal.gain, arrival, proposal.target_id)
    raise ValueError(f"unknown order: {order!r}")


def _run_commitment_exchange_round(
    env: AdmissionEnv,
    detected_ids: set,
    order: str = "deadline",
    stats: Optional[Dict] = None,
) -> None:
    """Fill pending admissions, then atomically apply profitable exchanges.

    Recompute the full candidate batch after each acceptance and stop
    when no profitable proposal remains. All orderings use the same
    candidate construction.
    """
    if order not in {"deadline", "arrival", "gain"}:
        raise ValueError(f"unknown order: {order!r}")
    detected_ids = set(detected_ids)
    _fill_detected_igreedy(env, detected_ids)
    while True:
        if not detected_ids - env.locked_ids:
            return
        proposals = _commitment_exchange_proposals(env, detected_ids, stats)
        if not proposals:
            return
        chosen = min(proposals, key=lambda p: _exchange_order_key(p, order))
        env.__dict__.update(chosen.current.__dict__)
        if stats is not None:
            stats["accepted"] = stats.get("accepted", 0) + 1
            stats["planned_gain"] = stats.get("planned_gain", 0.0) + chosen.gain


def run_admission_episode_commitment_exchange(
    arrivals: List[DynamicVehicle],
    episode_duration: float,
    detection_window: float,
    commit_window: float = 2.5,
    order: str = "deadline",
    stats: Optional[Dict] = None,
) -> Tuple[float, float, int]:
    """Revise beneficial tentative admissions before each actual lock gate.

    Each acceptance decreases the full detected-set departure sum under
    the same fixed locks and admission/retry rules. Counterfactuals omit
    future arrivals and may finish beyond the episode cutoff; only their
    current-time admissions are applied. This does not guarantee a better
    realized outcome after future arrivals.

    deadline/arrival/gain use identical proposal construction, applying
    one profitable exchange at a time until no profitable proposal remains.
    Eligibility deadlines may already have passed; actual TENTATIVE state,
    including its FIFO lock gate, determines revisability. Existing LOCKED
    reservations are never revised.
    """
    if order not in {"deadline", "arrival", "gain"}:
        raise ValueError(f"unknown order: {order!r}")
    if stats is not None:
        for key in ("proposed", "evaluated", "profitable", "accepted", "planned_gain"):
            stats.setdefault(key, 0)
    env = AdmissionEnv(detection_window=detection_window, commit_window=commit_window)
    env.reset(arrivals, episode_duration)
    detections = sorted({max(0.0, v.arrival_time - detection_window) for v in arrivals})
    detected_ids = set()
    t = 0.0
    while t <= episode_duration + 1e-9:
        env.current_time = t
        detected_ids.update(v.id for v in arrivals
                            if v.arrival_time - detection_window <= t + 1e-9)
        _run_commitment_exchange_round(env, detected_ids, order, stats=stats)
        env.lock_near()
        if t >= episode_duration - 1e-9:
            break
        events = [episode_duration]
        next_detection = next((event for event in detections if event > t + 1e-9), None)
        if next_detection is not None:
            events.append(next_detection)
        next_admission = _next_admission_event(env, detected_ids)
        if next_admission is not None:
            events.append(next_admission)
        t = min(events)

    for vid in sorted(env.tentative_ids):
        env.finalize_tentative(vid)
    for vehicle in arrivals:
        if vehicle.id not in env.admitted_ids:
            env.completed_log.append({
                "vehicle_id": vehicle.id,
                "arrival_time": vehicle.arrival_time,
                "finish_time": None,
                "waiting_time": max(0.0, episode_duration - vehicle.arrival_time),
            })
    count = len(arrivals)
    waiting = sum(entry["waiting_time"] for entry in env.completed_log) / count if count else 0.0
    completion = len(env.admitted_ids) / count if count else 1.0
    return waiting, completion, count
