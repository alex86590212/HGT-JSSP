"""Feasibility checker for DynamicIntersectionEnv's 3-state operation model.

Adapted from intersection_scheduler.environment.feasibility: same Type-1
(route order), Type-2 (same-lane FIFO), Type-3 (zone exclusivity) constraints
and deadlock check, but operating on UNSCHEDULED/TENTATIVE/LOCKED states
instead of the offline env's binary scheduled/unscheduled.

Key difference from the offline checker: an operation is "plannable" if it is
not LOCKED (i.e. UNSCHEDULED or TENTATIVE) — TENTATIVE ops remain eligible
since they can be revised (revision = requeue to the back of the zone queue).

Deadlock semantics under zone-binding tentative plans: planning candidate X
appends it to the BACK of its zone's tentative priority queue. The hard
"must precede" edges at that moment are:
  - route order: every op precedes its route successor;
  - zone queues: consecutive TENTATIVE ops in each queue (except edges OUT
    of X, which vacates its current slot when requeued);
  - every tentative op in X's zone precedes X (X lands at the back);
  - X precedes every UNSCHEDULED op in its zone (they will queue behind X);
  - every tentative op precedes UNSCHEDULED ops in its own zone (same reason).
LOCKED ops create no ordering edges: they are static occupancy windows with
frozen times that tentative ops gap-fill around, so they cannot participate
in a cycle. X is feasible only if placing it creates no cycle through X.
Cycles not through X cannot exist: they would have had to be created by
planning some earlier candidate, which this same check prevented.

Perf note: this runs once per policy decision (~1000+ times per episode), so
lookups go through O(1) index maps built once per call, and `candidates`
restricts the expensive checks to the ops the caller will actually consider
— entries outside it are simply left False, identical to the caller masking
them out afterwards.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple

import torch

if TYPE_CHECKING:
    from dynamic_scheduler.environment.dynamic_intersection import (
        DynamicIntersectionEnv,
        DynamicOperation,
    )

from dynamic_scheduler.environment.dynamic_intersection import OpState


def compute_feasible_set(
    env: "DynamicIntersectionEnv",
    candidates: Optional[Iterable[int]] = None,
) -> torch.BoolTensor:
    """Return bool mask [num_ops] where True = op is plannable right now.

    candidates: if given, only these op indices are checked (all others stay
    False). Feasibility of one op never depends on the mask of another, so
    this is exactly equivalent to computing the full mask and zeroing the
    rest — just without paying for checks whose result would be discarded.
    """
    n = len(env.operations)
    mask = [False] * n

    if candidates is None:
        idx_iter: Iterable[int] = range(n)
    else:
        idx_iter = [i for i in candidates if 0 <= i < n]
        if not idx_iter:
            return torch.tensor(mask, dtype=torch.bool)

    ctx = _FeasibilityContext(env)

    for idx in idx_iter:
        op = env.operations[idx]
        if _check_feasible(op, env, ctx):
            mask[idx] = True

    return torch.tensor(mask, dtype=torch.bool)


class _FeasibilityContext:
    """Shared per-call lookup structures (built once, used by every check)."""

    def __init__(self, env: "DynamicIntersectionEnv") -> None:
        self.pos_index: Dict[Tuple[int, int], "DynamicOperation"] = {
            (o.vehicle_id, o.route_position): o for o in env.operations
        }
        # Next tentative op in each zone queue (queue order = priority order;
        # queues contain only TENTATIVE ops — locked ops are static windows).
        #
        # Blocking (Huang et al. Section III-A): a precedes b in the queue,
        # but a's zone doesn't actually release for b until a's OWN route
        # successor starts (a retains its zone until then) — so the true
        # must-precede source is a's route successor, not a itself. Falls
        # back to a when a has no route successor (last op on its route) or
        # that successor isn't planned yet (nothing to redirect through). A
        # single op can be the redirect target for one queue edge AND a's
        # regular queue source for another, so this is list-valued.
        self.queue_next: Dict[int, List["DynamicOperation"]] = {}

        def add_redirect(a: "DynamicOperation", b: "DynamicOperation") -> None:
            route_succ_a = self.pos_index.get((a.vehicle_id, a.route_position + 1))
            source = (
                route_succ_a
                if route_succ_a is not None and route_succ_a.state != OpState.UNSCHEDULED
                else a
            )
            self.queue_next.setdefault(id(source), []).append(b)

        for queue in env._zone_queue.values():
            for a, b in zip(queue, queue[1:]):
                add_redirect(a, b)
            # A LOCKED operation's own zone needs the same redirect into
            # whichever TENTATIVE op is the head of that zone's queue: queues
            # only ever lose members from the front via locking, so the head
            # is exactly "whatever used to be queued right after the
            # now-locked op" — mirrors _recompute_times's identical edge.
            if queue:
                head = queue[0]
                for o in env.operations:
                    if o.state == OpState.LOCKED and o.zone_id == head.zone_id:
                        add_redirect(o, head)
        self.unscheduled_by_zone: Dict[int, List["DynamicOperation"]] = {}
        for o in env.operations:
            if o.state == OpState.UNSCHEDULED:
                self.unscheduled_by_zone.setdefault(o.zone_id, []).append(o)


def _check_feasible(
    op: "DynamicOperation",
    env: "DynamicIntersectionEnv",
    ctx: _FeasibilityContext,
) -> bool:
    # 1. Must not be locked (locked ops are irrevocably done planning)
    if op.state == OpState.LOCKED:
        return False

    # 2. Route predecessor must at least have a tentative plan
    if op.route_position > 0:
        pred = ctx.pos_index.get((op.vehicle_id, op.route_position - 1))
        if pred is None or pred.state == OpState.UNSCHEDULED:
            return False

    # 3. Same-lane predecessor: the vehicle immediately ahead in the same
    #    entry lane must have at least a tentative first operation.
    if op.route_position == 0:
        if not _same_lane_leaders_planned(op, env, ctx):
            return False

    # 4. Vehicle must be detected (present in env.vehicles at all — the env
    #    only ever adds detected vehicles' ops, so this is always true here,
    #    kept for parity with the offline checker's arrival-time gate).
    if op.vehicle_id not in env.vehicles:
        return False

    # 5. No deadlock: appending op to the back of its zone queue must not
    #    create a precedence cycle.
    if _would_cause_deadlock(op, env, ctx):
        return False

    return True


def _same_lane_leaders_planned(
    op: "DynamicOperation",
    env: "DynamicIntersectionEnv",
    ctx: _FeasibilityContext,
) -> bool:
    """All same-lane vehicles that arrived earlier must have at least a
    tentative (or locked) first operation."""
    vehicle = env.vehicles.get(op.vehicle_id)
    if vehicle is None or not vehicle.route:
        return True
    entry_zone = vehicle.route[0]
    for v2 in env.vehicles.values():
        if v2.id == vehicle.id:
            continue
        if not v2.route or v2.route[0] != entry_zone:
            continue
        is_leader = (
            v2.arrival_time < vehicle.arrival_time - 1e-9
            or (abs(v2.arrival_time - vehicle.arrival_time) < 1e-9 and v2.id < vehicle.id)
        )
        if is_leader:
            first_op = ctx.pos_index.get((v2.id, 0))
            if first_op is None or first_op.state == OpState.UNSCHEDULED:
                return False
    return True


def _would_cause_deadlock(
    x: "DynamicOperation", env: "DynamicIntersectionEnv", ctx: _FeasibilityContext
) -> bool:
    """DFS from the hypothetically-placed candidate X for a cycle through X.

    Neighbors follow the hard 'must precede' edges described in the module
    docstring, assembled lazily per node so nothing is copied per candidate.
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color: Dict[int, int] = {}

    # Blocking redirect, X's side, case 3 — X's own FUTURE zone-queue
    # predecessor: X lands at the back of its zone's queue, so whoever is
    # CURRENTLY last there (excluding X itself, for a replan already sitting
    # somewhere in that queue) becomes X's immediate predecessor the moment
    # this placement commits. That op's true release (its own route
    # successor, if TENTATIVE) must precede X — ctx has no edge for this
    # yet, since it was built from the queue as it stood BEFORE X's
    # insertion. This is an edge INTO x (tail_source -> x), so it belongs on
    # tail_source's outgoing neighbors, not x's — computed once here since
    # it never changes across the DFS.
    tail_source: Optional["DynamicOperation"] = None
    others = [m for m in env._zone_queue.get(x.zone_id, []) if m is not x]
    if others:
        tail = others[-1]
        route_succ_tail = ctx.pos_index.get((tail.vehicle_id, tail.route_position + 1))
        candidate_source = (
            route_succ_tail
            if route_succ_tail is not None and route_succ_tail.state != OpState.UNSCHEDULED
            else tail
        )
        if candidate_source is not x:
            tail_source = candidate_source

    def neighbors(o: "DynamicOperation") -> Iterable["DynamicOperation"]:
        route_succ = ctx.pos_index.get((o.vehicle_id, o.route_position + 1))
        if route_succ is not None:
            yield route_succ
        if tail_source is not None and o is tail_source:
            yield x
        if o is x:
            # X sits at the back of its queue: it precedes only the ops that
            # will be queued after it, i.e. the still-unscheduled ones.
            for u in ctx.unscheduled_by_zone.get(o.zone_id, []):
                if u is not x:
                    yield u
            # Blocking redirect, X's side, case 1 — X is a REPLAN (already
            # TENTATIVE): ctx was built from the CURRENT (pre-replan) state,
            # so if X is already the correctly-redirected release source for
            # some OTHER zone's queue successor (via X's own route
            # predecessor's zone membership, resolved when ctx was built),
            # that edge is a real, currently-binding obligation X carries
            # into its new back-of-queue slot — it must still be yielded
            # here, since the generic `o.state != UNSCHEDULED` branch below
            # is never reached for o is x (this branch returns).
            for existing_target in ctx.queue_next.get(id(o), ()):
                if existing_target is not x:
                    yield existing_target
            # Blocking redirect, X's side, case 2 — X is a FRESH placement
            # (was UNSCHEDULED): ctx.queue_next was built with X still
            # UNSCHEDULED, so any existing zone-queue edge sourced from X's
            # own route predecessor was left un-redirected (pointing from
            # the predecessor itself, per the same fallback rule used at
            # context-build time). Placing X changes that predecessor's true
            # release to X's own start — so X inherits that edge here, at the
            # one moment its route predecessor's successor identity actually
            # changes from "unscheduled" to "X".
            pred = ctx.pos_index.get((o.vehicle_id, o.route_position - 1))
            if pred is not None:
                for b_of_pred in ctx.queue_next.get(id(pred), ()):
                    if b_of_pred is not x:
                        yield b_of_pred
            return
        if o.state != OpState.UNSCHEDULED:
            for nxt in ctx.queue_next.get(id(o), ()):
                if nxt is not x:
                    yield nxt
            if o.zone_id == x.zone_id:
                yield x
            for u in ctx.unscheduled_by_zone.get(o.zone_id, []):
                yield u

    def dfs(o: "DynamicOperation") -> bool:
        color[id(o)] = GRAY
        for nb in neighbors(o):
            c = color.get(id(nb), WHITE)
            if c == GRAY:
                return True
            if c == WHITE and dfs(nb):
                return True
        color[id(o)] = BLACK
        return False

    return dfs(x)
