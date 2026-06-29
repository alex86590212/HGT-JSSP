"""Feasibility checker: compute A(t) with deadlock detection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

import torch

if TYPE_CHECKING:
    from intersection_scheduler.environment.intersection import IntersectionEnv, Operation


def compute_feasible_set(env: "IntersectionEnv") -> torch.BoolTensor:
    """Return bool mask [num_ops] where True = op is in A(t)."""
    n = len(env.operations)
    mask = [False] * n

    # Build adjacency for deadlock DFS: directed edges representing "must come before"
    # Initially from Type-1 (route order) and Type-2 (same-lane order) constraints.
    # We add a tentative edge when evaluating each candidate.
    adj = _build_base_adjacency(env)

    for idx, op in enumerate(env.operations):
        if _check_feasible(idx, op, env, adj):
            mask[idx] = True

    return torch.tensor(mask, dtype=torch.bool)


def _check_feasible(
    idx: int,
    op: "Operation",
    env: "IntersectionEnv",
    base_adj: Dict[int, List[int]],
) -> bool:
    # 1. Not yet scheduled
    if op.scheduled:
        return False

    # 2. Route predecessor must be done
    if op.route_position > 0:
        pred = _predecessor(op, env)
        if pred is None or not pred.scheduled:
            return False

    # 3. Same-lane predecessor: the vehicle immediately ahead in the same
    #    entry lane must have its first operation already scheduled.
    if op.route_position == 0:
        if not _same_lane_leaders_done(op, env):
            return False

    # 4. Zone must not be occupied (by another op whose time_free > current_time)
    zone = env.zones.get(op.zone_id)
    if zone is not None and zone.time_free > env.current_time + 1e-9:
        # Check if any unscheduled op holds the zone
        holder = _zone_holder(op.zone_id, env)
        if holder is not None and not holder.scheduled:
            return False

    # 5. Vehicle must have arrived
    vehicle = next((v for v in env.vehicles if v.id == op.vehicle_id), None)
    if vehicle is None or vehicle.arrival_time > env.current_time + 1e-9:
        return False

    # 6. No deadlock: tentatively resolve this op's Type-3 conflicts and check
    #    for cycles in the scheduling precedence graph.
    if _would_cause_deadlock(idx, op, env, base_adj):
        return False

    return True


def _predecessor(op: "Operation", env: "IntersectionEnv") -> Optional["Operation"]:
    for o in env.operations:
        if o.vehicle_id == op.vehicle_id and o.route_position == op.route_position - 1:
            return o
    return None


def _same_lane_leaders_done(op: "Operation", env: "IntersectionEnv") -> bool:
    """Check that all vehicles in the same entry lane that arrived earlier are fully scheduled (at least first op)."""
    vehicle = next((v for v in env.vehicles if v.id == op.vehicle_id), None)
    if vehicle is None or not vehicle.route:
        return True
    entry_zone = vehicle.route[0]
    for v2 in env.vehicles:
        if v2.id == vehicle.id:
            continue
        if not v2.route or v2.route[0] != entry_zone:
            continue
        if v2.arrival_time < vehicle.arrival_time - 1e-9:
            # v2 is a leader; its first operation must be scheduled
            first_op = next(
                (o for o in env.operations if o.vehicle_id == v2.id and o.route_position == 0),
                None,
            )
            if first_op is None or not first_op.scheduled:
                return False
        elif abs(v2.arrival_time - vehicle.arrival_time) < 1e-9 and v2.id < vehicle.id:
            # Tie-break by id
            first_op = next(
                (o for o in env.operations if o.vehicle_id == v2.id and o.route_position == 0),
                None,
            )
            if first_op is None or not first_op.scheduled:
                return False
    return True


def _zone_holder(zone_id: int, env: "IntersectionEnv") -> Optional["Operation"]:
    """Return the unscheduled op currently "holding" the zone, if any."""
    # The zone is "held" by an op that's the next unscheduled op of a vehicle
    # that already has its predecessor in the zone scheduled.
    # Simplified: find the most recently scheduled op for this zone.
    for op in env.operations:
        if op.zone_id == zone_id and not op.scheduled:
            # Check if this op is the current "front" op of its vehicle
            pred = _predecessor(op, env)
            if pred is not None and pred.scheduled:
                return op
    return None


def _build_base_adjacency(env: "IntersectionEnv") -> Dict[int, List[int]]:
    """Build directed graph of "must precede" edges for deadlock detection.

    Edges represent: if a -> b, then op a must complete before op b starts.
    - Type-1: route predecessor -> successor (all pairs)
    - Resolved Type-3 conflicts: the scheduled op points to all unscheduled ops
      it was in conflict with in the same zone.
    """
    adj: Dict[int, List[int]] = {i: [] for i in range(len(env.operations))}

    # Type-1 edges (route order)
    for idx, op in enumerate(env.operations):
        if op.route_position > 0:
            pred = _predecessor(op, env)
            if pred is not None:
                pred_idx = next(
                    i for i, o in enumerate(env.operations) if o is pred
                )
                adj[pred_idx].append(idx)

    # Resolved Type-3: scheduled op precedes unscheduled same-zone ops
    # (because scheduling established the order)
    all_conflict_pairs = env.conflict_edges
    active_set = set(map(frozenset, env.active_conflict_edges))
    for (a, b) in all_conflict_pairs:
        pair = frozenset({a, b})
        if pair not in active_set:
            # Conflict is resolved: whichever is scheduled goes first
            oa, ob = env.operations[a], env.operations[b]
            if oa.scheduled and not ob.scheduled:
                adj[a].append(b)
            elif ob.scheduled and not oa.scheduled:
                adj[b].append(a)

    return adj


def _would_cause_deadlock(
    idx: int,
    op: "Operation",
    env: "IntersectionEnv",
    base_adj: Dict[int, List[int]],
) -> bool:
    """Check if scheduling op `idx` creates a cycle.

    Tentatively adds edges: idx -> all other ops in its active conflicts
    (meaning idx is resolved first, others must wait for idx's zone to free).
    Then runs DFS cycle detection.
    """
    # Build tentative adjacency including the new resolution edges
    adj: Dict[int, List[int]] = {k: list(v) for k, v in base_adj.items()}

    # Adding tentative ordering: idx goes before all unscheduled ops it conflicts with
    for (a, b) in env.active_conflict_edges:
        if a == idx:
            adj[idx].append(b)
        elif b == idx:
            adj[idx].append(a)

    # DFS cycle detection
    WHITE, GRAY, BLACK = 0, 1, 2
    color = [WHITE] * len(env.operations)

    def dfs(node: int) -> bool:
        color[node] = GRAY
        for neighbor in adj.get(node, []):
            if color[neighbor] == GRAY:
                return True
            if color[neighbor] == WHITE and dfs(neighbor):
                return True
        color[node] = BLACK
        return False

    # Only check reachability from nodes that changed — start from idx
    return dfs(idx)
