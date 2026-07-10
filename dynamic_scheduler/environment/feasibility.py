"""Feasibility checker for DynamicIntersectionEnv's 3-state operation model.

Adapted from intersection_scheduler.environment.feasibility: same Type-1
(route order), Type-2 (same-lane FIFO), Type-3 (zone exclusivity) constraints
and deadlock check, but operating on UNSCHEDULED/TENTATIVE/LOCKED states
instead of the offline env's binary scheduled/unscheduled.

Key difference from the offline checker: an operation is "plannable" if it is
not LOCKED (i.e. UNSCHEDULED or TENTATIVE) — TENTATIVE ops remain eligible
since they can be revised. LOCKED ops are never plannable again.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

import torch

if TYPE_CHECKING:
    from dynamic_scheduler.environment.dynamic_intersection import (
        DynamicIntersectionEnv,
        DynamicOperation,
    )

from dynamic_scheduler.environment.dynamic_intersection import OpState


def compute_feasible_set(env: "DynamicIntersectionEnv") -> torch.BoolTensor:
    """Return bool mask [num_ops] where True = op is plannable right now."""
    n = len(env.operations)
    mask = [False] * n

    adj = _build_base_adjacency(env)

    for idx, op in enumerate(env.operations):
        if _check_feasible(idx, op, env, adj):
            mask[idx] = True

    return torch.tensor(mask, dtype=torch.bool)


def _check_feasible(
    idx: int,
    op: "DynamicOperation",
    env: "DynamicIntersectionEnv",
    base_adj: Dict[int, List[int]],
) -> bool:
    # 1. Must not be locked (locked ops are irrevocably done planning)
    if op.state == OpState.LOCKED:
        return False

    # 2. Route predecessor must at least have a tentative plan
    if op.route_position > 0:
        pred = _predecessor(op, env)
        if pred is None or pred.state == OpState.UNSCHEDULED:
            return False

    # 3. Same-lane predecessor: the vehicle immediately ahead in the same
    #    entry lane must have at least a tentative first operation.
    if op.route_position == 0:
        if not _same_lane_leaders_planned(op, env):
            return False

    # 4. Vehicle must be detected (present in env.vehicles at all — the env
    #    only ever adds detected vehicles' ops, so this is always true here,
    #    kept for parity with the offline checker's arrival-time gate).
    if op.vehicle_id not in env.vehicles:
        return False

    # 5. No deadlock: tentatively resolve this op's active Type-3 conflicts
    #    and check for cycles.
    if _would_cause_deadlock(idx, op, env, base_adj):
        return False

    return True


def _predecessor(op: "DynamicOperation", env: "DynamicIntersectionEnv") -> Optional["DynamicOperation"]:
    for o in env.operations:
        if o.vehicle_id == op.vehicle_id and o.route_position == op.route_position - 1:
            return o
    return None


def _same_lane_leaders_planned(op: "DynamicOperation", env: "DynamicIntersectionEnv") -> bool:
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
            first_op = next(
                (o for o in env.operations if o.vehicle_id == v2.id and o.route_position == 0),
                None,
            )
            if first_op is None or first_op.state == OpState.UNSCHEDULED:
                return False
    return True


def _build_base_adjacency(env: "DynamicIntersectionEnv") -> Dict[int, List[int]]:
    """Directed 'must precede' graph for deadlock detection.

    - Type-1: route predecessor -> successor.
    - Resolved Type-3: a LOCKED op precedes unresolved same-zone ops (its
      ordering is now irrevocable). TENTATIVE plans don't create a hard
      precedence edge since they can still be revised.
    """
    adj: Dict[int, List[int]] = {i: [] for i in range(len(env.operations))}

    for idx, op in enumerate(env.operations):
        if op.route_position > 0:
            pred = _predecessor(op, env)
            if pred is not None:
                pred_idx = next(i for i, o in enumerate(env.operations) if o is pred)
                adj[pred_idx].append(idx)

    all_conflict_pairs = env.conflict_edges
    active_set = set(map(frozenset, env.active_conflict_edges))
    for (a, b) in all_conflict_pairs:
        pair = frozenset({a, b})
        if pair not in active_set:
            oa, ob = env.operations[a], env.operations[b]
            if oa.state == OpState.LOCKED and ob.state != OpState.LOCKED:
                adj[a].append(b)
            elif ob.state == OpState.LOCKED and oa.state != OpState.LOCKED:
                adj[b].append(a)

    return adj


def _would_cause_deadlock(
    idx: int,
    op: "DynamicOperation",
    env: "DynamicIntersectionEnv",
    base_adj: Dict[int, List[int]],
) -> bool:
    adj: Dict[int, List[int]] = {k: list(v) for k, v in base_adj.items()}

    for (a, b) in env.active_conflict_edges:
        if a == idx:
            adj[idx].append(b)
        elif b == idx:
            adj[idx].append(a)

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

    return dfs(idx)
