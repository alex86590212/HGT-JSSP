"""Build PyG HeteroData from the current DynamicIntersectionEnv state.

Adapted from intersection_scheduler.environment.graph_builder for the 3-state
operation model. Unlike the offline builder, edges are NOT cached across
steps — the vehicle/operation set changes every time a new vehicle is
detected or a completed vehicle is removed, so seq/lane/owns/hosts edges are
rebuilt fresh each replan (the graphs here are small enough that this is
cheap; see dynamic_scheduler/environment/dynamic_intersection.py for the
lifecycle that drives these changes).

Operation feature index 0 ("open" flag: 1.0 if LOCKED else 0.0) matches the
role of feature 0 in the offline builder (SchedulingPolicy.forward and
ppo_update pool the critic over operations where feature 0 == 0, i.e. the
still-actionable set) — kept semantically compatible so
intersection_scheduler.model.policy.SchedulingPolicy works unchanged here.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Dict, Optional

import torch
from torch_geometric.data import HeteroData

if TYPE_CHECKING:
    from dynamic_scheduler.environment.dynamic_intersection import DynamicIntersectionEnv

from dynamic_scheduler.environment.dynamic_intersection import OpState


def build_hetero_graph(
    env: "DynamicIntersectionEnv",
    feasible_mask: Optional[torch.Tensor] = None,
) -> HeteroData:
    data = HeteroData()
    ops = env.operations
    vehicles = list(env.vehicles.values())
    zones = env.zones
    n_ops = len(ops)
    zone_ids = sorted(zones.keys())
    zone_idx = {zid: i for i, zid in enumerate(zone_ids)}
    veh_idx = {v.id: i for i, v in enumerate(vehicles)}

    max_p = env.norm_max_p or 1.0
    max_c = env.norm_max_c or 1.0
    max_r = env.norm_max_r or 1.0

    if feasible_mask is None:
        from dynamic_scheduler.environment.feasibility import compute_feasible_set
        feasible_mask = compute_feasible_set(env)

    # ----------------------------------------------------------------
    # n_conflicts per operation (active Type-3 edges)
    # ----------------------------------------------------------------
    n_conflicts_per_op = [0] * n_ops
    for (a, b) in env.active_conflict_edges:
        n_conflicts_per_op[a] += 1
        n_conflicts_per_op[b] += 1
    max_conflicts = max(n_conflicts_per_op) if n_conflicts_per_op else 1.0
    max_conflicts = max_conflicts or 1.0

    # ----------------------------------------------------------------
    # Operation nodes  [n_ops, 7]
    # [locked, tentative, c(i,j), p(i,j), j/n_i, is_plannable, n_conflicts]
    # ----------------------------------------------------------------
    op_feats = []
    for idx, op in enumerate(ops):
        locked = 1.0 if op.state == OpState.LOCKED else 0.0
        tentative = 1.0 if op.state == OpState.TENTATIVE else 0.0
        c = op.earliest_finish / max_c
        p = op.processing_time / max_p
        pos = op.route_position / op.route_length if op.route_length > 0 else 0.0
        plannable = 1.0 if feasible_mask[idx].item() else 0.0
        nc = n_conflicts_per_op[idx] / max_conflicts
        op_feats.append([locked, tentative, c, p, pos, plannable, nc])
    data["operation"].x = torch.tensor(op_feats, dtype=torch.float) if op_feats else torch.zeros((0, 7))

    # ----------------------------------------------------------------
    # Vehicle nodes  [n_veh, 5]
    # [r_i, n_i, velocity, zones_locked_frac, time_to_arrival]
    # ----------------------------------------------------------------
    max_route = max((len(v.route) for v in vehicles), default=1) or 1.0
    max_vel = max((v.velocity for v in vehicles), default=1.0) or 1.0
    max_tta = max((v.arrival_time - env.current_time for v in vehicles), default=1.0)
    max_tta = max_tta or 1.0
    veh_feats = []
    for v in vehicles:
        r = v.arrival_time / max_r
        n = len(v.route) / max_route
        vel = v.velocity / max_vel
        vops = [op for op in ops if op.vehicle_id == v.id]
        locked_frac = (
            sum(1 for op in vops if op.state == OpState.LOCKED) / len(vops)
            if vops else 0.0
        )
        tta = max(v.arrival_time - env.current_time, 0.0) / max_tta
        veh_feats.append([r, n, vel, locked_frac, tta])
    data["vehicle"].x = torch.tensor(veh_feats, dtype=torch.float) if veh_feats else torch.zeros((0, 5))

    # ----------------------------------------------------------------
    # Zone nodes  [n_zones, 5]
    # [occupied, time_free, n_competing, x, y]
    # ----------------------------------------------------------------
    max_tf = max((z.time_free for z in zones.values()), default=1.0) or 1.0
    max_nc_z = max((z.n_competing for z in zones.values()), default=1) or 1.0
    zone_feats = []
    for zid in zone_ids:
        z = zones[zid]
        occ = 1.0 if z.time_free > env.current_time + 1e-9 else 0.0
        tf = z.time_free / max_tf
        nc = z.n_competing / max_nc_z
        zone_feats.append([occ, tf, nc, z.x, z.y])
    data["zone"].x = torch.tensor(zone_feats, dtype=torch.float) if zone_feats else torch.zeros((0, 5))

    # ----------------------------------------------------------------
    # Type-1 seq edges: op(i,j) -> op(i,j+1), directed route order
    # ----------------------------------------------------------------
    seq_src, seq_dst, seq_attr = [], [], []
    for i, op in enumerate(ops):
        if op.route_position < op.route_length - 1:
            for j, op2 in enumerate(ops):
                if (op2.vehicle_id == op.vehicle_id
                        and op2.route_position == op.route_position + 1):
                    seq_src.append(i)
                    seq_dst.append(j)
                    seq_attr.append([op.processing_time / max_p])
                    break
    _set_edge(data, "operation", "seq", "operation", seq_src, seq_dst, seq_attr, attr_dim=1)

    # ----------------------------------------------------------------
    # Type-2 lane edges: same-lane ordering (directed, earlier -> later)
    # ----------------------------------------------------------------
    lane_src, lane_dst, lane_attr = [], [], []
    by_entry: dict = defaultdict(list)
    for v in vehicles:
        if v.route:
            by_entry[v.route[0]].append(v)
    for _entry_zone, group in by_entry.items():
        sorted_group = sorted(group, key=lambda v: v.arrival_time)
        for k in range(len(sorted_group) - 1):
            leader = sorted_group[k]
            follower = sorted_group[k + 1]
            gap = follower.arrival_time - leader.arrival_time
            li = next(
                (i for i, o in enumerate(ops)
                 if o.vehicle_id == leader.id and o.route_position == 0),
                None,
            )
            fi = next(
                (i for i, o in enumerate(ops)
                 if o.vehicle_id == follower.id and o.route_position == 0),
                None,
            )
            if li is not None and fi is not None:
                lane_src.append(li)
                lane_dst.append(fi)
                lane_attr.append([gap / (max_r or 1.0)])
    _set_edge(data, "operation", "lane", "operation", lane_src, lane_dst, lane_attr, attr_dim=1)

    # ----------------------------------------------------------------
    # Type-3 conflict edges: undirected active conflicts (both directions)
    # ----------------------------------------------------------------
    conf_src, conf_dst, conf_attr = [], [], []
    for (a, b) in env.active_conflict_edges:
        oa, ob = ops[a], ops[b]
        va = env.vehicles.get(oa.vehicle_id)
        vb = env.vehicles.get(ob.vehicle_id)
        if va is None or vb is None:
            continue
        delta = (va.arrival_time - vb.arrival_time) / (max_r or 1.0)
        urgency_a = 1.0 / (1.0 + oa.route_position)
        urgency_b = 1.0 / (1.0 + ob.route_position)
        conf_src.extend([a, b])
        conf_dst.extend([b, a])
        conf_attr.extend([
            [delta, urgency_a, urgency_b],
            [-delta, urgency_b, urgency_a],
        ])
    _set_edge(data, "operation", "conflict", "operation", conf_src, conf_dst, conf_attr, attr_dim=3)

    # ----------------------------------------------------------------
    # Vehicle -> Operation edges (vehicle owns its ops)
    # ----------------------------------------------------------------
    vo_src, vo_dst = [], []
    for i, op in enumerate(ops):
        if op.vehicle_id in veh_idx:
            vo_src.append(veh_idx[op.vehicle_id])
            vo_dst.append(i)
    data["vehicle", "owns", "operation"].edge_index = (
        torch.tensor([vo_src, vo_dst], dtype=torch.long) if vo_src else torch.zeros((2, 0), dtype=torch.long)
    )

    # ----------------------------------------------------------------
    # Zone -> Operation edges (zone hosts all ops that use it)
    # ----------------------------------------------------------------
    zo_src, zo_dst = [], []
    for i, op in enumerate(ops):
        if op.zone_id in zone_idx:
            zo_src.append(zone_idx[op.zone_id])
            zo_dst.append(i)
    data["zone", "hosts", "operation"].edge_index = (
        torch.tensor([zo_src, zo_dst], dtype=torch.long) if zo_src else torch.zeros((2, 0), dtype=torch.long)
    )

    return data


def _set_edge(data, src_type, rel, dst_type, src, dst, attr, attr_dim: int) -> None:
    if src:
        data[src_type, rel, dst_type].edge_index = torch.tensor([src, dst], dtype=torch.long)
        data[src_type, rel, dst_type].edge_attr = torch.tensor(attr, dtype=torch.float)
    else:
        data[src_type, rel, dst_type].edge_index = torch.zeros((2, 0), dtype=torch.long)
        data[src_type, rel, dst_type].edge_attr = torch.zeros((0, attr_dim))
