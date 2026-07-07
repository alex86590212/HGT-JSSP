"""Build PyG HeteroData from the current IntersectionEnv state."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Set, Tuple

import torch
from torch_geometric.data import HeteroData

if TYPE_CHECKING:
    from intersection_scheduler.environment.intersection import IntersectionEnv


def build_hetero_graph(
    env: "IntersectionEnv",
    feasible_mask: Optional[torch.Tensor] = None,
    static_edges: Optional[Dict] = None,
) -> HeteroData:
    """Build the heterogeneous graph for the current env state.

    feasible_mask: if provided, used directly instead of recomputing the
        feasible set (identical result, since it must be computed for the same
        env state anyway). If None, computed here.
    static_edges: if provided (from build_static_edges at reset time), the
        seq/lane/owns/hosts edge tensors are reused instead of rebuilt. These
        edges depend only on route_position/vehicle_id/zone_id/arrival_time,
        all immutable during an episode, so caching is exact. Type-3 conflict
        edges and all node features are always rebuilt (they change per step).
    """
    data = HeteroData()
    ops = env.operations
    vehicles = env.vehicles
    zones = env.zones
    n_ops = len(ops)
    n_veh = len(vehicles)
    zone_ids = sorted(zones.keys())
    n_zones = len(zone_ids)

    max_p = env.norm_max_p or 1.0
    max_c = env.norm_max_c or 1.0
    max_r = env.norm_max_r or 1.0
    max_n = env.norm_max_n or 1.0

    # ----------------------------------------------------------------
    # Feasible set — import lazily to avoid circular imports
    # ----------------------------------------------------------------
    if feasible_mask is None:
        from intersection_scheduler.environment.feasibility import compute_feasible_set
        feasible_mask = compute_feasible_set(env)  # BoolTensor [n_ops]

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
    # Operation nodes  [n_ops, 6]
    # [d(i,j), c(i,j), p(i,j), j/n_i, is_schedulable, n_conflicts]
    # ----------------------------------------------------------------
    op_feats = []
    for idx, op in enumerate(ops):
        d = 1.0 if op.scheduled else 0.0
        c = op.earliest_finish / max_c
        p = op.processing_time / max_p
        pos = op.route_position / op.route_length if op.route_length > 0 else 0.0
        sched = 1.0 if feasible_mask[idx].item() else 0.0
        nc = n_conflicts_per_op[idx] / max_conflicts
        op_feats.append([d, c, p, pos, sched, nc])
    data["operation"].x = torch.tensor(op_feats, dtype=torch.float)

    # ----------------------------------------------------------------
    # Vehicle nodes  [n_veh, 4]
    # [r_i, n_i, velocity, zones_done]
    # ----------------------------------------------------------------
    max_route = max((len(v.route) for v in vehicles), default=1) or 1.0
    max_vel = max((v.velocity for v in vehicles), default=1.0) or 1.0
    veh_feats = []
    for v in vehicles:
        r = v.arrival_time / max_r
        n = len(v.route) / max_route
        vel = v.velocity / max_vel
        done = sum(
            1 for op in ops
            if op.vehicle_id == v.id and op.scheduled
        ) / (len(v.route) or 1)
        veh_feats.append([r, n, vel, done])
    data["vehicle"].x = torch.tensor(veh_feats, dtype=torch.float)

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
    data["zone"].x = torch.tensor(zone_feats, dtype=torch.float)

    # ----------------------------------------------------------------
    # Static edges (seq, lane, owns, hosts) — reuse cache if provided,
    # otherwise build fresh via the single source of truth build_static_edges.
    # These depend only on immutable per-episode data, so the cached tensors
    # are byte-identical to a fresh build.
    # ----------------------------------------------------------------
    if static_edges is None:
        static_edges = build_static_edges(env)
    for (etype, key) in (
        (("operation", "seq", "operation"), "seq"),
        (("operation", "lane", "operation"), "lane"),
        (("vehicle", "owns", "operation"), "owns"),
        (("zone", "hosts", "operation"), "hosts"),
    ):
        edge_index, edge_attr = static_edges[key]
        data[etype].edge_index = edge_index
        if edge_attr is not None:
            data[etype].edge_attr = edge_attr

    # ----------------------------------------------------------------
    # Type-3 conflict edges: undirected active conflicts (both directions)
    # ----------------------------------------------------------------
    conf_src, conf_dst, conf_attr = [], [], []
    for (a, b) in env.active_conflict_edges:
        oa, ob = ops[a], ops[b]
        va = next((v for v in vehicles if v.id == oa.vehicle_id), None)
        vb = next((v for v in vehicles if v.id == ob.vehicle_id), None)
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
    if conf_src:
        data["operation", "conflict", "operation"].edge_index = torch.tensor(
            [conf_src, conf_dst], dtype=torch.long
        )
        data["operation", "conflict", "operation"].edge_attr = torch.tensor(
            conf_attr, dtype=torch.float
        )
    else:
        data["operation", "conflict", "operation"].edge_index = torch.zeros(
            (2, 0), dtype=torch.long
        )
        data["operation", "conflict", "operation"].edge_attr = torch.zeros((0, 3))

    return data


def build_static_edges(env: "IntersectionEnv") -> Dict:
    """Build the per-episode-static edge tensors (seq, lane, owns, hosts).

    These edges depend only on route_position, vehicle_id, zone_id, route[0]
    and arrival_time — all immutable during an episode — so they can be built
    once at reset time and reused every step. Returns a dict mapping each edge
    key to a (edge_index, edge_attr) tuple; edge_attr is None where the edge
    type has no attributes (owns, hosts).

    This is the single source of truth for these edges: build_hetero_graph
    calls it when no cache is supplied, so cached and fresh builds are exact.
    """
    ops = env.operations
    vehicles = env.vehicles
    zones = env.zones
    zone_ids = sorted(zones.keys())
    zone_idx = {zid: i for i, zid in enumerate(zone_ids)}
    veh_idx = {v.id: i for i, v in enumerate(vehicles)}
    max_p = env.norm_max_p or 1.0
    max_r = env.norm_max_r or 1.0

    # Type-1 seq edges: op(i,j) -> op(i,j+1), directed route order
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
    if seq_src:
        seq_edge = (
            torch.tensor([seq_src, seq_dst], dtype=torch.long),
            torch.tensor(seq_attr, dtype=torch.float),
        )
    else:
        seq_edge = (torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 1)))

    # Type-2 lane edges: same-lane ordering (directed, earlier -> later)
    lane_src, lane_dst, lane_attr = [], [], []
    from collections import defaultdict
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
    if lane_src:
        lane_edge = (
            torch.tensor([lane_src, lane_dst], dtype=torch.long),
            torch.tensor(lane_attr, dtype=torch.float),
        )
    else:
        lane_edge = (torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 1)))

    # Vehicle -> Operation edges (vehicle owns its ops)
    vo_src, vo_dst = [], []
    for i, op in enumerate(ops):
        vo_src.append(veh_idx[op.vehicle_id])
        vo_dst.append(i)
    owns_edge = (torch.tensor([vo_src, vo_dst], dtype=torch.long), None)

    # Zone -> Operation edges (zone hosts all ops that use it)
    zo_src, zo_dst = [], []
    for i, op in enumerate(ops):
        if op.zone_id in zone_idx:
            zo_src.append(zone_idx[op.zone_id])
            zo_dst.append(i)
    hosts_edge = (torch.tensor([zo_src, zo_dst], dtype=torch.long), None)

    return {"seq": seq_edge, "lane": lane_edge, "owns": owns_edge, "hosts": hosts_edge}
