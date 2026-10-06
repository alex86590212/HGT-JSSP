"""Heterogeneous graph of the admission state at one decision point.

Node types:
  pending   -- detected, not-yet-admitted vehicles (the action candidates;
               only the currently-feasible ones are selectable)
  committed -- TENTATIVE/LOCKED vehicles whose reservation is not yet over
  zone      -- the 16 conflict zones of the 4x4 intersection

All time features are relative to env.current_time and divided by
TIME_SCALE so they stay O(1) across arrival rates.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch_geometric.data import HeteroData

from dynamic_scheduler.data.traffic_generator import ROUTES
from dynamic_scheduler.environment.gapfree_admission import AdmissionEnv

TIME_SCALE = 10.0
ZONES = list(range(1, 17))
ZONE_INDEX = {z: i for i, z in enumerate(ZONES)}
MANOEUVRES = list(ROUTES.keys())
MANOEUVRE_INDEX = {m: i for i, m in enumerate(MANOEUVRES)}

NODE_TYPES = ["pending", "committed", "zone"]
EDGE_TYPES = [
    ("zone", "on_route", "pending"),
    ("pending", "uses", "zone"),
    ("zone", "on_route_c", "committed"),
    ("committed", "holds", "zone"),
    ("pending", "lane", "pending"),
    ("pending", "conflict", "pending"),
    ("committed", "blocks", "pending"),
]
METADATA = (NODE_TYPES, EDGE_TYPES)

_N_GLOBAL = 4
NODE_DIMS = {
    "pending": 10 + len(MANOEUVRES) + _N_GLOBAL,
    "committed": 5 + len(MANOEUVRES),
    "zone": 5 + len(ZONES),
}

# (feasible, retry_time) per pending vehicle id, as returned by
# AdmissionEnv.admission_check.
Status = Dict[int, Tuple[bool, Optional[float]]]


def _one_hot(index: int, size: int) -> List[float]:
    v = [0.0] * size
    v[index] = 1.0
    return v


def _edges(pairs: List[Tuple[int, int]]) -> torch.Tensor:
    if not pairs:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(pairs, dtype=torch.long).t().contiguous()


def build_admission_graph(
    env: AdmissionEnv,
    pending_order: List[int],
    status: Status,
) -> Tuple[HeteroData, torch.BoolTensor]:
    """Returns (graph, feasible_mask). feasible_mask[i] is True iff
    pending_order[i] can be admitted right now."""
    t = env.current_time
    vehicles = env._vehicles_by_id
    pending_set = set(pending_order)
    pending_idx = {vid: i for i, vid in enumerate(pending_order)}

    committed_order = []
    for vid in sorted(env.admitted_ids):
        v = vehicles[vid]
        if env.committed_s1[vid] + sum(v.processing_times) > t + 1e-9:
            committed_order.append(vid)
    committed_idx = {vid: i for i, vid in enumerate(committed_order)}

    global_feats = [
        t / max(env.episode_duration, 1e-9),
        len(pending_order) / 20.0,
        len(env.tentative_ids) / 20.0,
        len(committed_order) / 20.0,
    ]

    # --- pending nodes -------------------------------------------------
    pending_x, feasible = [], []
    blocks_pairs, lane_pairs = [], []
    zone_wanted = [0] * len(ZONES)
    for vid in pending_order:
        v = vehicles[vid]
        ok, retry = status[vid]
        s1, leaders_ready = env._effective_start(v)
        leaders = env._same_lane_leaders(v)
        pending_leaders = [l.id for l in leaders if l.id in pending_set]
        for lid in pending_leaders:
            lane_pairs.append((pending_idx[lid], pending_idx[vid]))
        n_blockers = 0
        if leaders_ready:
            _, blockers = env.blockers_for(v)
            n_blockers = len(blockers)
            for bid in blockers:
                if bid in committed_idx:
                    blocks_pairs.append((committed_idx[bid], pending_idx[vid]))
        for z in set(v.route):
            zone_wanted[ZONE_INDEX[z]] += 1
        pending_x.append([
            1.0 if ok else 0.0,
            1.0 if leaders_ready else 0.0,
            (s1 - t) / TIME_SCALE if leaders_ready else 0.0,
            (v.arrival_time - t) / env.detection_window,
            max(0.0, t - v.arrival_time) / TIME_SCALE,
            (retry - t) / TIME_SCALE if (not ok and retry is not None) else 0.0,
            len(pending_leaders) / 5.0,
            len(v.route) / 5.0,
            sum(v.processing_times) / TIME_SCALE,
            n_blockers / 5.0,
            *_one_hot(MANOEUVRE_INDEX[v.manoeuvre], len(MANOEUVRES)),
            *global_feats,
        ])
        feasible.append(ok)

    conflict_pairs = []
    for i, a in enumerate(pending_order):
        za = set(vehicles[a].route)
        for j in range(i + 1, len(pending_order)):
            if za & set(vehicles[pending_order[j]].route):
                conflict_pairs.append((i, j))
                conflict_pairs.append((j, i))

    # --- committed nodes -----------------------------------------------
    committed_x = []
    for vid in committed_order:
        v = vehicles[vid]
        s1 = env.committed_s1[vid]
        committed_x.append([
            1.0 if vid in env.locked_ids else 0.0,
            (s1 - t) / TIME_SCALE,
            (s1 + sum(v.processing_times) - t) / TIME_SCALE,
            (v.arrival_time - t) / TIME_SCALE,
            len(v.route) / 5.0,
            *_one_hot(MANOEUVRE_INDEX[v.manoeuvre], len(MANOEUVRES)),
        ])

    # --- zone nodes ----------------------------------------------------
    zone_x = []
    horizon = TIME_SCALE
    for z in ZONES:
        occ = env.zones.get(z)
        future = [(s, e) for s, e, _ in (occ.intervals if occ else []) if e > t + 1e-9]
        busy_until = max((e for _, e in future), default=t)
        occupied_now = any(s <= t + 1e-9 for s, _ in future)
        load = sum(max(0.0, min(e, t + horizon) - max(s, t)) for s, e in future) / horizon
        zone_x.append([
            (busy_until - t) / TIME_SCALE,
            1.0 if occupied_now else 0.0,
            load,
            len(future) / 5.0,
            zone_wanted[ZONE_INDEX[z]] / 10.0,
            *_one_hot(ZONE_INDEX[z], len(ZONES)),
        ])

    route_p, route_c = [], []
    for vid in pending_order:
        for z in set(vehicles[vid].route):
            route_p.append((ZONE_INDEX[z], pending_idx[vid]))
    for vid in committed_order:
        for z in set(vehicles[vid].route):
            route_c.append((ZONE_INDEX[z], committed_idx[vid]))

    data = HeteroData()
    data["pending"].x = torch.tensor(pending_x, dtype=torch.float).view(-1, NODE_DIMS["pending"])
    data["committed"].x = torch.tensor(committed_x, dtype=torch.float).view(-1, NODE_DIMS["committed"])
    data["zone"].x = torch.tensor(zone_x, dtype=torch.float).view(-1, NODE_DIMS["zone"])

    zp = _edges(route_p)
    zc = _edges(route_c)
    data["zone", "on_route", "pending"].edge_index = zp
    data["pending", "uses", "zone"].edge_index = zp.flip(0)
    data["zone", "on_route_c", "committed"].edge_index = zc
    data["committed", "holds", "zone"].edge_index = zc.flip(0)
    data["pending", "lane", "pending"].edge_index = _edges(lane_pairs)
    data["pending", "conflict", "pending"].edge_index = _edges(conflict_pairs)
    data["committed", "blocks", "pending"].edge_index = _edges(blocks_pairs)

    return data, torch.tensor(feasible, dtype=torch.bool)
