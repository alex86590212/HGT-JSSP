"""Equivalence tests: optimized graph building must produce byte-identical output.

The optimizations under test (threading a precomputed feasible mask, and caching
the static per-episode edge topology) must NOT change any tensor the model sees.
These tests capture the current build_hetero_graph output at every step of full
episodes across both the 3x3 and 4x4 topologies and assert the optimized paths
produce exactly equal tensors (torch.equal, not allclose).
"""

import torch

from intersection_scheduler.data.scenario_generator import ScenarioGenerator as Gen3x3
from intersection_scheduler.data.scenario_generator_4x4 import (
    ScenarioGenerator as Gen4x4,
    ZONE_POSITIONS as ZP_4x4,
)
from intersection_scheduler.environment.feasibility import (
    compute_feasible_set,
    next_feasible_time,
)
from intersection_scheduler.environment.graph_builder import (
    build_hetero_graph,
    build_static_edges,
)
from intersection_scheduler.environment.intersection import IntersectionEnv


def _assert_hetero_equal(a, b):
    """Assert two HeteroData objects are byte-for-byte identical."""
    # Node features
    assert set(a.x_dict.keys()) == set(b.x_dict.keys())
    for ntype in a.x_dict:
        assert torch.equal(a.x_dict[ntype], b.x_dict[ntype]), f"node feats differ: {ntype}"
    # Edge indices and attrs
    a_edges = set(a.edge_index_dict.keys())
    b_edges = set(b.edge_index_dict.keys())
    assert a_edges == b_edges, f"edge types differ: {a_edges} vs {b_edges}"
    for etype in a.edge_index_dict:
        assert torch.equal(a.edge_index_dict[etype], b.edge_index_dict[etype]), \
            f"edge_index differs: {etype}"
        a_store = a[etype]
        b_store = b[etype]
        if hasattr(a_store, "edge_attr") or hasattr(b_store, "edge_attr"):
            a_attr = getattr(a_store, "edge_attr", None)
            b_attr = getattr(b_store, "edge_attr", None)
            assert (a_attr is None) == (b_attr is None), f"edge_attr presence differs: {etype}"
            if a_attr is not None:
                assert torch.equal(a_attr, b_attr), f"edge_attr differs: {etype}"


def _run_equivalence_over_episode(env, scenario, zone_positions):
    """Step through a full episode, asserting optimized == baseline at each step."""
    env.reset(scenario.vehicles, zone_positions=zone_positions)
    static_edges = build_static_edges(env)

    steps = 0
    done = False
    while not done:
        mask = compute_feasible_set(env)

        # Baseline: recompute everything from scratch.
        baseline = build_hetero_graph(env)

        # Optimized path 1: thread the precomputed mask.
        opt_mask = build_hetero_graph(env, feasible_mask=mask)
        _assert_hetero_equal(baseline, opt_mask)

        # Optimized path 2: thread mask AND cached static edges.
        opt_full = build_hetero_graph(env, feasible_mask=mask, static_edges=static_edges)
        _assert_hetero_equal(baseline, opt_full)

        if not mask.any():
            nt = next_feasible_time(env)
            if nt is None:
                break
            env.current_time = nt
            continue

        # Deterministic action choice (iGreedy-style) to advance the episode.
        best_idx, best_key = None, None
        for idx, op in enumerate(env.operations):
            if not mask[idx].item():
                continue
            v = next(v for v in env.vehicles if v.id == op.vehicle_id)
            key = (v.arrival_time, op.route_position, op.vehicle_id)
            if best_key is None or key < best_key:
                best_key, best_idx = key, idx
        env, _, done = env.step(best_idx)
        steps += 1

    assert steps > 0


def test_equivalence_3x3_all_tiers():
    gen = Gen3x3(seed=123)
    env = IntersectionEnv()
    for tier in ("easy", "medium", "hard"):
        for _ in range(10):
            scenario = getattr(gen, tier)()
            _run_equivalence_over_episode(env, scenario, zone_positions=None)


def test_equivalence_4x4_all_tiers():
    gen = Gen4x4(seed=123)
    env = IntersectionEnv()
    for tier in ("easy", "medium", "hard"):
        for _ in range(10):
            scenario = getattr(gen, tier)()
            _run_equivalence_over_episode(env, scenario, zone_positions=ZP_4x4)
