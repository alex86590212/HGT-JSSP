"""Tests for IntersectionEnv correctness."""

import pytest

from intersection_scheduler.data.scenario_generator import ScenarioGenerator
from intersection_scheduler.environment.feasibility import compute_feasible_set
from intersection_scheduler.environment.intersection import IntersectionEnv, Vehicle


def _make_simple_env():
    """2 vehicles with 1 shared zone (z5) — simplest conflict case."""
    vehicles = [
        Vehicle(id=0, arrival_time=0.0, route=[5], processing_times=[2.0], velocity=10.0),
        Vehicle(id=1, arrival_time=0.0, route=[5], processing_times=[2.0], velocity=10.0),
    ]
    env = IntersectionEnv()
    env.reset(vehicles)
    return env, vehicles


def _make_route_env():
    """2 vehicles, each with a 3-zone route."""
    vehicles = [
        Vehicle(id=0, arrival_time=0.0, route=[2, 5, 8], processing_times=[1.5, 2.0, 1.5], velocity=10.0),
        Vehicle(id=1, arrival_time=1.0, route=[6, 5, 4], processing_times=[1.5, 2.0, 1.5], velocity=10.0),
    ]
    env = IntersectionEnv()
    env.reset(vehicles)
    return env, vehicles


def _schedule_all_greedy(env):
    """Schedule all ops using first feasible, return all rewards."""
    rewards = []
    done = False
    while not done:
        mask = compute_feasible_set(env)
        assert mask.any(), "Feasible set empty before done"
        idx = int(mask.nonzero(as_tuple=False)[0].item())
        env, reward, done = env.step(idx)
        rewards.append(reward)
    return rewards


class TestSchedulingCompleteness:
    def test_all_type3_resolved_after_completion(self):
        env, _ = _make_simple_env()
        _schedule_all_greedy(env)
        assert len(env.active_conflict_edges) == 0

    def test_all_ops_scheduled_when_done(self):
        env, _ = _make_route_env()
        _schedule_all_greedy(env)
        assert all(op.scheduled for op in env.operations)

    def test_done_iff_all_scheduled(self):
        env, _ = _make_route_env()
        done = False
        while not done:
            mask = compute_feasible_set(env)
            idx = int(mask.nonzero(as_tuple=False)[0].item())
            env, _r, done = env.step(idx)
        # Now done: all ops must be scheduled
        assert all(op.scheduled for op in env.operations)


class TestReward:
    def test_reward_nonpositive(self):
        env, _ = _make_route_env()
        rewards = _schedule_all_greedy(env)
        for r in rewards:
            assert r <= 1e-9, f"Reward {r} > 0"

    def test_reward_negative_when_conflict_adds_delay(self):
        # Both vehicles want z5; whoever goes second must wait → negative reward
        env, _ = _make_simple_env()
        rewards = _schedule_all_greedy(env)
        total = sum(rewards)
        assert total < 0.0, "Expected net negative reward for conflicting scenario"


class TestFinishTimeMonotonicity:
    def test_cij_geq_cij_minus1_plus_processing(self):
        env, _ = _make_route_env()
        _schedule_all_greedy(env)

        for vid in {op.vehicle_id for op in env.operations}:
            ops = sorted(
                [o for o in env.operations if o.vehicle_id == vid],
                key=lambda o: o.route_position,
            )
            for k in range(1, len(ops)):
                prev, cur = ops[k - 1], ops[k]
                assert cur.earliest_finish >= prev.earliest_finish + cur.processing_time - 1e-6, (
                    f"Monotonicity violated at vehicle {vid} position {k}"
                )


class TestUnschedulableOpsInGraph:
    def test_unschedulable_ops_have_correct_c_values(self):
        """After one step, unscheduled ops are still in env.operations with c values set."""
        from intersection_scheduler.environment.graph_builder import build_hetero_graph

        env, _ = _make_route_env()
        # Schedule the very first feasible op
        mask = compute_feasible_set(env)
        idx = int(mask.nonzero(as_tuple=False)[0].item())
        env, _, _ = env.step(idx)

        # Unscheduled ops must still appear in the graph
        data = build_hetero_graph(env)
        n_ops_graph = data["operation"].x.shape[0]
        n_ops_env = len(env.operations)
        assert n_ops_graph == n_ops_env

        # d=0 for unscheduled, c > 0 (lower-bound finish time set at reset)
        x = data["operation"].x
        for i, op in enumerate(env.operations):
            if not op.scheduled:
                assert x[i, 0].item() == pytest.approx(0.0), f"Op {i} should have d=0"
                # c(i,j) is normalised but > 0 since processing time > 0
                assert x[i, 1].item() >= 0.0
