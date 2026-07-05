"""Tests for feasibility checker invariants."""

import pytest

from intersection_scheduler.environment.feasibility import compute_feasible_set
from intersection_scheduler.environment.intersection import IntersectionEnv, Vehicle


def _two_vehicle_env(arr0: float = 0.0, arr1: float = 2.0):
    vehicles = [
        Vehicle(id=0, arrival_time=arr0, route=[2, 5, 8], processing_times=[1.5, 2.0, 1.5], velocity=10.0),
        Vehicle(id=1, arrival_time=arr1, route=[6, 5, 4], processing_times=[1.5, 2.0, 1.5], velocity=10.0),
    ]
    env = IntersectionEnv()
    env.reset(vehicles)
    return env, vehicles


class TestArrivalConstraint:
    def test_vehicle_not_arrived_ops_infeasible(self):
        env, _ = _two_vehicle_env(arr0=0.0, arr1=99.0)
        mask = compute_feasible_set(env)
        # Vehicle 1 hasn't arrived; none of its ops should be feasible
        v1_indices = [i for i, o in enumerate(env.operations) if o.vehicle_id == 1]
        for idx in v1_indices:
            assert not mask[idx].item(), f"Op {idx} of unarrived vehicle should be infeasible"


class TestPredecessorConstraint:
    def test_non_first_op_infeasible_if_predecessor_unscheduled(self):
        env, _ = _two_vehicle_env(arr0=0.0, arr1=0.0)
        mask = compute_feasible_set(env)
        # Second and third ops of each vehicle must be infeasible initially
        for vid in [0, 1]:
            later_ops = [
                (i, o) for i, o in enumerate(env.operations)
                if o.vehicle_id == vid and o.route_position > 0
            ]
            for idx, _ in later_ops:
                assert not mask[idx].item(), (
                    f"Op {idx} (vehicle {vid}, position > 0) should be infeasible before predecessor scheduled"
                )


class TestFeasibleSetNonEmpty:
    def test_feasible_set_never_empty_before_done(self):
        env, _ = _two_vehicle_env(arr0=0.0, arr1=0.0)
        done = False
        while not done:
            mask = compute_feasible_set(env)
            assert mask.any(), "Feasible set became empty before episode done"
            idx = int(mask.nonzero(as_tuple=False)[0].item())
            env, _, done = env.step(idx)


class TestDeadlockFree:
    def test_no_returned_op_causes_deadlock(self):
        """After masking, tentatively scheduling any returned feasible op
        must not create a cycle in the precedence graph."""
        from intersection_scheduler.environment.feasibility import (
            _build_base_adjacency,
            _would_cause_deadlock,
        )

        env, _ = _two_vehicle_env(arr0=0.0, arr1=0.0)
        mask = compute_feasible_set(env)
        adj = _build_base_adjacency(env)

        for idx, op in enumerate(env.operations):
            if mask[idx].item():
                # The op is claimed safe; verify deadlock check returns False
                assert not _would_cause_deadlock(idx, op, env, adj), (
                    f"Feasible op {idx} would cause deadlock — inconsistency"
                )
