"""Tests for HGT model and policy."""

import pytest
import torch

from intersection_scheduler.data.scenario_generator import ScenarioGenerator
from intersection_scheduler.environment.feasibility import compute_feasible_set
from intersection_scheduler.environment.graph_builder import build_hetero_graph
from intersection_scheduler.environment.intersection import IntersectionEnv, Vehicle
from intersection_scheduler.model.hgt import IntersectionHGT
from intersection_scheduler.model.policy import SchedulingPolicy
from intersection_scheduler.training.ppo import Transition, ppo_update


def _minimal_env():
    """2 vehicles, 4 distinct zones — minimal valid scenario."""
    vehicles = [
        Vehicle(id=0, arrival_time=0.0, route=[2, 5], processing_times=[1.5, 2.0], velocity=10.0),
        Vehicle(id=1, arrival_time=0.5, route=[6, 5], processing_times=[1.5, 2.0], velocity=10.0),
    ]
    env = IntersectionEnv()
    env.reset(vehicles)
    return env


class TestHGTForwardPass:
    def test_runs_without_error(self):
        env = _minimal_env()
        data = build_hetero_graph(env)
        model = IntersectionHGT(hidden_dim=32, num_heads=2, num_layers=2)
        out = model(data.x_dict, data.edge_index_dict)
        assert out is not None

    def test_output_shape(self):
        env = _minimal_env()
        data = build_hetero_graph(env)
        model = IntersectionHGT(hidden_dim=32, num_heads=2, num_layers=2)
        out = model(data.x_dict, data.edge_index_dict)
        n_ops = len(env.operations)
        assert out.shape == (n_ops, 32)


class TestPolicyMasking:
    def test_infeasible_ops_get_neg_inf_logits(self):
        env = _minimal_env()
        data = build_hetero_graph(env)
        policy = SchedulingPolicy(hidden_dim=32, num_heads=2, num_layers=2)
        mask = compute_feasible_set(env)

        dist, value = policy(data, mask)
        logits = dist.logits

        for idx in range(len(env.operations)):
            if not mask[idx].item():
                assert logits[idx].item() == float("-inf"), (
                    f"Op {idx} is infeasible but logit={logits[idx].item()}"
                )

    def test_value_scalar(self):
        env = _minimal_env()
        data = build_hetero_graph(env)
        policy = SchedulingPolicy(hidden_dim=32, num_heads=2, num_layers=2)
        mask = compute_feasible_set(env)
        _, value = policy(data, mask)
        assert value.shape == (1,) or value.numel() == 1


class TestPPONaNGradients:
    def test_no_nan_gradients_on_dummy_trajectory(self):
        env = _minimal_env()
        policy = SchedulingPolicy(hidden_dim=32, num_heads=2, num_layers=2)
        optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

        # Collect a short trajectory
        transitions = []
        done = False
        while not done:
            data = build_hetero_graph(env)
            mask = compute_feasible_set(env)
            if not mask.any():
                break
            dist, value = policy(data, mask)
            action = int(dist.sample().item())
            log_prob = dist.log_prob(torch.tensor(action))
            env, reward, done = env.step(action)
            transitions.append(Transition(
                data=data,
                feasible_mask=mask,
                action=action,
                log_prob=log_prob,
                value=value,
                reward=reward,
                done=done,
            ))

        assert len(transitions) > 0, "No transitions collected"

        ppo_update(policy, optimizer, transitions, epochs=1)

        for name, param in policy.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
