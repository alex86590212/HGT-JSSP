import pytest
import torch
from torch_geometric.data import Batch

from admission_rl.graph import build_admission_graph
from admission_rl.policy import DEFER, AdmissionPolicy
from admission_rl.rollout import policy_decide, run_rl_episode
from admission_rl.train import main as train_main
from dynamic_scheduler.data.traffic_generator import TrafficGenerator
from dynamic_scheduler.environment.gapfree_admission import (
    AdmissionEnv,
    igreedy_admission_select,
    run_admission_episode,
)

T, DW = 60.0, 10.0


def _arrivals(rate, seed):
    return TrafficGenerator(seed=seed).generate_episode_arrivals(T, arrival_rate=rate)


def igreedy_decide(data, mask, defer_allowed, pending, env):
    feasible = [i for i, ok in enumerate(mask.tolist()) if ok]
    i = min(feasible, key=lambda i: (env._effective_start(pending[i])[0], pending[i].arrival_time, pending[i].id))
    return i, 0.0, 0.0


@pytest.mark.parametrize("rate", [0.1, 0.5, 1.0, 1.2])
@pytest.mark.parametrize("seed", [0, 1])
def test_order_only_igreedy_matches_reference(rate, seed):
    arrivals = _arrivals(rate, seed)
    ref, _, _ = run_admission_episode(arrivals, T, DW, igreedy_admission_select)
    _, stats = run_rl_episode(igreedy_decide, arrivals, T, DW, allow_defer=False)
    assert stats["waiting_time"] == pytest.approx(ref, abs=1e-9)


@pytest.mark.parametrize("allow_defer", [False, True])
def test_rewards_sum_to_scaled_wait_minus_baseline(allow_defer):
    torch.manual_seed(0)
    policy = AdmissionPolicy(hidden_dim=32, num_heads=2, num_layers=1, defer_init_bias=0.0)
    arrivals = _arrivals(0.8, 3)
    trs, stats = run_rl_episode(policy_decide(policy), arrivals, T, DW, allow_defer=allow_defer,
                                reward_scale=10.0, baseline_wait=0.7)
    assert trs and trs[-1].done and not any(tr.done for tr in trs[:-1])
    assert sum(tr.reward for tr in trs) == pytest.approx(-10.0 * (stats["waiting_time"] - 0.7), abs=1e-6)
    if allow_defer:
        assert stats["n_defers"] > 0


def test_max_hold_forces_admission(monkeypatch):
    """A policy that always defers must still admit within max_hold seconds."""
    log = []  # ("defer" | "admit", time), in order
    real_admit = AdmissionEnv.admit

    def admit(self, v):
        log.append(("admit", self.current_time))
        return real_admit(self, v)

    monkeypatch.setattr(AdmissionEnv, "admit", admit)

    def always_defer(data, mask, defer_allowed, pending, env):
        if defer_allowed:
            log.append(("defer", env.current_time))
            return DEFER, 0.0, 0.0
        return igreedy_decide(data, mask, defer_allowed, pending, env)

    _, stats = run_rl_episode(always_defer, _arrivals(1.2, 0), T, DW, defer_step=0.5, max_hold=2.0)
    assert stats["n_defers"] > 0 and stats["completion_rate"] > 0.5
    hold_start = None
    for kind, t in log:
        if kind == "admit":
            hold_start = None
        else:
            hold_start = t if hold_start is None else hold_start
            assert t + 0.5 - hold_start <= 2.0 + 1e-9


def _graphs(n=4):
    """A few real decision states, including one with no committed vehicles."""
    out = []
    env = AdmissionEnv(detection_window=DW)
    arrivals = _arrivals(1.0, 5)
    env.reset(arrivals, T)
    env.current_time = 0.0
    pending = sorted([v for v in arrivals if v.arrival_time <= DW], key=lambda v: (v.arrival_time, v.id))
    while pending and len(out) < n:
        order = [v.id for v in pending]
        status = {v.id: env.admission_check(v) for v in pending}
        data, mask = build_admission_graph(env, order, status)
        out.append((data, mask))
        feasible = [v for v in pending if status[v.id][0]]
        if not feasible:
            break
        env.admit(feasible[0])
        pending.remove(feasible[0])
    return out


def test_policy_batch_matches_single_and_masks():
    torch.manual_seed(0)
    policy = AdmissionPolicy(hidden_dim=32, num_heads=2, num_layers=2).eval()
    graphs = _graphs()
    assert graphs[0][0]["committed"].x.shape[0] == 0
    assert len(graphs) >= 2
    defer_allowed = torch.tensor([i % 2 == 0 for i in range(len(graphs))])
    batch = Batch.from_data_list([g for g, _ in graphs])
    logits, values = policy(batch, torch.cat([m for _, m in graphs]), defer_allowed)
    for i, (data, mask) in enumerate(graphs):
        single, v1 = policy(Batch.from_data_list([data]), mask, defer_allowed[i:i + 1])
        n = mask.shape[0]
        assert torch.allclose(logits[i, :n], single[0, :n], atol=1e-5)
        assert torch.allclose(logits[i, -1], single[0, -1], atol=1e-5)
        assert torch.allclose(values[i], v1[0], atol=1e-5)
        assert torch.isinf(logits[i, :n][~mask]).all()
        assert torch.isinf(logits[i, n:-1]).all()
        assert torch.isinf(logits[i, -1]) != bool(defer_allowed[i])
        for _ in range(20):
            a, _, _ = policy.act(data, mask, bool(defer_allowed[i]))
            assert (a == DEFER and defer_allowed[i]) or (a != DEFER and mask[a])


def test_training_smoke(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("""
seed: 0
env: {episode_duration: 20.0, detection_window: 10.0, commit_window: 2.5, allow_defer: true, defer_step: 0.5, max_hold: 2.0}
model: {hidden_dim: 32, num_heads: 2, num_layers: 1, defer_init_bias: -2.0}
train: {num_episodes: 8, rate_min: 0.3, rate_max: 0.8, rollout_episodes: 4, num_workers: 2, torch_threads: 1,
        reward_scale: 10.0, igreedy_baseline: true, log_interval: 4, eval_interval: 8, checkpoint_interval: 4}
ppo: {lr: 3.0e-4, gamma: 1.0, gae_lambda: 0.95, clip_epsilon: 0.2, epochs: 2, mini_batch_size: 64,
      value_loss_coef: 0.5, entropy_coef: 0.01, entropy_coef_final: 0.001, max_grad_norm: 0.5}
eval: {rates: [0.3, 0.8], seeds: [0], n_scenarios: 2}
""")
    out = tmp_path / "run"
    train_main(["--config", str(cfg), "--output", str(out)])
    for name in ("checkpoint_last.pt", "checkpoint_best.pt", "train_log.csv", "eval_log.csv"):
        assert (out / name).exists(), name
    train_main(["--resume", str(out / "checkpoint_last.pt"), "--output", str(out), "--episodes", "12"])
    assert torch.load(out / "checkpoint_last.pt", weights_only=False)["episode"] == 12
