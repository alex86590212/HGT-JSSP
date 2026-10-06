"""Helpers shared by training and evaluation: config, policy construction,
scenario generation, and single-threaded worker processes."""

from __future__ import annotations

import io
from typing import Any, Dict, List

import torch
import yaml

from admission_rl.policy import AdmissionPolicy
from dynamic_scheduler.data.traffic_generator import TrafficGenerator
from dynamic_scheduler.environment.dynamic_intersection import DynamicVehicle


def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def make_policy(model_cfg: Dict[str, Any]) -> AdmissionPolicy:
    return AdmissionPolicy(
        hidden_dim=model_cfg["hidden_dim"],
        num_heads=model_cfg["num_heads"],
        num_layers=model_cfg["num_layers"],
        defer_init_bias=model_cfg.get("defer_init_bias", -2.0),
    )


def eval_scenarios(rate: float, seed: int, n_scenarios: int, episode_duration: float) -> List[DynamicVehicle]:
    """Same scenario stream as evaluation/eval_gapfree_admission.py, so HGT
    numbers are directly comparable with the existing baseline CSVs."""
    gen = TrafficGenerator(seed=seed * 100 + int(rate * 1000))
    return [gen.generate_episode_arrivals(episode_duration, arrival_rate=rate) for _ in range(n_scenarios)]


def pack(obj: Any) -> bytes:
    # Plain bytes through the process pool, not torch's shared-memory
    # reducer (which leaks file descriptors with many small tensors).
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()


def unpack(data: bytes) -> Any:
    # weights_only=False: local IPC of our own Transition/HeteroData objects.
    return torch.load(io.BytesIO(data), weights_only=False)  # noqa: S301


def worker_init() -> None:
    """Each rollout worker is one single-threaded process: the per-decision
    forwards are tiny, so intra-op threading only adds overhead."""
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
