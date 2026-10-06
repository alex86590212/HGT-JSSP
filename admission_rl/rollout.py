"""One RL episode on the gap-free admission model.

Mirrors dynamic_scheduler.environment.gapfree_admission.run_admission_episode
event for event (lock, detect, wipe-and-rerank tentative admissions, admit
until no candidate is feasible), with two differences:
  - the choice among feasible candidates comes from `decide`, and
  - if allow_defer, `decide` may return DEFER: admit nobody more at this
    event and re-check at t + defer_step. Consecutive defers are capped at
    max_hold seconds; after that the next decision must admit a vehicle.
With allow_defer=False and an iGreedy `decide`, the episode is identical to
run_admission_episode with igreedy_admission_select (see tests).

Reward: a vehicle's waiting time is charged once, when it becomes final --
on LOCK (irrevocable), or at episode end for still-tentative and
never-admitted vehicles. Tentative admissions are revoked and re-made on
every new detection, so charging at admission time would count a vehicle
several times. Each charge is -wait / n_vehicles * reward_scale, so an
episode's rewards sum to exactly -reward_scale * (mean waiting time), the
evaluation metric. Charges land on the most recent recorded decision.
Optionally the iGreedy episode metric is added back on the last step, so the
return is reward_scale * (W_igreedy - W_policy).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch_geometric.data import HeteroData

from admission_rl.graph import build_admission_graph
from admission_rl.policy import DEFER, AdmissionPolicy
from dynamic_scheduler.environment.dynamic_intersection import DynamicVehicle
from dynamic_scheduler.environment.gapfree_admission import AdmissionEnv

MAX_DECISIONS = 50_000


@dataclass
class Transition:
    data: HeteroData
    feasible_mask: torch.BoolTensor
    defer_allowed: bool
    action: int
    log_prob: float
    value: float
    reward: float = 0.0
    done: bool = False
    advantage: float = 0.0
    ret: float = 0.0


# decide(data, feasible_mask, defer_allowed, pending_vehicles, env)
#   -> (action, log_prob, value); action indexes pending_vehicles or is DEFER.
Decide = Callable[[HeteroData, torch.BoolTensor, bool, List[DynamicVehicle], AdmissionEnv],
                  Tuple[int, float, float]]


def policy_decide(policy: AdmissionPolicy, deterministic: bool = False) -> Decide:
    def decide(data, feasible_mask, defer_allowed, pending_vehicles, env):
        return policy.act(data, feasible_mask, defer_allowed, deterministic)
    return decide


def run_rl_episode(
    decide: Decide,
    arrivals: List[DynamicVehicle],
    episode_duration: float,
    detection_window: float,
    commit_window: float = 2.5,
    allow_defer: bool = True,
    defer_step: float = 0.5,
    max_hold: Optional[float] = None,
    reward_scale: float = 1.0,
    baseline_wait: Optional[float] = None,
    record: bool = True,
) -> Tuple[List[Transition], Dict[str, float]]:
    env = AdmissionEnv(detection_window=detection_window, commit_window=commit_window)
    env.reset(arrivals, episode_duration)

    by_id = {v.id: v for v in arrivals}
    arrivals_sorted = sorted(arrivals, key=lambda v: v.arrival_time)
    n_vehicles = len(arrivals)
    pending_ids: set = set()
    transitions: List[Transition] = []
    log_cursor = 0
    n_decisions = 0
    n_defers = 0
    # Start of the current run of consecutive defers; reset by an admission.
    # A run may last at most max_hold seconds, then someone must be admitted,
    # so a deterministic policy cannot defer forever.
    hold_start: Optional[float] = None

    def charge_finalized() -> None:
        nonlocal log_cursor
        new = env.completed_log[log_cursor:]
        log_cursor = len(env.completed_log)
        if new and transitions:
            transitions[-1].reward -= reward_scale * sum(c["waiting_time"] for c in new) / n_vehicles

    events: List[float] = []
    for v in arrivals_sorted:
        heapq.heappush(events, max(0.0, v.arrival_time - detection_window))
    heapq.heappush(events, episode_duration)

    processed: set = set()
    while events:
        t = heapq.heappop(events)
        if t in processed or t > episode_duration + 1e-9:
            continue
        processed.add(t)
        env.current_time = t

        env.lock_near()
        charge_finalized()

        newly_detected = False
        for v in arrivals_sorted:
            if v.id in env.admitted_ids or v.id in pending_ids:
                continue
            if v.arrival_time - t <= detection_window + 1e-9:
                pending_ids.add(v.id)
                newly_detected = True

        if newly_detected:
            for vid in list(env.tentative_ids):
                env.revoke_tentative(vid)
                pending_ids.add(vid)

        while True:
            pending_order = sorted(pending_ids, key=lambda vid: (by_id[vid].arrival_time, vid))
            status = {}
            n_feasible = 0
            for vid in pending_order:
                feasible, retry = env.admission_check(by_id[vid])
                status[vid] = (feasible, retry)
                if feasible:
                    n_feasible += 1
                elif retry is not None and retry <= episode_duration + 1e-9:
                    heapq.heappush(events, retry)
            if n_feasible == 0:
                break

            defer_allowed = (
                allow_defer
                and t + defer_step <= episode_duration + 1e-9
                and (max_hold is None or hold_start is None
                     or t + defer_step - hold_start <= max_hold + 1e-9)
            )
            if n_feasible == 1 and not defer_allowed:
                chosen_id = next(vid for vid in pending_order if status[vid][0])
            else:
                n_decisions += 1
                if n_decisions > MAX_DECISIONS:
                    raise RuntimeError(f"run_rl_episode exceeded {MAX_DECISIONS} decisions")
                data, mask = build_admission_graph(env, pending_order, status)
                pending_vehicles = [by_id[vid] for vid in pending_order]
                action, log_prob, value = decide(data, mask, defer_allowed, pending_vehicles, env)
                if record:
                    transitions.append(Transition(data, mask, defer_allowed, action, log_prob, value))
                if action == DEFER:
                    n_defers += 1
                    if hold_start is None:
                        hold_start = t
                    heapq.heappush(events, t + defer_step)
                    break
                chosen_id = pending_order[action]
                assert status[chosen_id][0], "decide() chose an infeasible vehicle"
            hold_start = None
            env.admit(by_id[chosen_id])
            pending_ids.discard(chosen_id)

        env.lock_near()
        charge_finalized()

    for vid in list(env.tentative_ids):
        env.finalize_tentative(vid)
    for v in arrivals_sorted:
        if v.id not in env.admitted_ids:
            env.completed_log.append({
                "vehicle_id": v.id,
                "arrival_time": v.arrival_time,
                "finish_time": None,
                "waiting_time": max(0.0, episode_duration - v.arrival_time),
            })
    charge_finalized()

    n_logged = len(env.completed_log)
    wait = sum(c["waiting_time"] for c in env.completed_log) / n_logged if n_logged else 0.0
    if transitions:
        if baseline_wait is not None:
            transitions[-1].reward += reward_scale * baseline_wait
        transitions[-1].done = True

    stats = {
        "waiting_time": wait,
        "completion_rate": len(env.admitted_ids) / n_vehicles if n_vehicles else 1.0,
        "n_vehicles": n_vehicles,
        "n_decisions": n_decisions,
        "n_defers": n_defers,
        "total_reward": sum(tr.reward for tr in transitions),
    }
    return transitions, stats
