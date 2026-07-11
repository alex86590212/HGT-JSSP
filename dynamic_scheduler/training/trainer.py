"""Episode loop and training orchestration for the online dynamic 4x4 scheduler.

Reuses intersection_scheduler.training.ppo.{Transition, compute_gae, ppo_update}
unchanged — PPO/GAE only need (data, feasible_mask, action, log_prob, value,
reward, done) per transition, which this loop produces the same way the
offline run_episode does, just driven by wall-clock replanning events instead
of a fixed action-per-operation sequence.
"""

from __future__ import annotations

import copy
import random
import resource
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.multiprocessing as torch_mp
from torch.utils.tensorboard import SummaryWriter

# Default Linux tensor-sharing strategy ("file_descriptor") ships tensors
# between processes via /dev/shm-backed shared memory — this applies to ANY
# multiprocessing IPC carrying tensors, including plain
# concurrent.futures.ProcessPoolExecutor, not just torch.multiprocessing.Pool.
# HPC/SLURM nodes commonly cap /dev/shm per job (or count it against the
# job's cgroup memory limit) to something far smaller than --mem; each
# rollout episode returns 1000+ small tensors (log_prob/value + each
# HeteroData graph's tensors), so with num_workers>1 this exhausts /dev/shm
# almost immediately (observed: "unable to mmap ... Cannot allocate memory"
# -> BrokenProcessPool). "file_system" routes tensor IPC through regular temp
# files instead, avoiding /dev/shm entirely. Must be set before any tensor
# is shared across a process boundary, so this runs at import time.
torch_mp.set_sharing_strategy("file_system")

from dynamic_scheduler.data.traffic_generator import TrafficGenerator, get_curriculum_arrivals
from dynamic_scheduler.environment.dynamic_intersection import DynamicIntersectionEnv
from dynamic_scheduler.environment.feasibility import compute_feasible_set
from dynamic_scheduler.environment.graph_builder import build_hetero_graph
from dynamic_scheduler.utils.metrics import (
    completion_rate,
    episode_makespan,
    episode_waiting_time,
    episode_waiting_time_all,
)
from intersection_scheduler.data.scenario_generator_4x4 import ZONE_POSITIONS
from intersection_scheduler.model.policy import SchedulingPolicy
from intersection_scheduler.training.ppo import Transition, compute_gae, ppo_update


class InfiniteLoopError(RuntimeError):
    """Raised when run_episode's replan pass or outer event loop exceeds its
    hard iteration cap without converging — indicates a real bug (e.g. an
    operation state that never reaches a terminal condition), not a slow but
    valid episode. Fail loudly rather than hang or silently truncate."""


# Hard caps: generous enough for any real scenario (a replan pass visits each
# currently-open op at most once, so this cap is really "how many ops could
# possibly be open at once"; the outer loop cap is "how many distinct events
# could occur in one episode"), but finite so a real non-convergence bug
# raises immediately instead of hanging the process.
MAX_REPLAN_ITERS = 500
MAX_OUTER_ITERS = 5000


def run_episode(
    policy: SchedulingPolicy,
    env: DynamicIntersectionEnv,
    arrivals,
    episode_duration: float,
    *,
    deterministic: bool = False,
) -> "tuple[List[Transition], Dict[str, float]]":
    """Run one wall-clock episode of online scheduling.

    At each event (initial detections at t=0, then every subsequent
    detection event returned by env.advance_time), replan every currently
    open (non-LOCKED) operation AT MOST ONCE per pass: build the graph,
    compute the feasible mask (excluding ops already visited this pass),
    let the policy choose one operation to (re)plan, apply it, repeat until
    every open op has been visited once or none remain plannable. Then
    advance time to the next event and repeat until the episode ends.

    Raises InfiniteLoopError if either loop exceeds its hard iteration cap —
    this indicates a bug (e.g. env.advance_time not making progress), not a
    legitimately long episode.
    """
    device = next(policy.parameters()).device
    env.reset(arrivals, episode_duration)
    transitions: List[Transition] = []

    n_vehicles_seen = len(env.vehicles)
    # At t=0 every already-detected vehicle is a "trigger" (nothing is planned
    # yet, so all their ops must be planned this first pass).
    trigger_vids = list(env.vehicles.keys())

    done = False
    outer_iters = 0
    while not done:
        outer_iters += 1
        if outer_iters > MAX_OUTER_ITERS:
            raise InfiniteLoopError(
                f"run_episode outer event loop exceeded {MAX_OUTER_ITERS} iterations "
                f"without reaching episode_duration={episode_duration} "
                f"(current_time={env.current_time}). Likely env.advance_time() is not "
                f"making forward progress."
            )

        # Replan pass: only (re)plan operations AFFECTED by this event —
        # UNSCHEDULED ops (need a first plan) plus TENTATIVE ops sharing a zone
        # with a newly-detected vehicle (their timing could have changed).
        # Unaffected tentative plans are left untouched (no churn, no penalty).
        # Each affected op is visited at most once (a TENTATIVE op stays
        # feasible after planning, so we must exclude already-visited ones or
        # the pass never terminates).
        affected = set(env.affected_op_indices(trigger_vids))
        visited_this_pass: set = set()
        replan_iters = 0
        while True:
            replan_iters += 1
            if replan_iters > MAX_REPLAN_ITERS:
                raise InfiniteLoopError(
                    f"run_episode replan pass exceeded {MAX_REPLAN_ITERS} iterations "
                    f"at current_time={env.current_time} with {len(env.operations)} "
                    f"operations open. Likely compute_feasible_set is not converging."
                )

            mask = compute_feasible_set(env)
            # Restrict to affected, not-yet-visited ops.
            for i in range(len(mask)):
                if i not in affected or i in visited_this_pass:
                    mask[i] = False
            if not mask.any():
                break

            data_cpu = build_hetero_graph(env, feasible_mask=mask)  # keep on CPU

            # Rollout ALWAYS runs under no_grad: ppo_update recomputes fresh
            # log-probs/values (and their gradients) from the stored `data` in
            # its own forward pass, and only ever uses the rollout-time
            # log_prob/value as detached constants (old_log_prob baseline,
            # GAE value). Retaining gradient graphs here would pin one full
            # autograd graph per transition on the GPU — harmless offline
            # (~40 transitions/episode) but fatal at the dynamic hard tier
            # (3000+ transitions x 8 buffered episodes => OOM on a 32GB GPU).
            #
            # The stored graph is kept on CPU; ppo_update moves each mini-batch
            # to the GPU itself (Batch.from_data_list(...).to(device)), so the
            # rollout buffer never holds thousands of graphs in GPU memory at
            # once — only the single graph being forwarded here is on-device.
            with torch.no_grad():
                dist, value = policy(data_cpu.to(device), mask.to(device))

            if deterministic:
                action = int(dist.probs.argmax().item())
            else:
                action = int(dist.sample().item())

            log_prob = dist.log_prob(torch.tensor(action, device=device))

            env, reward, _ep_done = env.plan_operation(action)
            visited_this_pass.add(action)

            transitions.append(Transition(
                data=data_cpu,
                feasible_mask=mask,  # CPU
                action=action,
                log_prob=log_prob.detach().cpu(),
                value=value.detach().cpu(),
                reward=reward,
                done=False,  # done is only True on the final transition below
            ))

        done, newly_detected = env.advance_time()
        n_vehicles_seen += len(newly_detected)
        # Next pass only re-plans ops affected by these new detections.
        trigger_vids = newly_detected

    if transitions:
        transitions[-1].done = True

    inflight = env.inflight_waiting_times()
    n_completed = len(env.completed_log)
    stats = {
        "n_vehicles_seen": n_vehicles_seen,
        "n_vehicles_completed": n_completed,
        # Unbiased metric (completed + in-flight) — the one to report/select on.
        "waiting_time": episode_waiting_time_all(env.completed_log, inflight),
        # Completed-only, kept for comparison / to see the survivorship gap.
        "waiting_time_completed": episode_waiting_time(env.completed_log),
        "completion_rate": completion_rate(n_completed, n_vehicles_seen),
        "makespan": episode_makespan(env.completed_log),
        "steps": len(transitions),
        "total_reward": sum(t.reward for t in transitions),
        "final_n_active_vehicles": len(env.vehicles),
        "outer_iters": outer_iters,
    }
    return transitions, stats


def _worker_init() -> None:
    """ProcessPoolExecutor initializer, run once per worker process.

    Rollout episodes are ~1000+ tiny sequential HGT forwards on a small
    graph (~188 nodes) — intra-op multithreading was measured to make this
    SLOWER (thread-pool coordination overhead exceeds the tiny compute per
    call), confirmed on cluster with matched OMP_NUM_THREADS/--cpus-per-task.
    Each worker process must stay single-threaded internally; parallelism
    comes from running N whole episodes in N separate processes instead.
    set_num_interop_threads can only be called once per process (raises on
    a second call), so this must live in the pool initializer, not in the
    per-task worker function below (which the pool re-invokes every round).
    """
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # already set / interop pool already started — harmless


def _rollout_worker(
    state_dict: Dict[str, torch.Tensor],
    model_kwargs: Dict[str, Any],
    env_kwargs: Dict[str, Any],
    episode_duration: float,
    arrivals,
    seed: int,
) -> "tuple[List[Transition], Dict[str, float]]":
    """Run one full training episode in a worker process.

    Rebuilds a fresh policy/env from plain (picklable) kwargs rather than
    receiving the live objects, so this has no dependency on anything the
    main process holds open (optimizer, SummaryWriter, etc.). `seed` is
    derived from the episode number (not worker identity) so results don't
    depend on which physical process happens to draw which episode, and so
    a persistent pool (workers reused across rounds) doesn't replay
    correlated RNG state inherited from fork.
    """
    torch.manual_seed(seed)
    policy = SchedulingPolicy(**model_kwargs)
    policy.load_state_dict(state_dict)
    env = DynamicIntersectionEnv(**env_kwargs)
    return run_episode(policy, env, arrivals, episode_duration, deterministic=False)


def train(cfg: Dict[str, Any], output_dir: str = "results_dynamic", resume: Optional[str] = None) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(out / "tb"))

    ppo_cfg = cfg.get("ppo", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})
    env_cfg = cfg.get("environment", {})

    # Device: default CPU for the dynamic scheduler. The online rollout does
    # ~1200 tiny, SEQUENTIAL forward passes per episode (each planning decision
    # depends on the previous one's env state, so they can't be batched during
    # rollout). At ~188 nodes/graph, GPU kernel-launch + host<->device transfer
    # overhead per call dominates and runs ~10x SLOWER than CPU (observed: 27%
    # GPU utilisation, episodes ~36s on V100 vs ~5s on CPU). Set
    # training.device: cuda in the config only if you have a specific reason.
    device_str = train_cfg.get("device", "cpu")
    device = torch.device(device_str)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("Requested cuda but not available; falling back to CPU", flush=True)
        device = torch.device("cpu")
    print(f"Training on {device}"
          + (f": {torch.cuda.get_device_name(device)}" if device.type == "cuda" else ""),
          flush=True)

    model_kwargs: Dict[str, Any] = {
        "hidden_dim": model_cfg.get("hidden_dim", 128),
        "num_heads": model_cfg.get("num_heads", 4),
        "num_layers": model_cfg.get("num_layers", 3),
    }
    policy = SchedulingPolicy(**model_kwargs).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=ppo_cfg.get("lr", 3e-4))

    start_episode = 1
    if resume is not None:
        checkpoint = torch.load(resume, weights_only=True, map_location=device)
        if isinstance(checkpoint, dict) and "policy" in checkpoint:
            policy.load_state_dict(checkpoint["policy"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_episode = checkpoint.get("episode", 0) + 1
        else:
            policy.load_state_dict(checkpoint)
        print(f"Resumed from {resume}, starting at episode {start_episode}", flush=True)

    gen = TrafficGenerator(seed=42)
    detection_window = env_cfg.get("detection_window", 10.0)
    commit_window = env_cfg.get("commit_window", 2.5)
    episode_duration = env_cfg.get("episode_duration", 60.0)
    env_kwargs: Dict[str, Any] = {
        "detection_window": detection_window,
        "commit_window": commit_window,
        "zone_positions": ZONE_POSITIONS,
        "penalty_coef": env_cfg.get("penalty_coef", 0.1),
        "max_proximity_weight": env_cfg.get("max_proximity_weight", 2.0),
    }
    env = DynamicIntersectionEnv(**env_kwargs)

    num_episodes = train_cfg.get("num_episodes", 50_000)
    log_interval = train_cfg.get("log_interval", 100)
    eval_interval = train_cfg.get("eval_interval", 1_000)
    checkpoint_interval = train_cfg.get("checkpoint_interval", 5_000)

    # num_workers > 1 collects rollout episodes in separate single-threaded
    # PROCESSES (not intra-op threads — those were measured to make the tiny
    # per-op HGT forwards SLOWER, even with --cpus-per-task matched to
    # OMP_NUM_THREADS). Match this to --cpus-per-task in the sbatch script;
    # keep OMP_NUM_THREADS=1 etc. regardless of how many workers this is set to.
    num_workers = max(1, int(train_cfg.get("num_workers", 1)))

    if num_workers > 1:
        # SchedulingPolicy's HGT uses lazy-initialized modules (materialize
        # parameter shapes on first real forward, see intersection_scheduler/
        # model/hgt.py). The serial loop below naturally triggers that on
        # episode 1 before ever touching state_dict(); the parallel path
        # snapshots state_dict() up front (before the main process has run
        # any episode), so force materialization now with one throwaway
        # warm-up episode — discarded, doesn't touch `gen`/curriculum or
        # count toward num_episodes/start_episode.
        _warmup_env = DynamicIntersectionEnv(**env_kwargs)
        _warmup_duration = min(episode_duration, 5.0)
        run_episode(policy, _warmup_env, TrafficGenerator(seed=0).easy(_warmup_duration), _warmup_duration)

    best_eval_metric = float("inf")
    best_state = None

    ROLLOUT_N = 8
    buffer: List[Transition] = []
    update_stats = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}

    print(
        f"Starting training: {num_episodes} episodes (log every {log_interval}), "
        f"num_workers={num_workers}",
        flush=True,
    )

    ep_time_ema = None  # exponential moving average of per-episode wall-clock

    def _process_episode(ep: int, ep_wall: float, transitions: List[Transition], stats: Dict[str, float]) -> None:
        nonlocal buffer, update_stats, best_eval_metric, best_state, ep_time_ema
        ep_time_ema = ep_wall if ep_time_ema is None else 0.98 * ep_time_ema + 0.02 * ep_wall

        compute_gae(
            transitions,
            gamma=ppo_cfg.get("gamma", 0.99),
            gae_lambda=ppo_cfg.get("gae_lambda", 0.95),
        )
        buffer.extend(transitions)

        if ep == start_episode:
            print(
                f"[{ep:6d}] first episode done "
                f"({stats['steps']} steps, {stats['n_vehicles_seen']} vehicles)",
                flush=True,
            )

        if ep % ROLLOUT_N == 0 and buffer:
            random.shuffle(buffer)
            update_stats = ppo_update(
                policy,
                optimizer,
                buffer,
                clip_epsilon=ppo_cfg.get("clip_epsilon", 0.5),
                epochs=ppo_cfg.get("epochs_per_update", 4),
                value_loss_coef=ppo_cfg.get("value_loss_coef", 0.5),
                entropy_coef=ppo_cfg.get("entropy_coef", 0.01),
                max_grad_norm=ppo_cfg.get("max_grad_norm", 0.5),
                mini_batch_size=ppo_cfg.get("mini_batch_size", 16),
            )
            buffer = []

        if ep % log_interval == 0:
            writer.add_scalar("train/n_vehicles_seen", stats["n_vehicles_seen"], ep)
            writer.add_scalar("train/n_vehicles_completed", stats["n_vehicles_completed"], ep)
            writer.add_scalar("train/waiting_time", stats["waiting_time"], ep)
            writer.add_scalar("train/makespan", stats["makespan"], ep)
            writer.add_scalar("train/total_reward", stats["total_reward"], ep)
            writer.add_scalar("train/steps", stats["steps"], ep)
            writer.add_scalar("train/actor_loss", update_stats["actor_loss"], ep)
            writer.add_scalar("train/critic_loss", update_stats["critic_loss"], ep)
            writer.add_scalar("train/entropy", update_stats["entropy"], ep)
            writer.add_scalar("train/completion_rate", stats["completion_rate"], ep)
            # Resource tracking: per-episode wall-clock (smoothed) and peak
            # process RSS. ru_maxrss is KB on Linux, bytes on macOS. Under
            # num_workers > 1, ep_wall is the round's wall-clock divided by
            # round size (an approximation, not this exact episode's time),
            # and peak_rss reflects only the main process, not the workers.
            import sys
            peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            _rss_divisor = 1024 ** 3 if sys.platform == "darwin" else 1024 ** 2
            peak_gb = peak_rss / _rss_divisor
            writer.add_scalar("perf/episode_seconds", ep_wall, ep)
            writer.add_scalar("perf/peak_rss_gb", peak_gb, ep)
            # Projected time to finish all remaining episodes at current pace.
            eta_hours = ep_time_ema * (num_episodes - ep) / 3600.0
            print(
                f"[{ep:6d}] seen={stats['n_vehicles_seen']:3d}  "
                f"done={stats['n_vehicles_completed']:3d}  "
                f"wt={stats['waiting_time']:.2f}  "
                f"comp={stats['completion_rate']:.2f}  "
                f"rew={stats['total_reward']:.2f}  "
                f"steps={stats['steps']}  "
                f"| {ep_wall:.1f}s/ep (avg {ep_time_ema:.1f})  "
                f"mem={peak_gb:.1f}GB  eta={eta_hours:.1f}h",
                flush=True,
            )

        if ep % eval_interval == 0:
            eval_wt, eval_comp = _evaluate(policy, env, gen, episode_duration, n_episodes=10)
            writer.add_scalar("eval/waiting_time", eval_wt, ep)
            writer.add_scalar("eval/completion_rate", eval_comp, ep)
            print(f"  >>> EVAL waiting_time={eval_wt:.3f}  completion_rate={eval_comp:.2f}", flush=True)
            if eval_wt < best_eval_metric:
                best_eval_metric = eval_wt
                best_state = copy.deepcopy(policy.state_dict())
                torch.save(best_state, out / "checkpoint_best.pt")

        if ep % checkpoint_interval == 0:
            torch.save({
                "episode": ep,
                "policy": policy.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, out / f"checkpoint_{ep}.pt")

    if num_workers <= 1:
        for episode in range(start_episode, num_episodes + 1):
            ep_start = time.perf_counter()
            arrivals = get_curriculum_arrivals(episode, gen, episode_duration)
            transitions, stats = run_episode(policy, env, arrivals, episode_duration)
            ep_wall = time.perf_counter() - ep_start
            _process_episode(episode, ep_wall, transitions, stats)
    else:
        # Episodes within a round share the SAME (frozen) policy weights —
        # the policy only changes at a ppo_update boundary — so a round of
        # up to num_workers episodes can be collected fully in parallel
        # across single-threaded worker processes, then merged back into
        # the identical serial bookkeeping (_process_episode) in episode
        # order. This mirrors the offline 4x4 trainer's device comment:
        # the win here comes from process-level parallelism over whole
        # episodes, never from intra-op threading over one tiny forward.
        with ProcessPoolExecutor(max_workers=num_workers, initializer=_worker_init) as executor:
            for round_start in range(start_episode, num_episodes + 1, num_workers):
                round_episodes = list(range(round_start, min(round_start + num_workers, num_episodes + 1)))
                arrivals_list = [get_curriculum_arrivals(ep, gen, episode_duration) for ep in round_episodes]
                # Snapshot CPU weights once per round; workers never mutate
                # the main process's policy/optimizer.
                state_dict = {k: v.detach().cpu() for k, v in policy.state_dict().items()}

                round_start_t = time.perf_counter()
                futures = [
                    executor.submit(
                        _rollout_worker, state_dict, model_kwargs, env_kwargs,
                        episode_duration, arrivals, ep,
                    )
                    for ep, arrivals in zip(round_episodes, arrivals_list)
                ]
                results = [f.result() for f in futures]
                per_ep_wall = (time.perf_counter() - round_start_t) / len(round_episodes)

                for ep, (transitions, stats) in zip(round_episodes, results):
                    _process_episode(ep, per_ep_wall, transitions, stats)

    writer.close()
    if best_state is not None:
        torch.save(best_state, out / "checkpoint_best.pt")
    print(f"Training complete. Best eval metric: {best_eval_metric:.3f}", flush=True)


def _evaluate(
    policy: SchedulingPolicy,
    env: DynamicIntersectionEnv,
    gen: TrafficGenerator,
    episode_duration: float,
    n_episodes: int = 10,
) -> "tuple[float, float]":
    """Return (mean_waiting_time, completion_rate) over n_episodes hard
    (high arrival-rate) episodes with the deterministic policy. Waiting time
    is the unbiased all-vehicles metric (see dynamic_scheduler.utils.metrics)."""
    policy.eval()
    total_wt = 0.0
    total_seen = 0
    total_completed = 0
    for _ in range(n_episodes):
        arrivals = gen.hard(episode_duration)
        _, stats = run_episode(policy, env, arrivals, episode_duration, deterministic=True)
        # stats["waiting_time"] is the unbiased all-vehicles metric (completed
        # + in-flight), whose denominator is vehicles SEEN — so weight by seen,
        # not completed, to get a correct pooled mean across episodes.
        total_wt += stats["waiting_time"] * stats["n_vehicles_seen"]
        total_seen += stats["n_vehicles_seen"]
        total_completed += stats["n_vehicles_completed"]
    policy.train()
    mean_wt = total_wt / total_seen if total_seen > 0 else 0.0
    comp_rate = total_completed / total_seen if total_seen > 0 else 0.0
    # Return the mean waiting time; completion rate is logged separately by
    # the caller via the second return value.
    return mean_wt, comp_rate
