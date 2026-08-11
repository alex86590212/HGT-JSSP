from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import yaml

from dynamic_scheduler.data.traffic_generator import TrafficGenerator
from dynamic_scheduler.environment.dynamic_intersection import DynamicIntersectionEnv
from evaluation.eval_dynamic import (
    backpressure_select,
    edf_select,
    igreedy_select,
    lifo_select,
)
from intersection_scheduler.data.scenario_generator_4x4 import ZONE_POSITIONS
from intersection_scheduler.model.policy import SchedulingPolicy
from sumo_validation import sumo_control
from sumo_validation.route_generator import build_route_file
from sumo_validation.translate import translate

HERE = Path(__file__).resolve().parent
NET_DIR = HERE / "network"


def _ensure_traci_on_path() -> None:
    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home:
        tools = os.path.join(sumo_home, "tools")
        if tools not in sys.path:
            sys.path.append(tools)


def make_hgt_selector(checkpoint: str, model_cfg: dict):
    from dynamic_scheduler.environment.feasibility import compute_feasible_set  # noqa: F401
    from dynamic_scheduler.environment.graph_builder import build_hetero_graph

    policy = SchedulingPolicy(
        hidden_dim=model_cfg.get("hidden_dim", 128),
        num_heads=model_cfg.get("num_heads", 4),
        num_layers=model_cfg.get("num_layers", 3),
    )
    ckpt = torch.load(checkpoint, weights_only=True, map_location="cpu")
    state_dict = ckpt["policy"] if isinstance(ckpt, dict) and "policy" in ckpt else ckpt
    policy.load_state_dict(state_dict)
    policy.eval()

    def select(env, mask):
        data = build_hetero_graph(env, feasible_mask=mask)
        with torch.no_grad():
            dist, _ = policy(data, mask)
        return int(dist.probs.argmax().item())

    return select


SELECTORS = {
    "igreedy": lambda checkpoint, model_cfg: igreedy_select,
    "lifo": lambda checkpoint, model_cfg: lifo_select,
    "backpressure": lambda checkpoint, model_cfg: backpressure_select,
    "edf": lambda checkpoint, model_cfg: edf_select,
    "hgt": make_hgt_selector,
}


def run_one_method(
    method: str,
    checkpoint: str,
    model_cfg: dict,
    env_cfg: dict,
    arrivals,
    episode_duration: float,
    gui: bool,
    sumocfg: Path,
) -> float:
    from simulation.run_simulation_dynamic import run_recorded_episode

    env = DynamicIntersectionEnv(
        detection_window=env_cfg.get("detection_window", 10.0),
        commit_window=env_cfg.get("commit_window", 2.5),
        zone_positions=ZONE_POSITIONS,
        penalty_coef=env_cfg.get("penalty_coef", 0.1),
        max_proximity_weight=env_cfg.get("max_proximity_weight", 2.0),
    )
    selector = SELECTORS[method](checkpoint, model_cfg)
    _, _, op_timeline = run_recorded_episode(env, arrivals, episode_duration, selector)

    schedules = translate(op_timeline, arrivals)

    route_path = NET_DIR / "grid_4x4.rou.xml"
    build_route_file(arrivals, route_path)

    _ensure_traci_on_path()
    import traci

    binary = "sumo-gui" if gui else "sumo"
    if shutil.which(binary) is None:
        raise RuntimeError(f"{binary} not found on PATH. Install SUMO first (brew install sumo).")

    tripinfo_path = NET_DIR / f"tripinfo_{method}.xml"
    traci.start([
        binary, "-c", str(sumocfg),
        "--no-warnings", "true",
        "--time-to-teleport", "-1",
        # Vehicles depart at the scheduler's real velocity (8-14 m/s) right
        # onto lanes feeding priority junctions; SUMO's default insertion
        # safety check rejects this as "unpriorised junction too close"
        # even though the scheduler already resolves right-of-way via the
        # reservation schedule enforced in sumo_control.py. Confirmed via a
        # plain (non-TraCI) sumo run: identical scenario has zero rejected
        # departures with this flag, many without it.
        "--insertion-checks", "none",
        "--tripinfo-output", str(tripinfo_path),
        "--tripinfo-output.write-unfinished", "true",
    ])
    state = sumo_control.ControllerState()
    max_sim_time = episode_duration + 300.0
    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            sim_time = traci.simulation.getTime()
            if sim_time > max_sim_time:
                stuck = traci.vehicle.getIDList()
                print(f"  WARNING: hit max_sim_time={max_sim_time}s with "
                      f"{len(stuck)} vehicles still active: {sorted(stuck, key=int)}")
                break
            traci.simulationStep()
            sim_time = traci.simulation.getTime()
            sumo_control.step_control(schedules, sim_time, state)
    finally:
        traci.close()

    from sumo_validation.tripinfo import read_time_loss
    time_losses = read_time_loss(tripinfo_path)
    if not time_losses:
        return 0.0
    return sum(time_losses.values()) / len(time_losses)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="results_dynamic_reward_fix/checkpoint_best_30_so_far (1).pt")
    ap.add_argument("--config", default="configs/default_dynamic.yaml")
    ap.add_argument("--tier", choices=["easy", "medium", "hard"], default="hard")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gui-method", default=None,
                     help="Run this one method (e.g. hgt) in SUMO-GUI instead of headless.")
    ap.add_argument("--methods", default="hgt,igreedy,lifo,backpressure,edf")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg.get("model", {})
    env_cfg = cfg.get("environment", {})
    episode_duration = env_cfg.get("episode_duration", 60.0)

    net_path = NET_DIR / "grid_4x4.net.xml"
    sumocfg = NET_DIR / "grid_4x4.sumocfg"
    if not net_path.exists():
        subprocess.run([sys.executable, str(NET_DIR / "build_grid_network.py")], check=True)

    gen = TrafficGenerator(seed=args.seed)
    arrivals = getattr(gen, args.tier)(episode_duration)
    print(f"Scenario: {args.tier}, {len(arrivals)} vehicles, seed={args.seed}")

    results = {}
    for method in args.methods.split(","):
        gui = method == args.gui_method
        print(f"--- {method} ({'gui' if gui else 'headless'}) ---")
        wt = run_one_method(
            method, args.checkpoint, model_cfg, env_cfg,
            arrivals, episode_duration, gui, sumocfg,
        )
        results[method] = wt
        print(f"  sumo avg waiting time (timeLoss) = {wt:.3f}s")

    print()
    print(f"{'method':14s} {'avg_waiting_time_s':>20s}")
    for method, wt in results.items():
        print(f"{method:14s} {wt:20.3f}")


if __name__ == "__main__":
    main()
