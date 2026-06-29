"""Abstract intersection simulation mirroring the FCFS TraCI controller structure.

The loop follows the same pattern as traci_runner.py:
  - simulationStep() → advance time by dt
  - get_approaching_vehicles() → build vehicle records
  - build_jssp_graph() → construct graph
  - scheduler.schedule() → HGT policy or iGreedy
  - apply_control() → release/hold vehicles
  - print_debug_step() → same console format

No SUMO required. Vehicles move through the abstract 3×3 zone grid.

Usage:
    PYTHONPATH=. python simulation/run_simulation.py --checkpoint results/checkpoint_best.pt
    PYTHONPATH=. python simulation/run_simulation.py --mode igreedy --difficulty hard
    PYTHONPATH=. python simulation/run_simulation.py --mode both --seed 7
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import numpy as np
import torch

from intersection_scheduler.data.scenario_generator import (
    ZONE_POSITIONS, ROUTES, ScenarioGenerator, Scenario,
)
from intersection_scheduler.environment.intersection import IntersectionEnv, Operation, Vehicle
from intersection_scheduler.environment.feasibility import compute_feasible_set
from intersection_scheduler.environment.graph_builder import build_hetero_graph
from intersection_scheduler.model.policy import SchedulingPolicy
from intersection_scheduler.utils.metrics import episode_waiting_time, episode_makespan


# ── Simulation constants (mirror traci_runner.py) ────────────────────────────

SIM_STEP_SECONDS = 0.1          # dt per simulationStep()
DETECTION_DISTANCE = 999.0      # all vehicles always detected (abstract env)
HOLD_DISTANCE      = 999.0

VEHICLE_COLORS = [
    "#e74c3c", "#2ecc71", "#3498db", "#f39c12",
    "#9b59b6", "#1abc9c", "#e67e22", "#34495e",
]

CENTRE_ZONES = {5}
EDGE_ZONES   = {2, 4, 6, 8}
CORNER_ZONES = {1, 3, 7, 9}


# ── Vehicle state (mirrors LiveVehicleRecord) ─────────────────────────────────

@dataclass
class SimVehicleRecord:
    vehicle_id: str
    vehicle_idx: int            # index into scenario.vehicles
    route: List[int]            # zone ids
    arrival_time: float
    velocity: float
    processing_times: List[float]
    # Dynamic
    status: str = "waiting"     # waiting | crossing | done
    current_zone_idx: int = -1  # index into route (-1 = not yet entered)
    zone_enter_time: float = 0.0
    zone_exit_time: float = 0.0


@dataclass
class ControllerState:
    """Mirrors traci_runner.ControllerState."""
    released_vehicles: Set[str] = field(default_factory=set)
    cleared_vehicles:  Set[str] = field(default_factory=set)
    stopped_vehicles:  Set[str] = field(default_factory=set)
    # zone_id -> vehicle_id currently crossing it
    zone_occupants: Dict[int, str] = field(default_factory=dict)
    # vehicle_id -> zone_exit_time for current zone
    zone_exit_times: Dict[str, float] = field(default_factory=dict)
    # which operation (zone_idx) each vehicle is on next
    next_op_idx: Dict[str, int] = field(default_factory=dict)
    # scheduled decisions: vehicle_id -> list of (zone_id, start, finish)
    schedule: Dict[str, List[Tuple[int, float, float]]] = field(default_factory=dict)


# ── Scheduler interface (mirrors BaseScheduler) ───────────────────────────────

class AbstractScheduler:
    name = "abstract"

    def next_action(
        self,
        records: List[SimVehicleRecord],
        state: ControllerState,
        sim_time: float,
    ) -> Optional[str]:
        """Return vehicle_id to release next, or None."""
        raise NotImplementedError


class HGTScheduler(AbstractScheduler):
    name = "hgt"

    def __init__(self, policy: SchedulingPolicy, env: IntersectionEnv) -> None:
        self.policy = policy
        self.env = env
        self._scheduled = False

    def set_scenario(self, scenario: Scenario) -> None:
        self.env.reset(scenario.vehicles)
        self._scheduled = False
        self._action_queue: List[int] = []  # pre-computed full schedule

    def next_action(self, records, state, sim_time):
        if not self._action_queue:
            return None
        return f"v{self._action_queue.pop(0)}"

    def precompute(self) -> None:
        """Run full HGT episode and store the action order."""
        env = self.env
        self._action_queue = []
        done = False
        while not done:
            data = build_hetero_graph(env)
            mask = compute_feasible_set(env)
            if not mask.any():
                break
            with torch.no_grad():
                dist, _ = self.policy(data, mask)
            action = int(dist.probs.argmax().item())
            self._action_queue.append(env.operations[action].vehicle_id)
            env, _, done = env.step(action)


class IGreedyScheduler(AbstractScheduler):
    name = "igreedy"

    def __init__(self, env: IntersectionEnv) -> None:
        self.env = env
        self._action_queue: List[int] = []

    def set_scenario(self, scenario: Scenario) -> None:
        self.env.reset(scenario.vehicles)
        self._action_queue = []

    def precompute(self) -> None:
        env = self.env
        self._action_queue = []
        done = False
        while not done:
            mask = compute_feasible_set(env)
            if not mask.any():
                break
            best_idx, best_key = None, None
            for idx, op in enumerate(env.operations):
                if not mask[idx].item():
                    continue
                v = next(v for v in env.vehicles if v.id == op.vehicle_id)
                key = (v.arrival_time, op.route_position, op.vehicle_id)
                if best_key is None or key < best_key:
                    best_key, best_idx = key, idx
            self._action_queue.append(env.operations[best_idx].vehicle_id)
            env, _, done = env.step(best_idx)

    def next_action(self, records, state, sim_time):
        if not self._action_queue:
            return None
        return f"v{self._action_queue.pop(0)}"


# ── Simulation step functions (mirror traci_runner.py) ────────────────────────

def get_approaching_vehicles(
    records: List[SimVehicleRecord],
    sim_time: float,
) -> List[SimVehicleRecord]:
    """Return vehicles that have arrived and aren't done yet."""
    return [
        r for r in records
        if r.arrival_time <= sim_time and r.status != "done"
    ]


def apply_control(
    approaching: List[SimVehicleRecord],
    all_records: List[SimVehicleRecord],
    state: ControllerState,
    scheduler: AbstractScheduler,
    sim_time: float,
) -> Dict:
    """Mirror apply_sumo_control() — release/hold vehicles each step."""

    stopped_now:  List[str] = []
    released_now: List[str] = []
    cleared_now:  List[str] = []

    # 1. Advance vehicles that are currently crossing a zone
    for rec in all_records:
        if rec.status != "crossing":
            continue
        if sim_time >= rec.zone_exit_time:
            # Finished this zone
            zone_id = rec.route[rec.current_zone_idx]
            if state.zone_occupants.get(zone_id) == rec.vehicle_id:
                del state.zone_occupants[zone_id]
            state.zone_exit_times.pop(rec.vehicle_id, None)

            next_idx = rec.current_zone_idx + 1
            if next_idx >= len(rec.route):
                # Vehicle has cleared the intersection
                rec.status = "done"
                state.cleared_vehicles.add(rec.vehicle_id)
                cleared_now.append(rec.vehicle_id)
            else:
                # Move to waiting for next zone
                rec.status = "waiting"
                rec.current_zone_idx = next_idx
                state.stopped_vehicles.add(rec.vehicle_id)

    # 2. Ask scheduler who to release next
    next_vid = scheduler.next_action(approaching, state, sim_time)

    # 3. Release the chosen vehicle into its next zone
    if next_vid is not None:
        rec = next((r for r in all_records if r.vehicle_id == next_vid), None)
        if rec is not None and rec.status in ("waiting", "arriving"):
            if rec.current_zone_idx < 0:
                rec.current_zone_idx = 0
            zone_id = rec.route[rec.current_zone_idx]

            # Only release if zone is free
            if zone_id not in state.zone_occupants:
                pt = rec.processing_times[rec.current_zone_idx]
                rec.zone_enter_time = sim_time
                rec.zone_exit_time  = sim_time + pt
                rec.status = "crossing"
                state.zone_occupants[zone_id] = rec.vehicle_id
                state.zone_exit_times[rec.vehicle_id] = rec.zone_exit_time
                state.released_vehicles.add(rec.vehicle_id)
                state.stopped_vehicles.discard(rec.vehicle_id)
                released_now.append(next_vid)

    # 4. Hold vehicles that aren't released
    for rec in approaching:
        if rec.status == "waiting" and rec.vehicle_id not in state.released_vehicles:
            state.stopped_vehicles.add(rec.vehicle_id)
            if rec.vehicle_id not in released_now:
                stopped_now.append(rec.vehicle_id)

    return {
        "released_now": released_now,
        "stopped_now":  list(set(stopped_now)),
        "cleared_now":  cleared_now,
        "zone_occupants": dict(state.zone_occupants),
        "active_vehicles": [r.vehicle_id for r in all_records if r.status == "crossing"],
    }


def print_debug_step(
    sim_time: float,
    approaching: List[SimVehicleRecord],
    control_result: Dict,
    decision_number: int,
) -> None:
    """Mirror print_debug_step() from traci_runner.py."""
    print(
        f"\n[t={sim_time:.1f}s | decision={decision_number}] "
        f"approaching={len(approaching)}  "
        f"active={control_result['active_vehicles']}  "
        f"released={control_result['released_now']}"
    )
    for rec in approaching:
        status_str = f"{rec.status:8s}"
        zone_str = (
            f"zone=z{rec.route[rec.current_zone_idx]}"
            if rec.current_zone_idx >= 0 and rec.status == "crossing"
            else "zone=---"
        )
        print(
            f"  {rec.vehicle_id:<6}  arr={rec.arrival_time:.2f}s  "
            f"vel={rec.velocity:.1f}m/s  {status_str}  {zone_str}"
        )
    if control_result["stopped_now"]:
        print(f"  Held:    {control_result['stopped_now']}")
    if control_result["cleared_now"]:
        print(f"  Cleared: {control_result['cleared_now']}")


# ── Simulation runner ─────────────────────────────────────────────────────────

@dataclass
class SimSnapshot:
    """One frame of simulation state for animation playback."""
    sim_time: float
    vehicle_statuses: Dict[str, str]          # vehicle_id -> status
    vehicle_zone: Dict[str, Optional[int]]    # vehicle_id -> current zone_id
    vehicle_progress: Dict[str, float]        # vehicle_id -> 0..1 within zone
    zone_occupants: Dict[int, str]            # zone_id -> vehicle_id


def run_simulation(
    scenario: Scenario,
    scheduler: AbstractScheduler,
    decision_period: float = 1.0,
    max_time: float = 60.0,
) -> Tuple[List[SimSnapshot], Dict]:
    """Run the full simulation loop, mirror of run_controller()."""

    # Build vehicle records
    records: List[SimVehicleRecord] = []
    for v in scenario.vehicles:
        rec = SimVehicleRecord(
            vehicle_id=f"v{v.id}",
            vehicle_idx=v.id,
            route=v.route,
            arrival_time=v.arrival_time,
            velocity=v.velocity,
            processing_times=v.processing_times,
        )
        records.append(rec)

    # Pre-compute full schedule from policy/igreedy
    scheduler.precompute()

    state = ControllerState()
    snapshots: List[SimSnapshot] = []
    sim_time = 0.0
    step_count = 0
    decision_number = 0
    next_decision_time = 0.0

    # Mark initial vehicle status
    for rec in records:
        rec.status = "arriving"
        rec.current_zone_idx = 0

    print(f"\n{'='*60}")
    print(f"Scheduler: {scheduler.name.upper()}  |  {len(records)} vehicles")
    print(f"{'='*60}")

    while sim_time <= max_time:
        # simulationStep()
        sim_time = round(sim_time + SIM_STEP_SECONDS, 6)
        step_count += 1

        # Vehicles that have arrived
        for rec in records:
            if rec.arrival_time <= sim_time and rec.status == "arriving":
                rec.status = "waiting"

        approaching = get_approaching_vehicles(records, sim_time)

        # Apply control every step
        control_result = apply_control(approaching, records, state, scheduler, sim_time)

        # Capture snapshot every step for animation
        snap = SimSnapshot(
            sim_time=sim_time,
            vehicle_statuses={r.vehicle_id: r.status for r in records},
            vehicle_zone={
                r.vehicle_id: (r.route[r.current_zone_idx] if r.status == "crossing" else None)
                for r in records
            },
            vehicle_progress={
                r.vehicle_id: (
                    (sim_time - r.zone_enter_time) / max(r.zone_exit_time - r.zone_enter_time, 1e-9)
                    if r.status == "crossing" else 0.0
                )
                for r in records
            },
            zone_occupants=dict(state.zone_occupants),
        )
        snapshots.append(snap)

        # Debug print at decision_period intervals
        if sim_time + 1e-9 >= next_decision_time:
            print_debug_step(sim_time, approaching, control_result, decision_number)
            next_decision_time = sim_time + decision_period
            decision_number += 1

        # Done when all vehicles cleared
        if len(state.cleared_vehicles) == len(records):
            print(f"\n[t={sim_time:.1f}s] All {len(records)} vehicles cleared.")
            break

    # Compute final metrics using env state
    env = scheduler.env
    wt  = episode_waiting_time(env)
    ms  = episode_makespan(env)
    stats = {
        "waiting_time": wt,
        "makespan":     ms,
        "steps":        step_count,
        "sim_time":     sim_time,
        "scheduler":    scheduler.name,
    }
    print(f"\nResults: waiting_time={wt:.3f}s  makespan={ms:.3f}s")
    return snapshots, stats


# ── Animation ─────────────────────────────────────────────────────────────────

def animate(
    snapshots: List[SimSnapshot],
    scenario: Scenario,
    title: str,
    fps: int = 30,
    playback_speed: float = 2.0,
) -> None:
    # Sub-sample snapshots to match fps and playback speed
    step_dt = SIM_STEP_SECONDS
    skip = max(1, int(playback_speed / (fps * step_dt)))
    frames = snapshots[::skip]

    fig, ax = plt.subplots(figsize=(9, 9))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")
    ax.set_xlim(-0.8, 2.8)
    ax.set_ylim(-0.8, 2.8)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.suptitle(title, color="white", fontsize=13, fontweight="bold")

    # Road background
    ax.fill_between([-0.8, 2.8], [0.55, 0.55], [1.45, 1.45], color="#2c3e50", zorder=1)
    ax.fill_betweenx([-0.8, 2.8], [0.55, 0.55], [1.45, 1.45], color="#2c3e50", zorder=1)
    # Lane dashes
    for pos in [0.0, 2.0]:
        ax.plot([0.95, 1.05], [pos, pos], "--", color="#f39c12", lw=1, alpha=0.5, zorder=2)
        ax.plot([pos, pos], [0.95, 1.05], "--", color="#f39c12", lw=1, alpha=0.5, zorder=2)

    # Zone circles
    zone_patches: Dict[int, plt.Circle] = {}
    for zid, (x, y) in ZONE_POSITIONS.items():
        base = "#fadbd8" if zid in CENTRE_ZONES else "#fdebd0" if zid in EDGE_ZONES else "#d6eaf8"
        c = plt.Circle((x, y), 0.32, color=base, zorder=3, alpha=0.9)
        ax.add_patch(c)
        zone_patches[zid] = c
        ax.text(x, y - 0.42, f"z{zid}", ha="center", va="center",
                fontsize=7, color="#bdc3c7", zorder=4)

    # Vehicle circles + labels
    veh_patches: Dict[str, plt.Circle] = {}
    veh_texts:   Dict[str, plt.Text]   = {}
    for v in scenario.vehicles:
        vid = f"v{v.id}"
        col = VEHICLE_COLORS[v.id % len(VEHICLE_COLORS)]
        c = plt.Circle((0, 0), 0.13, color=col, zorder=6, visible=False)
        ax.add_patch(c)
        veh_patches[vid] = c
        t = ax.text(0, 0, vid, ha="center", va="center",
                    fontsize=6, color="white", fontweight="bold", zorder=7, visible=False)
        veh_texts[vid] = t

    # Direction labels
    for label, (x, y) in [("N",(1.0,2.7)),("S",(1.0,-0.7)),("W",(-0.7,1.0)),("E",(2.7,1.0))]:
        ax.text(x, y, label, ha="center", va="center", color="#95a5a6",
                fontsize=11, fontweight="bold", zorder=2)

    # Clock + stats
    time_text  = ax.text(1.0, -0.6,  "", ha="center", color="white", fontsize=10, zorder=8)
    stats_text = ax.text(-0.75, 2.72, "", ha="left", color="#ecf0f1",
                         fontsize=7, va="top", zorder=8, family="monospace")

    # Legend
    handles = [
        mpatches.Patch(color=VEHICLE_COLORS[v.id % len(VEHICLE_COLORS)],
                       label=f"v{v.id} {scenario.manoeuvres[i]} (arr {v.arrival_time:.1f}s)")
        for i, v in enumerate(scenario.vehicles)
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=6,
              facecolor="#0f3460", edgecolor="#e94560", labelcolor="white")

    def _vehicle_display_pos(snap: SimSnapshot, vid: str) -> Optional[Tuple[float, float]]:
        status = snap.vehicle_statuses.get(vid, "arriving")
        if status == "arriving":
            return None
        if status == "done":
            return None
        zone_id = snap.vehicle_zone.get(vid)

        if status == "crossing" and zone_id is not None:
            x, y = ZONE_POSITIONS[zone_id]
            return float(x), float(y)

        if status == "waiting":
            # Show vehicle hovering just outside its next zone
            vi = next(v for v in scenario.vehicles if f"v{v.id}" == vid)
            rec_route = vi.route
            # Find current zone_idx from scenario
            next_zone = rec_route[0]
            for other_snap_vid, oz in snap.vehicle_zone.items():
                if other_snap_vid == vid and oz is not None:
                    next_zone = oz
                    break
            x, y = ZONE_POSITIONS[next_zone]
            cx, cy = 1.0, 1.0
            dx, dy = x - cx, y - cy
            dist = max(np.sqrt(dx**2 + dy**2), 1e-9)
            return x + dx/dist * 0.55, y + dy/dist * 0.55

        return None

    def update(frame_idx: int):
        snap = frames[frame_idx]

        # Zone colours
        for zid, patch in zone_patches.items():
            occupant = snap.zone_occupants.get(zid)
            if occupant is not None:
                vid_idx = int(occupant[1:])
                patch.set_facecolor(VEHICLE_COLORS[vid_idx % len(VEHICLE_COLORS)])
                patch.set_alpha(0.95)
            else:
                base = "#fadbd8" if zid in CENTRE_ZONES else "#fdebd0" if zid in EDGE_ZONES else "#d6eaf8"
                patch.set_facecolor(base)
                patch.set_alpha(0.9)

        # Vehicle positions
        for v in scenario.vehicles:
            vid = f"v{v.id}"
            pos = _vehicle_display_pos(snap, vid)
            patch = veh_patches[vid]
            txt   = veh_texts[vid]
            if pos is None:
                patch.set_visible(False)
                txt.set_visible(False)
            else:
                patch.center = pos
                txt.set_position(pos)
                patch.set_visible(True)
                txt.set_visible(True)

        # Clock
        time_text.set_text(f"t = {snap.sim_time:.2f}s")

        # Stats
        done_count = sum(1 for s in snap.vehicle_statuses.values() if s == "done")
        active = [vid for vid, s in snap.vehicle_statuses.items() if s == "crossing"]
        stats_text.set_text(
            f"Vehicles done:  {done_count}/{len(scenario.vehicles)}\n"
            f"Active in zone: {', '.join(active) or '—'}\n"
            f"Held:           {', '.join(vid for vid, s in snap.vehicle_statuses.items() if s == 'waiting') or '—'}"
        )

        return (list(zone_patches.values()) + list(veh_patches.values()) +
                list(veh_texts.values()) + [time_text, stats_text])

    fig._anim = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=1000 // fps, blit=False
    )
    plt.tight_layout()
    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Abstract intersection simulation — mirrors FCFS TraCI controller structure."
    )
    parser.add_argument("--checkpoint", default="results/checkpoint_best.pt",
                        help="Path to HGT policy checkpoint")
    parser.add_argument("--mode", choices=["hgt", "igreedy", "both"], default="hgt")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="hard")
    parser.add_argument("--n_vehicles", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decision-period", type=float, default=1.0,
                        help="Seconds between debug log lines (mirrors --decision-period)")
    parser.add_argument("--speed", type=float, default=2.0,
                        help="Animation playback speed multiplier")
    parser.add_argument("--no-gui", action="store_true",
                        help="Run headless (no animation window)")
    args = parser.parse_args()

    gen = ScenarioGenerator(seed=args.seed)
    n = args.n_vehicles or {"easy": 2, "medium": 4, "hard": 5}[args.difficulty]
    scenario = getattr(gen, args.difficulty)(n_vehicles=n)

    print(f"Scenario: {args.difficulty}, {n} vehicles, seed={args.seed}")
    for v, m in zip(scenario.vehicles, scenario.manoeuvres):
        print(f"  v{v.id}: {m:5s}  route={v.route}  "
              f"arrival={v.arrival_time:.2f}s  vel={v.velocity:.1f}m/s")

    env = IntersectionEnv()

    modes = ["hgt", "igreedy"] if args.mode == "both" else [args.mode]

    for mode in modes:
        if mode == "hgt":
            policy = SchedulingPolicy(hidden_dim=128, num_heads=4, num_layers=3)
            ckpt = torch.load(args.checkpoint, weights_only=True)
            if isinstance(ckpt, dict) and "policy" in ckpt:
                policy.load_state_dict(ckpt["policy"])
            else:
                policy.load_state_dict(ckpt)
            policy.eval()
            print(f"\nLoaded checkpoint: {args.checkpoint}")
            scheduler = HGTScheduler(policy, env)
        else:
            scheduler = IGreedyScheduler(env)

        scheduler.set_scenario(scenario)

        snapshots, stats = run_simulation(
            scenario=scenario,
            scheduler=scheduler,
            decision_period=args.decision_period,
        )

        if not args.no_gui:
            title = (
                f"{'HGT Policy' if mode == 'hgt' else 'iGreedy'}  —  "
                f"{args.difficulty} ({n} vehicles)  |  "
                f"wt={stats['waiting_time']:.2f}s  mkspan={stats['makespan']:.2f}s"
            )
            animate(snapshots, scenario, title, playback_speed=args.speed)


if __name__ == "__main__":
    main()
