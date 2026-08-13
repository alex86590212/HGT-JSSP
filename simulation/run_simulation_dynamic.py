"""Intersection simulation and visualization — online dynamic 4x4 scheduler.

Adapted from simulation/run_simulation_4x4.py for the ONLINE model. The
static sim solves a whole episode first, then time-steps through the
finished schedule for animation — that split doesn't apply here, since
vehicles arrive continuously and plans are revised live. Instead this
records one snapshot after every replanning event AS the real event-driven
loop runs (same mechanics as dynamic_scheduler.training.trainer.run_episode
/ evaluation/eval_dynamic.py's run_episode_with), so the animation shows the
actual online decision process, not a replay of a precomputed result.

Differences from the static view, reflecting what's actually new here:
  - Vehicles are invisible until DETECTED (detection_window before arrival),
    not present from t=0 — matches the online model's information limit.
  - Operations are three-state (UNSCHEDULED / TENTATIVE / LOCKED), not
    binary scheduled/unscheduled. Zone circles and vehicle markers are
    colour-coded for this; TENTATIVE plans can still be revised on-screen.
  - The middle panel is a live node/edge graph, same visual language as the
    static sim's JSSP timing-conflict graph, but drawn from THIS model's
    actual precedence structure (see dynamic_intersection.py's
    _try_recompute_times / feasibility.py's deadlock check) rather than
    forcing the static Type-1/2/3 scheme onto a different mechanism:
      ROUTE edges   — op(i,j) -> op(i,j+1), a vehicle's own route order.
      QUEUE edges   — consecutive TENTATIVE ops in a zone's priority queue,
                      in planning order. This is what replaces Type-3: a
                      zone's queue order literally IS the resolved passing
                      order here, so there's no separate "undecided
                      conflict" edge type to draw — the queue edge already
                      encodes the current decision, and re-drawing it live
                      as replans reorder queues is the dynamic-model
                      analogue of watching Type-3 edges get fixed one at a
                      time in the static graph.
    LOCKED windows show as static (frozen) boxes alongside the graph rather
    than nodes that keep moving, since a locked op's timing/edges no longer
    change — mirrors env._recompute_times treating them as fixed windows.

Usage:
    PYTHONPATH=. python simulation/run_simulation_dynamic.py --checkpoint results_dynamic_realistic_v2/checkpoint_best.pt
    PYTHONPATH=. python simulation/run_simulation_dynamic.py --mode all --tier hard --seed 7
    PYTHONPATH=. python simulation/run_simulation_dynamic.py --mode edf --no-gui

--mode all runs hgt, igreedy, lifo, backpressure, edf in sequence (one
animation window each, in --no-gui mode one printed summary each) — same
five methods evaluation/eval_dynamic.py compares.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.animation as animation
import numpy as np
import torch

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False

from dynamic_scheduler.data.traffic_generator import TrafficGenerator
from dynamic_scheduler.environment.dynamic_intersection import (
    DynamicIntersectionEnv,
    OpState,
)
from dynamic_scheduler.environment.feasibility import compute_feasible_set
from dynamic_scheduler.environment.graph_builder import build_hetero_graph
from dynamic_scheduler.utils.metrics import (
    completion_rate,
    episode_waiting_time_all,
)
from intersection_scheduler.data.scenario_generator_4x4 import ZONE_POSITIONS
from intersection_scheduler.model.policy import SchedulingPolicy

try:
    from evaluation.eval_dynamic import igreedy_select, lifo_select, backpressure_select, edf_select
except ImportError:
    igreedy_select = lifo_select = backpressure_select = edf_select = None  # --mode hgt only still works without this


# Same hard caps as trainer.run_episode / eval_dynamic.run_episode_with.
MAX_REPLAN_ITERS = 500
MAX_OUTER_ITERS = 5000

VEHICLE_COLORS = [
    "#e74c3c", "#2ecc71", "#3498db", "#f39c12",
    "#9b59b6", "#1abc9c", "#e67e22", "#34495e",
    "#c0392b", "#27ae60", "#2980b9", "#d35400",
]

TENTATIVE_RING = "#f2c94c"   # amber — revisable
LOCKED_RING    = "#d94f45"   # red — frozen
UNDETECTED_ALPHA = 0.0

ROUTE_EDGE_COLOR = "#2f2f2f"   # route order (Type-1 analog)
QUEUE_EDGE_COLOR = "#d62728"   # zone-queue order (Type-3 analog — the resolved priority)
ACTIVE_NODE_RING = "#f2c94c"
LOCKED_NODE_RING = "#d94f45"

CORE_ZONES = {6, 7, 10, 11}

GRID_MIN = -0.8
GRID_MAX = 3.8
GRID_CENTRE = 1.5


def _zone_base_color(zid: int) -> str:
    return "#fadbd8" if zid in CORE_ZONES else "#d6eaf8"


# ── Phase 1: run the REAL online episode loop, recording one snapshot per event ──

@dataclass
class DynSnapshot:
    sim_time: float
    vehicle_pos: Dict[int, Optional[Tuple[float, float]]]   # None = not yet detected / done
    vehicle_op_state: Dict[int, Optional[OpState]]           # state of the vehicle's ACTIVE op
    zone_occupant: Dict[int, Optional[int]]                  # zone_id -> vehicle_id currently locked there
    zone_queues: Dict[int, List[Tuple[int, str]]]             # zone_id -> [(vehicle_id, state_str), ...] in queue order
    n_detected: int
    n_completed: int
    just_planned: Optional[int]   # vehicle_id of the action just taken this frame, or None
    op_graph: object = None   # nx.DiGraph of all planned (TENTATIVE+LOCKED) ops, or None


def _vehicle_positions(env: DynamicIntersectionEnv) -> Dict[int, Optional[Tuple[float, float]]]:
    """Map each currently-tracked vehicle to a map coordinate.

    A vehicle mid-crossing (its active op is TENTATIVE/LOCKED and currently
    running) sits at its zone's position; a detected-but-not-yet-crossing
    vehicle sits just outside its entry zone (same offset convention as the
    static sim's arriving-vehicle placement)."""
    pos: Dict[int, Optional[Tuple[float, float]]] = {}
    ops_by_vid: Dict[int, list] = {}
    for op in env.operations:
        ops_by_vid.setdefault(op.vehicle_id, []).append(op)

    for vid, vehicle in env.vehicles.items():
        ops = sorted(ops_by_vid.get(vid, []), key=lambda o: o.route_position)
        if not ops:
            pos[vid] = None
            continue
        active = None
        for op in ops:
            if op.state != OpState.UNSCHEDULED and op.start_time <= env.current_time < op.earliest_finish:
                active = op
                break
        if active is not None:
            x, y = ZONE_POSITIONS[active.zone_id]
            pos[vid] = (float(x), float(y))
            continue
        # Not currently mid-crossing: sit just outside the first zone, offset
        # from the intersection centre (matches the static sim's approach).
        zone_id = vehicle.route[0]
        x, y = ZONE_POSITIONS[zone_id]
        cx, cy = GRID_CENTRE, GRID_CENTRE
        dx, dy = x - cx, y - cy
        dist = max(np.sqrt(dx**2 + dy**2), 1e-9)
        pos[vid] = (x + dx / dist * 0.55, y + dy / dist * 0.55)
    return pos


def _vehicle_op_states(env: DynamicIntersectionEnv) -> Dict[int, Optional[OpState]]:
    """Each tracked vehicle's most-advanced non-UNSCHEDULED op state, for
    ring-colour coding (TENTATIVE vs LOCKED)."""
    states: Dict[int, Optional[OpState]] = {}
    ops_by_vid: Dict[int, list] = {}
    for op in env.operations:
        ops_by_vid.setdefault(op.vehicle_id, []).append(op)
    for vid in env.vehicles:
        ops = ops_by_vid.get(vid, [])
        planned = [o for o in ops if o.state != OpState.UNSCHEDULED]
        states[vid] = planned[-1].state if planned else None
    return states


def _zone_occupants(env: DynamicIntersectionEnv) -> Dict[int, Optional[int]]:
    occ: Dict[int, Optional[int]] = {}
    for op in env.operations:
        if op.state == OpState.LOCKED and op.start_time <= env.current_time < op.earliest_finish:
            occ[op.zone_id] = op.vehicle_id
    return occ


def _zone_queue_snapshot(env: DynamicIntersectionEnv) -> Dict[int, List[Tuple[int, str]]]:
    return {
        zid: [(op.vehicle_id, op.state.name) for op in queue]
        for zid, queue in env._zone_queue.items()
        if queue
    }


def _build_op_graph(env: DynamicIntersectionEnv, just_planned_op: Optional[object] = None):
    """Node/edge graph of every currently PLANNED (TENTATIVE or LOCKED)
    operation — the same visual language as the static sim's JSSP graph,
    built from THIS model's real precedence structure instead of forcing
    the static Type-1/2/3 scheme onto it:

      node  = one planned operation (vehicle, zone)
      ROUTE edge  op(i,j) -> op(i,j+1), same vehicle, consecutive route hop
      QUEUE edge  consecutive TENTATIVE ops in one zone's priority queue,
                  in planning order — this IS the resolved passing order
                  (see dynamic_intersection.py._try_recompute_times), so
                  there's no separate "undecided conflict" edge to draw;
                  LOCKED ops are static windows and contribute no queue
                  edges (matches _try_recompute_times treating them as
                  fixed, non-participating constants).
    """
    if not HAS_NX:
        return None
    g = nx.DiGraph()

    planned = [op for op in env.operations if op.state != OpState.UNSCHEDULED]
    if not planned:
        return g

    for op in planned:
        node_id = f"v{op.vehicle_id}_z{op.zone_id}"
        g.add_node(
            node_id,
            vehicle_id=op.vehicle_id,
            zone_id=op.zone_id,
            route_position=op.route_position,
            processing_time=op.processing_time,
            state=op.state.name,
            start_time=op.start_time,
            just_planned=(op is just_planned_op),
        )

    pos_index = {(op.vehicle_id, op.route_position): op for op in env.operations}
    for op in planned:
        succ = pos_index.get((op.vehicle_id, op.route_position + 1))
        if succ is not None and succ.state != OpState.UNSCHEDULED:
            g.add_edge(f"v{op.vehicle_id}_z{op.zone_id}", f"v{succ.vehicle_id}_z{succ.zone_id}",
                       edge_type="route")

    for zid, queue in env._zone_queue.items():
        for a, b in zip(queue, queue[1:]):
            g.add_edge(f"v{a.vehicle_id}_z{a.zone_id}", f"v{b.vehicle_id}_z{b.zone_id}",
                       edge_type="queue")

    return g


def run_recorded_episode(
    env: DynamicIntersectionEnv,
    arrivals,
    episode_duration: float,
    select_action: Callable,
    n_completed_seen: Optional[List[int]] = None,
    verbose_replan: bool = False,
) -> Tuple[List[DynSnapshot], dict]:
    """Drive the real online replanning loop (identical structure to
    trainer.run_episode / eval_dynamic.run_episode_with) and record one
    DynSnapshot after every plan_operation call and every advance_time call
    — the actual sequence of events the policy/env goes through, not a
    solve-then-replay reconstruction."""
    env.reset(copy.deepcopy(arrivals), episode_duration)
    snapshots: List[DynSnapshot] = []
    n_seen = len(env.vehicles)

    # Per-zone timeline, latest known timing per (vehicle_id, route_position)
    # — a vehicle's ops are DELETED from env.operations once it completes
    # (see DynamicIntersectionEnv._remove_completed_vehicles), so this has to
    # be captured live as the episode runs, not read back at the end. Every
    # snapshot overwrites with whatever's currently known; since a locked
    # op's timing never changes again, this naturally converges to the
    # final value by the time an op locks, for BOTH completed and
    # still-in-flight vehicles.
    op_timeline: Dict[Tuple[int, int], dict] = {}

    def _record_timeline() -> None:
        for op in env.operations:
            if op.state == OpState.UNSCHEDULED:
                continue
            op_timeline[(op.vehicle_id, op.route_position)] = {
                "zone_id": op.zone_id,
                "start": op.start_time,
                "finish": op.earliest_finish,
                "state": op.state.name,
            }

    def snap(just_planned: Optional[int] = None, just_planned_op: Optional[object] = None) -> None:
        _record_timeline()
        snapshots.append(DynSnapshot(
            sim_time=env.current_time,
            vehicle_pos=_vehicle_positions(env),
            vehicle_op_state=_vehicle_op_states(env),
            zone_occupant=_zone_occupants(env),
            zone_queues=_zone_queue_snapshot(env),
            n_detected=n_seen,
            n_completed=len(env.completed_log),
            just_planned=just_planned,
            op_graph=_build_op_graph(env, just_planned_op),
        ))

    trigger_vids = list(env.vehicles.keys())
    snap()

    done = False
    outer_iters = 0
    while not done:
        outer_iters += 1
        if outer_iters > MAX_OUTER_ITERS:
            raise RuntimeError("episode outer loop exceeded MAX_OUTER_ITERS")

        affected = set(env.affected_op_indices(trigger_vids))
        visited: set = set()
        replan_iters = 0
        while True:
            replan_iters += 1
            if replan_iters > MAX_REPLAN_ITERS:
                raise RuntimeError("replan pass exceeded MAX_REPLAN_ITERS")
            candidates = affected - visited
            if not candidates:
                break
            mask = compute_feasible_set(env, candidates)
            if not mask.any():
                break
            action = select_action(env, mask)
            planned_op = env.operations[action]
            planned_vid = planned_op.vehicle_id

            # A REPLAN (requeue) is an op that was already TENTATIVE before
            # this call — plan_operation transitions BOTH first plans and
            # replans to TENTATIVE, so this must be captured before the call
            # or the two cases become indistinguishable afterward.
            is_replan = verbose_replan and planned_op.state == OpState.TENTATIVE
            if is_replan:
                # Position is 1-based (1 = next through the zone), captured
                # before plan_operation mutates the queue. start/finish are
                # captured too — a requeue can leave POSITION unchanged (e.g.
                # sole occupant of its queue) while still recomputing timing,
                # or vice versa, so position alone doesn't tell you whether
                # anything actually moved; report both explicitly rather
                # than let "going 1, now going 1" read as a no-op when the
                # timing may (or may not) have changed underneath it.
                queue_before = env._zone_queue.get(planned_op.zone_id, [])
                old_pos = queue_before.index(planned_op) + 1 if planned_op in queue_before else "?"
                old_start = planned_op.start_time

            env, reward, _ = env.plan_operation(action)
            visited.add(action)

            if is_replan:
                # plan_operation always appends to the back, so the new
                # position is simply the (now-updated) queue length.
                new_pos = len(env._zone_queue.get(planned_op.zone_id, []))
                timing_changed = abs(planned_op.start_time - old_start) > 1e-6
                if timing_changed:
                    timing_note = (
                        f"timing SHIFTED {old_start:.2f}→{planned_op.start_time:.2f} "
                        f"(penalty paid, reward={reward:+.3f})"
                    )
                else:
                    timing_note = f"timing UNCHANGED (no penalty, reward={reward:+.3f})"
                print(
                    f"  [t={env.current_time:6.2f}] REPLAN: v{planned_vid}'s z{planned_op.zone_id} op "
                    f"was going {old_pos}, now going {new_pos} in that zone's queue  |  {timing_note}"
                )

            snap(just_planned=planned_vid, just_planned_op=planned_op)

        done, newly_detected = env.advance_time()
        n_seen += len(newly_detected)
        snap()

    inflight = env.inflight_waiting_times()
    stats = {
        "waiting_time": episode_waiting_time_all(env.completed_log, inflight),
        "completion_rate": completion_rate(len(env.completed_log), n_seen),
        "n_vehicles_seen": n_seen,
        "n_vehicles_completed": len(env.completed_log),
    }
    return snapshots, stats, op_timeline


def print_schedule(op_timeline: Dict[Tuple[int, int], dict], arrivals, scheduler_name: str) -> None:
    """Per-vehicle zone timeline, same format as the static sim's
    print_schedule — one line per vehicle, z{zone}[start->finish] in route
    order. Sourced from op_timeline (captured live during the episode,
    since completed vehicles' ops are deleted from the env by the time this
    runs — see run_recorded_episode's _record_timeline)."""
    manoeuvre_by_vid = {v.id: v.manoeuvre for v in arrivals}
    by_vehicle: Dict[int, List[Tuple[int, dict]]] = {}
    for (vid, route_pos), entry in op_timeline.items():
        by_vehicle.setdefault(vid, []).append((route_pos, entry))

    print(f"\n{'='*60}")
    print(f"Scheduler: {scheduler_name.upper()}  |  {len(arrivals)} vehicles, "
          f"{len(by_vehicle)} reached at least one planned op")
    print(f"{'='*60}")
    for vid in sorted(by_vehicle):
        man = manoeuvre_by_vid.get(vid, "")
        ops = sorted(by_vehicle[vid], key=lambda t: t[0])
        timeline = "  ".join(
            f"z{e['zone_id']}[{e['start']:.2f}→{e['finish']:.2f}]"
            + ("" if e["state"] == "LOCKED" else "~")   # ~ marks still-tentative (not yet frozen)
            for _, e in ops
        )
        print(f"  v{vid} ({man}): {timeline}")
    unreached = [v.id for v in arrivals if v.id not in by_vehicle]
    if unreached:
        print(f"  (never planned / not yet detected: {', '.join(f'v{vid}' for vid in unreached)})")
    print("  (~ = still tentative at episode end, not yet locked)")


def make_hgt_selector(policy: SchedulingPolicy):
    device = next(policy.parameters()).device

    def select(env: DynamicIntersectionEnv, mask: torch.Tensor) -> int:
        data = build_hetero_graph(env, feasible_mask=mask)
        with torch.no_grad():
            dist, _ = policy(data.to(device), mask.to(device))
        return int(dist.probs.argmax().item())

    return select


# ── Animation ──────────────────────────────────────────────────────────────

def animate(
    snapshots: List[DynSnapshot],
    arrivals,
    title: str,
    stats: dict,
    fps: int = 15,
    playback_speed: float = 1.0,
) -> None:
    """Step through recorded event snapshots (NOT a fixed-timestep replay —
    each snapshot is one real event: a plan_operation call or a time
    advance), holding each frame for a duration proportional to the
    simulated time it represents so fast bursts of replanning don't crawl
    and long idle gaps don't stall."""
    manoeuvre_by_vid = {v.id: v.manoeuvre for v in arrivals}
    arrival_by_vid = {v.id: v.arrival_time for v in arrivals}
    vehicle_ids = sorted(arrival_by_vid.keys())

    fig = plt.figure(figsize=(18, 8))
    fig.patch.set_facecolor("#1a1a2e")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.3, 0.9], wspace=0.08)
    ax_map = fig.add_subplot(gs[0])
    ax_queues = fig.add_subplot(gs[1])
    ax_panel = fig.add_subplot(gs[2])

    fig.suptitle(title, color="white", fontsize=12, fontweight="bold")

    # ── Intersection map ──
    ax_map.set_facecolor("#16213e")
    ax_map.set_xlim(GRID_MIN, GRID_MAX)
    ax_map.set_ylim(GRID_MIN, GRID_MAX)
    ax_map.set_aspect("equal")
    ax_map.axis("off")
    ax_map.set_title("Intersection (live)", color="white", fontsize=10)

    ax_map.fill_between([GRID_MIN, GRID_MAX], [-0.5, -0.5], [3.5, 3.5], color="#2c3e50", zorder=1)
    ax_map.fill_betweenx([GRID_MIN, GRID_MAX], [-0.5, -0.5], [3.5, 3.5], color="#2c3e50", zorder=1)

    zone_patches: Dict[int, plt.Circle] = {}
    for zid, (x, y) in ZONE_POSITIONS.items():
        c = plt.Circle((x, y), 0.30, color=_zone_base_color(zid), zorder=3, alpha=0.85)
        ax_map.add_patch(c)
        zone_patches[zid] = c
        ax_map.text(x, y, f"z{zid}", ha="center", va="center",
                    fontsize=7, color="#2c3e50", fontweight="bold", zorder=4)

    veh_patches: Dict[int, plt.Circle] = {}
    veh_texts: Dict[int, plt.Text] = {}
    for vid in vehicle_ids:
        col = VEHICLE_COLORS[vid % len(VEHICLE_COLORS)]
        c = plt.Circle((0, 0), 0.15, facecolor=col, edgecolor=col, linewidth=2.5, zorder=6, visible=False)
        ax_map.add_patch(c)
        veh_patches[vid] = c
        t = ax_map.text(0, 0, f"v{vid}", ha="center", va="center",
                        fontsize=5, color="white", fontweight="bold", zorder=7, visible=False)
        veh_texts[vid] = t

    for label, (x, y) in [
        ("N", (GRID_CENTRE, 3.7)), ("S", (GRID_CENTRE, -0.65)),
        ("W", (-0.65, GRID_CENTRE)), ("E", (3.65, GRID_CENTRE)),
    ]:
        ax_map.text(x, y, label, ha="center", va="center", color="#95a5a6",
                    fontsize=11, fontweight="bold", zorder=2)

    time_text = ax_map.text(GRID_CENTRE, -0.58, "", ha="center", color="white", fontsize=9, zorder=8)

    ring_legend = [
        mlines.Line2D([0], [0], marker="o", color="w", markerfacecolor="#888",
                      markeredgecolor=TENTATIVE_RING, markeredgewidth=2.5, markersize=8, label="Tentative"),
        mlines.Line2D([0], [0], marker="o", color="w", markerfacecolor="#888",
                      markeredgecolor=LOCKED_RING, markeredgewidth=2.5, markersize=8, label="Locked"),
    ]
    ax_map.legend(handles=ring_legend, loc="lower right", fontsize=7,
                  facecolor="#0f3460", edgecolor="#e94560", labelcolor="white")

    # ── Op precedence graph (route + zone-queue edges) ──
    ax_queues.set_facecolor("#16213e")
    ax_queues.axis("off")
    ax_queues.set_title("Op Graph — route order + zone-queue order", color="white", fontsize=10)

    vid_to_row = {vid: -float(i) for i, vid in enumerate(vehicle_ids)}

    def _draw_op_graph(snap: DynSnapshot) -> None:
        ax_queues.clear()
        ax_queues.set_facecolor("#16213e")
        ax_queues.axis("off")
        ax_queues.set_title("Op Graph — route order + zone-queue order", color="white", fontsize=10)

        g = snap.op_graph
        if g is None or g.number_of_nodes() == 0:
            ax_queues.text(0.5, 0.5, "No planned operations yet", ha="center", va="center",
                            color="#95a5a6", transform=ax_queues.transAxes, fontsize=10)
            return

        pos = {}
        for node_id, attrs in g.nodes(data=True):
            pos[node_id] = (float(attrs["route_position"]), vid_to_row.get(attrs["vehicle_id"], 0.0))

        route_edges = [(u, v) for u, v, d in g.edges(data=True) if d.get("edge_type") == "route"]
        queue_edges = [(u, v) for u, v, d in g.edges(data=True) if d.get("edge_type") == "queue"]

        node_colors, edge_colors, edge_widths, node_sizes, labels = [], [], [], [], {}
        for node_id, attrs in g.nodes(data=True):
            vid = attrs["vehicle_id"]
            node_colors.append(VEHICLE_COLORS[vid % len(VEHICLE_COLORS)])
            if attrs.get("just_planned"):
                edge_colors.append("#ffffff")
                edge_widths.append(3.5)
                node_sizes.append(900)
            elif attrs["state"] == "LOCKED":
                edge_colors.append(LOCKED_NODE_RING)
                edge_widths.append(2.5)
                node_sizes.append(750)
            else:
                edge_colors.append(ACTIVE_NODE_RING)
                edge_widths.append(2.0)
                node_sizes.append(700)
            labels[node_id] = f"v{vid}\nz{attrs['zone_id']}\n{attrs['processing_time']:.1f}s"

        nx.draw_networkx_edges(g, pos, edgelist=route_edges, edge_color=ROUTE_EDGE_COLOR,
                                arrows=True, arrowstyle="-|>", width=2.0,
                                connectionstyle="arc3,rad=0.05", ax=ax_queues)
        nx.draw_networkx_edges(g, pos, edgelist=queue_edges, edge_color=QUEUE_EDGE_COLOR,
                                arrows=True, arrowstyle="-|>", style="dashed", width=1.8,
                                connectionstyle="arc3,rad=0.2", ax=ax_queues)
        nx.draw_networkx_nodes(g, pos, node_color=node_colors, node_size=node_sizes,
                                edgecolors=edge_colors, linewidths=edge_widths, ax=ax_queues)
        nx.draw_networkx_labels(g, pos, labels=labels, font_size=5,
                                 font_color="white", ax=ax_queues)
        ax_queues.margins(x=0.25, y=0.25)

        legend = [
            mlines.Line2D([0], [0], color=ROUTE_EDGE_COLOR, lw=2, label="Route order"),
            mlines.Line2D([0], [0], color=QUEUE_EDGE_COLOR, lw=1.8, ls="--", label="Zone-queue order"),
            mlines.Line2D([0], [0], marker="o", color="w", markerfacecolor="#888",
                          markeredgecolor="#ffffff", markeredgewidth=3, markersize=8, label="Just (re)planned"),
            mlines.Line2D([0], [0], marker="o", color="w", markerfacecolor="#888",
                          markeredgecolor=ACTIVE_NODE_RING, markeredgewidth=2, markersize=8, label="Tentative"),
            mlines.Line2D([0], [0], marker="o", color="w", markerfacecolor="#888",
                          markeredgecolor=LOCKED_NODE_RING, markeredgewidth=2, markersize=8, label="Locked"),
        ]
        ax_queues.legend(handles=legend, loc="upper left", fontsize=6,
                          facecolor="#0f3460", edgecolor="#555", labelcolor="white")

    # ── Debug/stats panel ──
    ax_panel.set_facecolor("#16213e")
    ax_panel.axis("off")
    ax_panel.set_title("Debug", color="white", fontsize=10)
    panel_text_obj = ax_panel.text(
        0.02, 0.98, "", ha="left", va="top",
        family="monospace", fontsize=7, color="#ecf0f1",
        transform=ax_panel.transAxes,
    )

    def update(frame_idx: int):
        snap = snapshots[frame_idx]

        for zid, patch in zone_patches.items():
            occupant = snap.zone_occupant.get(zid)
            if occupant is not None:
                patch.set_facecolor(VEHICLE_COLORS[occupant % len(VEHICLE_COLORS)])
                patch.set_alpha(0.95)
            else:
                patch.set_facecolor(_zone_base_color(zid))
                patch.set_alpha(0.85)

        for vid in vehicle_ids:
            pos = snap.vehicle_pos.get(vid)
            state = snap.vehicle_op_state.get(vid)
            if pos is None:
                veh_patches[vid].set_visible(False)
                veh_texts[vid].set_visible(False)
                continue
            veh_patches[vid].center = pos
            veh_texts[vid].set_position(pos)
            veh_patches[vid].set_visible(True)
            veh_texts[vid].set_visible(True)
            if state == OpState.LOCKED:
                veh_patches[vid].set_edgecolor(LOCKED_RING)
                veh_patches[vid].set_linewidth(3.0)
            elif state == OpState.TENTATIVE:
                veh_patches[vid].set_edgecolor(TENTATIVE_RING)
                veh_patches[vid].set_linewidth(3.0)
            else:
                col = VEHICLE_COLORS[vid % len(VEHICLE_COLORS)]
                veh_patches[vid].set_edgecolor(col)
                veh_patches[vid].set_linewidth(1.0)

        time_text.set_text(f"t = {snap.sim_time:.2f}s")

        _draw_op_graph(snap)

        # Compact per-zone queue order, as a secondary text summary
        # alongside the graph (same info, quicker to read the raw order).
        queue_lines = []
        for zid in sorted(snap.zone_queues.keys()):
            q = snap.zone_queues[zid]
            entries = []
            for vid, state in q:
                mark = "*" if vid == snap.just_planned else ""
                tag = "L" if state == "LOCKED" else "T"
                entries.append(f"{mark}v{vid}({tag})")
            queue_lines.append(f"z{zid}:" + ">".join(entries))

        active_vids = [f"v{vid}" for vid, s in snap.vehicle_op_state.items()
                       if s is not None and snap.vehicle_pos.get(vid) is not None]
        panel_text_obj.set_text(
            f"t = {snap.sim_time:.2f}s\n"
            f"Detected: {snap.n_detected}\n"
            f"Completed:{snap.n_completed}\n"
            f"Active:   {', '.join(active_vids) or '—'}\n"
            f"Last action: {'v' + str(snap.just_planned) if snap.just_planned is not None else '—'}\n"
            "\nZone queues\n-----------\n" +
            ("\n".join(queue_lines) if queue_lines else "(none yet)") +
            "\n\nVehicles\n"
            "--------\n" +
            "\n".join(
                f"v{vid} {manoeuvre_by_vid.get(vid, '')}\n"
                f"  arr={arrival_by_vid[vid]:.1f}s"
                for vid in vehicle_ids
                if snap.vehicle_pos.get(vid) is not None
            )
        )
        return []

    # Hold each event-frame for a duration proportional to how much
    # simulated time elapsed since the previous one (bursts of replanning
    # at the same instant render fast; idle gaps don't stall the viewer).
    sim_deltas = [0.0] + [
        max(0.0, snapshots[i].sim_time - snapshots[i - 1].sim_time)
        for i in range(1, len(snapshots))
    ]
    MIN_FRAME_MS = 1000 // fps
    frame_intervals = [
        max(MIN_FRAME_MS, int(1000 * dt / max(playback_speed, 1e-6)))
        for dt in sim_deltas
    ]

    # matplotlib's FuncAnimation wants one interval; approximate with the
    # median non-trivial gap and let the debug panel show real sim time —
    # good enough for a demo viewer, not a frame-accurate scientific plot.
    nonzero = [d for d in frame_intervals if d > MIN_FRAME_MS]
    interval = int(np.median(nonzero)) if nonzero else MIN_FRAME_MS

    fig._anim = animation.FuncAnimation(
        fig, update, frames=len(snapshots), interval=interval, blit=False
    )
    plt.tight_layout()
    plt.show()


# ── Entry point ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="results_dynamic_realistic_v2/checkpoint_best.pt")
    parser.add_argument("--config", default="configs/default_dynamic.yaml")
    parser.add_argument("--mode", choices=["hgt", "igreedy", "lifo", "backpressure", "edf", "all", "baselines"], default="hgt")
    parser.add_argument("--tier", choices=["easy", "medium", "hard"], default="hard")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier (1.0 = real-time-ish)")
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--verbose-replan", action="store_true",
                        help="Print a line every time an already-tentative op is requeued "
                             "(replanned), showing its old and new position in the zone's "
                             "priority queue — makes active rescheduling visible as it happens, "
                             "not just as an aggregate count.")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg.get("model", {})
    env_cfg = cfg.get("environment", {})
    episode_duration = env_cfg.get("episode_duration", 60.0)

    gen = TrafficGenerator(seed=args.seed)
    arrivals = getattr(gen, args.tier)(episode_duration)

    print(f"Scenario: {args.tier}, {len(arrivals)} vehicles, seed={args.seed}, "
          f"episode_duration={episode_duration}s")
    for v in sorted(arrivals, key=lambda v: v.arrival_time):
        print(f"  v{v.id}: {v.manoeuvre:6s}  route={v.route}  "
              f"arrival={v.arrival_time:.2f}s  vel={v.velocity:.1f}m/s")

    if args.mode == "all":
        modes = ["hgt", "igreedy", "lifo", "backpressure", "edf"]
    elif args.mode == "baselines":
        modes = ["igreedy", "lifo", "backpressure", "edf"]
    else:
        modes = [args.mode]

    for mode in modes:
        env = DynamicIntersectionEnv(
            detection_window=env_cfg.get("detection_window", 10.0),
            commit_window=env_cfg.get("commit_window", 2.5),
            zone_positions=ZONE_POSITIONS,
            penalty_coef=env_cfg.get("penalty_coef", 0.1),
            max_proximity_weight=env_cfg.get("max_proximity_weight", 2.0),
        )

        if mode == "hgt":
            policy = SchedulingPolicy(
                hidden_dim=model_cfg.get("hidden_dim", 128),
                num_heads=model_cfg.get("num_heads", 4),
                num_layers=model_cfg.get("num_layers", 3),
            )
            ckpt = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
            if isinstance(ckpt, dict) and "policy" in ckpt:
                policy.load_state_dict(ckpt["policy"])
                print(f"\nLoaded checkpoint: {args.checkpoint}  (episode {ckpt.get('episode', '?')})")
            else:
                policy.load_state_dict(ckpt)
                print(f"\nLoaded weights: {args.checkpoint}")
            policy.eval()
            selector = make_hgt_selector(policy)
        elif mode == "igreedy":
            if igreedy_select is None:
                raise RuntimeError("Could not import igreedy_select from evaluation.eval_dynamic")
            selector = igreedy_select
        elif mode == "lifo":
            if lifo_select is None:
                raise RuntimeError("Could not import lifo_select from evaluation.eval_dynamic")
            selector = lifo_select
        elif mode == "backpressure":
            if backpressure_select is None:
                raise RuntimeError("Could not import backpressure_select from evaluation.eval_dynamic")
            selector = backpressure_select
        elif mode == "edf":
            if edf_select is None:
                raise RuntimeError("Could not import edf_select from evaluation.eval_dynamic")
            selector = edf_select
        else:
            raise ValueError(mode)

        if args.verbose_replan:
            print(f"\n--- {mode.upper()}: live replan log ---")
        snapshots, stats, op_timeline = run_recorded_episode(
            env, arrivals, episode_duration, selector, verbose_replan=args.verbose_replan,
        )

        print_schedule(op_timeline, arrivals, mode)

        print(f"\n{'='*60}")
        print(f"Scheduler: {mode.upper()}  |  {len(arrivals)} vehicles  |  {len(snapshots)} events")
        print(f"{'='*60}")
        print(f"  waiting_time={stats['waiting_time']:.3f}s  "
              f"completion_rate={stats['completion_rate']:.2f}  "
              f"seen={stats['n_vehicles_seen']}  completed={stats['n_vehicles_completed']}")

        if not args.no_gui:
            title = (
                f"{mode.upper()}  —  {args.tier} ({len(arrivals)} vehicles)  |  "
                f"wt={stats['waiting_time']:.2f}s  comp={stats['completion_rate']:.2f}"
            )
            animate(snapshots, arrivals, title, stats, playback_speed=args.speed)


if __name__ == "__main__":
    main()
