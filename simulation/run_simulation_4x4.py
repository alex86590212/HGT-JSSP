"""Intersection simulation and visualization — 4x4 two-lane intersection.

Adapted copy of run_simulation.py for the 4x4 topology (16 zones, 2 lanes per
approach). Differences from the 3x3 version:
  - imports the 4x4 scenario_generator constants
  - threads ZONE_POSITIONS into env.reset (graph node features + map geometry)
  - map geometry (axis limits, road bands, direction labels, waiting-vehicle
    offset centre) scaled for the 4x4 grid which spans (0,0)-(3,3), centre (1.5,1.5)
  - core (crossing) zones {6,7,10,11} coloured distinctly from the 12 entry/exit zones

Usage:
    PYTHONPATH=. python simulation/run_simulation_4x4.py --checkpoint results_4x4/checkpoint_best.pt
    PYTHONPATH=. python simulation/run_simulation_4x4.py --mode both --difficulty hard --seed 7
    PYTHONPATH=. python simulation/run_simulation_4x4.py --mode igreedy --no-gui
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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

from intersection_scheduler.data.scenario_generator_4x4 import (
    ZONE_POSITIONS, ROUTES, SAME_LANE_GROUPS, ScenarioGenerator, Scenario,
)
from intersection_scheduler.environment.intersection import IntersectionEnv
from intersection_scheduler.environment.feasibility import compute_feasible_set, next_feasible_time
from intersection_scheduler.environment.graph_builder import build_hetero_graph
from intersection_scheduler.model.policy import SchedulingPolicy
from intersection_scheduler.utils.metrics import episode_waiting_time, episode_makespan


# ── Constants ─────────────────────────────────────────────────────────────────

SIM_STEP_SECONDS = 0.05

VEHICLE_COLORS = [
    "#e74c3c", "#2ecc71", "#3498db", "#f39c12",
    "#9b59b6", "#1abc9c", "#e67e22", "#34495e",
    "#c0392b", "#27ae60", "#2980b9", "#d35400",
]

TYPE1_COLOR   = "#2f2f2f"
TYPE2_COLOR   = "#1f77b4"
TYPE3_COLOR   = "#d62728"
ACTIVE_COLOR  = "#f2c94c"
STOPPED_COLOR = "#d94f45"

# 4x4: the 2x2 inner core are the shared crossing zones.
CORE_ZONES = {6, 7, 10, 11}

# Grid geometry (4x4 spans (0,0)-(3,3)).
GRID_MIN = -0.8
GRID_MAX = 3.8
GRID_CENTRE = 1.5

_MANOEUVRE_TO_LANE: Dict[str, str] = {
    m: lane
    for lane, manoeuvres in SAME_LANE_GROUPS.items()
    for m in manoeuvres
}


def _zone_base_color(zid: int) -> str:
    return "#fadbd8" if zid in CORE_ZONES else "#d6eaf8"


# ── Scheduled operation (output of phase 1) ───────────────────────────────────

@dataclass
class ScheduledOp:
    vehicle_id: int
    zone_id: int
    route_position: int
    start_time: float
    finish_time: float
    processing_time: float


# ── Phase 1: run episode and extract schedule ──────────────────────────────────

def run_hgt_episode(policy: SchedulingPolicy, env: IntersectionEnv, scenario: Scenario) -> List[ScheduledOp]:
    """Run HGT exactly as training does — no precomputation, online decisions."""
    env.reset(scenario.vehicles, zone_positions=ZONE_POSITIONS)
    done = False
    while not done:
        data = build_hetero_graph(env)
        mask = compute_feasible_set(env)
        if not mask.any():
            next_event = next_feasible_time(env)
            if next_event is None:
                break
            env.current_time = next_event
            continue
        with torch.no_grad():
            dist, _ = policy(data, mask)
        action = int(dist.probs.argmax().item())
        env, _, done = env.step(action)
    return _extract_schedule(env)


def run_igreedy_episode(env: IntersectionEnv, scenario: Scenario) -> List[ScheduledOp]:
    """Run iGreedy: earliest arrival first, tie-break by route position."""
    env.reset(scenario.vehicles, zone_positions=ZONE_POSITIONS)
    done = False
    while not done:
        mask = compute_feasible_set(env)
        if not mask.any():
            next_event = next_feasible_time(env)
            if next_event is None:
                break
            env.current_time = next_event
            continue
        best_idx, best_key = None, None
        for idx, op in enumerate(env.operations):
            if not mask[idx].item():
                continue
            v = next(v for v in env.vehicles if v.id == op.vehicle_id)
            key = (v.arrival_time, op.route_position, op.vehicle_id)
            if best_key is None or key < best_key:
                best_key, best_idx = key, idx
        env, _, done = env.step(best_idx)
    return _extract_schedule(env)


def _extract_schedule(env: IntersectionEnv) -> List[ScheduledOp]:
    ops = []
    for op in env.operations:
        if op.scheduled:
            ops.append(ScheduledOp(
                vehicle_id=op.vehicle_id,
                zone_id=op.zone_id,
                route_position=op.route_position,
                start_time=op.earliest_finish - op.processing_time,
                finish_time=op.earliest_finish,
                processing_time=op.processing_time,
            ))
    return ops


def print_schedule(schedule: List[ScheduledOp], scenario: Scenario, scheduler_name: str, env: IntersectionEnv) -> None:
    wt = episode_waiting_time(env)
    ms = episode_makespan(env)
    print(f"\n{'='*60}")
    print(f"Scheduler: {scheduler_name.upper()}  |  {len(scenario.vehicles)} vehicles")
    print(f"{'='*60}")
    by_vehicle: Dict[int, List[ScheduledOp]] = {}
    for op in sorted(schedule, key=lambda o: o.start_time):
        by_vehicle.setdefault(op.vehicle_id, []).append(op)
    for vid in sorted(by_vehicle):
        man = scenario.manoeuvres[vid]
        ops = sorted(by_vehicle[vid], key=lambda o: o.route_position)
        timeline = "  ".join(f"z{o.zone_id}[{o.start_time:.2f}→{o.finish_time:.2f}]" for o in ops)
        print(f"  v{vid} ({man}): {timeline}")
    print(f"\n  waiting_time={wt:.3f}s  makespan={ms:.3f}s")


# ── Phase 2: time-stepped simulation from schedule ────────────────────────────

@dataclass
class SimSnapshot:
    sim_time: float
    vehicle_zone: Dict[int, Optional[int]]
    vehicle_status: Dict[int, str]
    zone_occupants: Dict[int, int]
    released_now: List[int]
    waiting_vids: List[int]
    jssp_graph: object


def build_snapshots(
    schedule: List[ScheduledOp],
    scenario: Scenario,
    max_time: float = 120.0,
) -> List[SimSnapshot]:
    """Convert a solved schedule into per-tick snapshots for animation."""
    snapshots: List[SimSnapshot] = []
    sim_time = 0.0

    by_vehicle: Dict[int, List[ScheduledOp]] = {}
    for op in schedule:
        by_vehicle.setdefault(op.vehicle_id, []).append(op)
    for vid in by_vehicle:
        by_vehicle[vid].sort(key=lambda o: o.route_position)

    while sim_time <= max_time:
        sim_time = round(sim_time + SIM_STEP_SECONDS, 6)

        zone_occupants: Dict[int, int] = {}
        vehicle_zone: Dict[int, Optional[int]] = {}
        vehicle_status: Dict[int, str] = {}
        released_now: List[int] = []
        waiting_vids: List[int] = []

        for v in scenario.vehicles:
            vid = v.id
            ops = by_vehicle.get(vid, [])
            if not ops:
                vehicle_status[vid] = "done"
                vehicle_zone[vid] = None
                continue

            if sim_time < v.arrival_time:
                vehicle_status[vid] = "arriving"
                vehicle_zone[vid] = None
                continue

            active_op = None
            for op in ops:
                if op.start_time <= sim_time < op.finish_time:
                    active_op = op
                    break

            last_op = ops[-1]

            if sim_time >= last_op.finish_time:
                vehicle_status[vid] = "done"
                vehicle_zone[vid] = None
            elif active_op is not None:
                vehicle_status[vid] = "crossing"
                vehicle_zone[vid] = active_op.zone_id
                zone_occupants[active_op.zone_id] = vid
                if abs(active_op.start_time - sim_time) < SIM_STEP_SECONDS + 1e-6:
                    released_now.append(vid)
            else:
                vehicle_status[vid] = "waiting"
                vehicle_zone[vid] = None
                waiting_vids.append(vid)

        jssp_g = _build_jssp_graph(schedule, sim_time, scenario) if HAS_NX else None

        snapshots.append(SimSnapshot(
            sim_time=sim_time,
            vehicle_zone=vehicle_zone,
            vehicle_status=vehicle_status,
            zone_occupants=zone_occupants,
            released_now=released_now,
            waiting_vids=waiting_vids,
            jssp_graph=jssp_g,
        ))

        if all(s == "done" for s in vehicle_status.values()):
            break

    return snapshots


def _build_jssp_graph(schedule: List[ScheduledOp], sim_time: float, scenario: Scenario) -> "nx.DiGraph":
    """Build a JSSP graph showing remaining (unfinished) operations."""
    g = nx.DiGraph()

    active = [op for op in schedule if op.finish_time > sim_time]
    if not active:
        return g

    for op in active:
        node_id = f"v{op.vehicle_id}_z{op.zone_id}"
        status = "crossing" if op.start_time <= sim_time < op.finish_time else "pending"
        g.add_node(node_id, vehicle_id=op.vehicle_id, zone_id=op.zone_id,
                   route_position=op.route_position, processing_time=op.processing_time,
                   start_time=op.start_time, status=status)

    by_vid: Dict[int, List[ScheduledOp]] = {}
    for op in active:
        by_vid.setdefault(op.vehicle_id, []).append(op)
    for vid, ops in by_vid.items():
        ops_sorted = sorted(ops, key=lambda o: o.route_position)
        for i in range(len(ops_sorted) - 1):
            src = f"v{ops_sorted[i].vehicle_id}_z{ops_sorted[i].zone_id}"
            dst = f"v{ops_sorted[i+1].vehicle_id}_z{ops_sorted[i+1].zone_id}"
            if g.has_node(src) and g.has_node(dst):
                g.add_edge(src, dst, edge_type="type1_precedence")

    man_by_vid = {v.id: scenario.manoeuvres[v.id] for v in scenario.vehicles}
    by_lane: Dict[str, List[int]] = {}
    for vid in by_vid:
        lane = _MANOEUVRE_TO_LANE.get(man_by_vid.get(vid, ""), "")
        by_lane.setdefault(lane, []).append(vid)
    for lane_vids in by_lane.values():
        sorted_vids = sorted(lane_vids, key=lambda vid: next(
            v.arrival_time for v in scenario.vehicles if v.id == vid))
        for i in range(len(sorted_vids) - 1):
            e_ops = sorted(by_vid.get(sorted_vids[i], []), key=lambda o: o.route_position)
            l_ops = sorted(by_vid.get(sorted_vids[i+1], []), key=lambda o: o.route_position)
            if e_ops and l_ops:
                src = f"v{e_ops[0].vehicle_id}_z{e_ops[0].zone_id}"
                dst = f"v{l_ops[0].vehicle_id}_z{l_ops[0].zone_id}"
                if g.has_node(src) and g.has_node(dst):
                    g.add_edge(src, dst, edge_type="type2_same_lane_order")

    zone_to_nodes: Dict[int, List[str]] = {}
    for node_id, attrs in g.nodes(data=True):
        zone_to_nodes.setdefault(attrs["zone_id"], []).append(node_id)
    for zone_id, node_ids in zone_to_nodes.items():
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                a, b = node_ids[i], node_ids[j]
                if g.nodes[a]["vehicle_id"] != g.nodes[b]["vehicle_id"]:
                    g.add_edge(a, b, edge_type="type3_conflict")
                    g.add_edge(b, a, edge_type="type3_conflict")

    return g


# ── Animation ──────────────────────────────────────────────────────────────────

def animate(
    snapshots: List[SimSnapshot],
    scenario: Scenario,
    title: str,
    fps: int = 15,
    playback_speed: float = 0.3,
) -> None:
    skip = max(1, int(playback_speed / (fps * SIM_STEP_SECONDS)))
    frames = snapshots[::skip]

    has_jssp = HAS_NX and any(f.jssp_graph is not None for f in frames)
    if has_jssp:
        fig = plt.figure(figsize=(18, 8))
        fig.patch.set_facecolor("#1a1a2e")
        gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.6, 0.7], wspace=0.08)
        ax_map   = fig.add_subplot(gs[0])
        ax_jssp  = fig.add_subplot(gs[1])
        ax_panel = fig.add_subplot(gs[2])
    else:
        fig = plt.figure(figsize=(13, 8))
        fig.patch.set_facecolor("#1a1a2e")
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.7], wspace=0.08)
        ax_map   = fig.add_subplot(gs[0])
        ax_jssp  = None
        ax_panel = fig.add_subplot(gs[1])

    fig.suptitle(title, color="white", fontsize=12, fontweight="bold")

    # ── Intersection map ──────────────────────────────────────────────────────
    ax_map.set_facecolor("#16213e")
    ax_map.set_xlim(GRID_MIN, GRID_MAX)
    ax_map.set_ylim(GRID_MIN, GRID_MAX)
    ax_map.set_aspect("equal")
    ax_map.axis("off")
    ax_map.set_title("Intersection", color="white", fontsize=10)

    # Road bands: the 4 lanes span rows/cols 0..3, road covers the full band.
    ax_map.fill_between([GRID_MIN, GRID_MAX], [-0.5, -0.5], [3.5, 3.5], color="#2c3e50", zorder=1)
    ax_map.fill_betweenx([GRID_MIN, GRID_MAX], [-0.5, -0.5], [3.5, 3.5], color="#2c3e50", zorder=1)

    _draw_lane_arrows(ax_map, scenario)

    zone_patches: Dict[int, plt.Circle] = {}
    for zid, (x, y) in ZONE_POSITIONS.items():
        base = _zone_base_color(zid)
        c = plt.Circle((x, y), 0.30, color=base, zorder=3, alpha=0.85)
        ax_map.add_patch(c)
        zone_patches[zid] = c
        ax_map.text(x, y, f"z{zid}", ha="center", va="center",
                    fontsize=7, color="#2c3e50", fontweight="bold", zorder=4)

    veh_patches: Dict[int, plt.Circle] = {}
    veh_texts: Dict[int, plt.Text] = {}
    for v in scenario.vehicles:
        col = VEHICLE_COLORS[v.id % len(VEHICLE_COLORS)]
        c = plt.Circle((0, 0), 0.13, color=col, zorder=6, visible=False)
        ax_map.add_patch(c)
        veh_patches[v.id] = c
        t = ax_map.text(0, 0, f"v{v.id}", ha="center", va="center",
                        fontsize=5, color="white", fontweight="bold", zorder=7, visible=False)
        veh_texts[v.id] = t

    for label, (x, y) in [
        ("N", (GRID_CENTRE, 3.7)), ("S", (GRID_CENTRE, -0.65)),
        ("W", (-0.65, GRID_CENTRE)), ("E", (3.65, GRID_CENTRE)),
    ]:
        ax_map.text(x, y, label, ha="center", va="center", color="#95a5a6",
                    fontsize=11, fontweight="bold", zorder=2)

    time_text = ax_map.text(GRID_CENTRE, -0.58, "", ha="center", color="white", fontsize=9, zorder=8)

    handles = [
        mpatches.Patch(color=VEHICLE_COLORS[v.id % len(VEHICLE_COLORS)],
                       label=f"v{v.id} {scenario.manoeuvres[v.id]} (arr {v.arrival_time:.1f}s)")
        for v in scenario.vehicles
    ]
    ax_map.legend(handles=handles, loc="lower right", fontsize=5,
                  facecolor="#0f3460", edgecolor="#e94560", labelcolor="white")

    # ── Debug panel ───────────────────────────────────────────────────────────
    ax_panel.set_facecolor("#16213e")
    ax_panel.axis("off")
    ax_panel.set_title("Debug", color="white", fontsize=10)
    panel_text_obj = ax_panel.text(
        0.02, 0.98, "", ha="left", va="top",
        family="monospace", fontsize=7, color="#ecf0f1",
        transform=ax_panel.transAxes,
    )

    def _vehicle_map_pos(snap: SimSnapshot, vid: int) -> Optional[Tuple[float, float]]:
        status = snap.vehicle_status.get(vid, "arriving")
        if status in ("arriving", "done"):
            return None
        if status == "crossing":
            zone_id = snap.vehicle_zone.get(vid)
            if zone_id is not None:
                return float(ZONE_POSITIONS[zone_id][0]), float(ZONE_POSITIONS[zone_id][1])
        v = next(v for v in scenario.vehicles if v.id == vid)
        zone_id = v.route[0]
        x, y = ZONE_POSITIONS[zone_id]
        cx, cy = GRID_CENTRE, GRID_CENTRE
        dx, dy = x - cx, y - cy
        dist = max(np.sqrt(dx**2 + dy**2), 1e-9)
        return x + dx / dist * 0.55, y + dy / dist * 0.55

    def _draw_jssp_frame(snap: SimSnapshot) -> None:
        if ax_jssp is None:
            return
        ax_jssp.clear()
        ax_jssp.set_facecolor("#16213e")
        ax_jssp.axis("off")
        ax_jssp.set_title("JSSP Timing-Conflict Graph", color="white", fontsize=10)

        g = snap.jssp_graph
        if g is None or g.number_of_nodes() == 0:
            ax_jssp.text(0.5, 0.5, "No active operations", ha="center", va="center",
                         color="#95a5a6", transform=ax_jssp.transAxes, fontsize=10)
            return

        vid_to_row = {v.id: -float(i) for i, v in enumerate(scenario.vehicles)}
        pos = {}
        for node_id, attrs in g.nodes(data=True):
            pos[node_id] = (float(attrs["route_position"]), vid_to_row.get(attrs["vehicle_id"], 0.0))

        t1, t2, t3_pairs = [], [], set()
        for src, dst, attrs in g.edges(data=True):
            et = attrs.get("edge_type", "")
            if et == "type1_precedence":
                t1.append((src, dst))
            elif et == "type2_same_lane_order":
                t2.append((src, dst))
            elif et == "type3_conflict":
                t3_pairs.add(tuple(sorted([src, dst])))
        t3 = list(t3_pairs)

        node_colors, edge_colors, edge_widths, node_sizes, labels = [], [], [], [], {}
        for node_id, attrs in g.nodes(data=True):
            vid = attrs["vehicle_id"]
            node_colors.append(VEHICLE_COLORS[vid % len(VEHICLE_COLORS)])
            status = attrs.get("status", "pending")
            if status == "crossing":
                edge_colors.append(ACTIVE_COLOR)
                edge_widths.append(3.5)
                node_sizes.append(900)
            elif vid in snap.waiting_vids:
                edge_colors.append(STOPPED_COLOR)
                edge_widths.append(2.5)
                node_sizes.append(800)
            else:
                edge_colors.append("#555555")
                edge_widths.append(1.2)
                node_sizes.append(700)
            labels[node_id] = f"v{vid}\nz{attrs['zone_id']}\n{attrs['processing_time']:.1f}s"

        nx.draw_networkx_edges(g, pos, edgelist=t1, edge_color=TYPE1_COLOR,
                               arrows=True, arrowstyle="-|>", width=2.0,
                               connectionstyle="arc3,rad=0.05", ax=ax_jssp)
        nx.draw_networkx_edges(g, pos, edgelist=t2, edge_color=TYPE2_COLOR,
                               arrows=True, arrowstyle="-|>", style="dashed",
                               width=1.8, connectionstyle="arc3,rad=0.18", ax=ax_jssp)
        g_t3 = nx.Graph()
        g_t3.add_nodes_from(g.nodes())
        g_t3.add_edges_from(t3)
        nx.draw_networkx_edges(g_t3, pos, edge_color=TYPE3_COLOR,
                               style="dotted", width=1.5, alpha=0.8, ax=ax_jssp)
        nx.draw_networkx_nodes(g, pos, node_color=node_colors, node_size=node_sizes,
                               edgecolors=edge_colors, linewidths=edge_widths, ax=ax_jssp)
        nx.draw_networkx_labels(g, pos, labels=labels, font_size=5,
                                font_color="white", ax=ax_jssp)
        ax_jssp.margins(x=0.25, y=0.25)

        legend = [
            mlines.Line2D([0],[0], color=TYPE1_COLOR, lw=2, label="Type-1 route order"),
            mlines.Line2D([0],[0], color=TYPE2_COLOR, lw=2, ls="--", label="Type-2 lane order"),
            mlines.Line2D([0],[0], color=TYPE3_COLOR, lw=1.5, ls=":", label="Type-3 conflict"),
            mlines.Line2D([0],[0], marker="o", color="w", markerfacecolor="#888",
                          markeredgecolor=ACTIVE_COLOR, markeredgewidth=2.5, markersize=8, label="Crossing"),
            mlines.Line2D([0],[0], marker="o", color="w", markerfacecolor="#888",
                          markeredgecolor=STOPPED_COLOR, markeredgewidth=2.5, markersize=8, label="Waiting"),
        ]
        ax_jssp.legend(handles=legend, loc="upper left", fontsize=6,
                       facecolor="#0f3460", edgecolor="#555", labelcolor="white")

    def update(frame_idx: int):
        snap = frames[frame_idx]

        for zid, patch in zone_patches.items():
            occupant = snap.zone_occupants.get(zid)
            if occupant is not None:
                patch.set_facecolor(VEHICLE_COLORS[occupant % len(VEHICLE_COLORS)])
                patch.set_alpha(0.95)
            else:
                patch.set_facecolor(_zone_base_color(zid))
                patch.set_alpha(0.85)

        for v in scenario.vehicles:
            pos = _vehicle_map_pos(snap, v.id)
            if pos is None:
                veh_patches[v.id].set_visible(False)
                veh_texts[v.id].set_visible(False)
            else:
                veh_patches[v.id].center = pos
                veh_texts[v.id].set_position(pos)
                veh_patches[v.id].set_visible(True)
                veh_texts[v.id].set_visible(True)

        time_text.set_text(f"t = {snap.sim_time:.2f}s")
        _draw_jssp_frame(snap)

        done_count = sum(1 for s in snap.vehicle_status.values() if s == "done")
        active_vids = [f"v{vid}" for vid, s in snap.vehicle_status.items() if s == "crossing"]
        held_vids   = [f"v{vid}" for vid in snap.waiting_vids]
        g = snap.jssp_graph
        t1c = t2c = t3c = 0
        if g is not None:
            for _, _, d in g.edges(data=True):
                et = d.get("edge_type", "")
                if et == "type1_precedence":   t1c += 1
                elif et == "type2_same_lane_order": t2c += 1
                elif et == "type3_conflict":   t3c += 1
            t3c //= 2
        panel_text_obj.set_text(
            f"t = {snap.sim_time:.2f}s\n"
            f"Done:    {done_count}/{len(scenario.vehicles)}\n"
            f"Crossing:{', '.join(active_vids) or '—'}\n"
            f"Waiting: {', '.join(held_vids) or '—'}\n"
            "\nGraph edges\n"
            "-----------\n"
            f"T1 (route):   {t1c}\n"
            f"T2 (lane):    {t2c}\n"
            f"T3 (conflict):{t3c}\n"
            f"Nodes:        {g.number_of_nodes() if g else 0}\n"
            "\nVehicles\n"
            "--------\n" +
            "\n".join(
                f"v{v.id} {scenario.manoeuvres[v.id]}\n"
                f"  arr={v.arrival_time:.1f}s "
                f"  {snap.vehicle_status.get(v.id, '?')}"
                for v in scenario.vehicles
            )
        )

    fig._anim = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=1000 // fps, blit=False
    )
    plt.tight_layout()
    plt.show()


# ── Lane arrow drawing ─────────────────────────────────────────────────────────

def _draw_lane_arrows(ax: plt.Axes, scenario: Scenario) -> None:
    seen = set()
    for v in scenario.vehicles:
        key = tuple(v.route)
        if key in seen:
            continue
        seen.add(key)
        for i in range(len(v.route) - 1):
            x0, y0 = ZONE_POSITIONS[v.route[i]]
            x1, y1 = ZONE_POSITIONS[v.route[i + 1]]
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="-|>", color="#7f8c8d", lw=1.0, alpha=0.4),
                        zorder=2)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="results_4x4/checkpoint_best.pt")
    parser.add_argument("--mode", choices=["hgt", "igreedy", "both"], default="hgt")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="hard")
    parser.add_argument("--n_vehicles", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--speed", type=float, default=0.3)
    parser.add_argument("--no-gui", action="store_true")
    args = parser.parse_args()

    if not HAS_NX and not args.no_gui:
        print("Warning: networkx not installed — JSSP panel skipped. pip install networkx")

    gen = ScenarioGenerator(seed=args.seed)
    n = args.n_vehicles or {"easy": 4, "medium": 8, "hard": 12}[args.difficulty]
    scenario = getattr(gen, args.difficulty)(n_vehicles=n)

    print(f"Scenario: {args.difficulty}, {n} vehicles, seed={args.seed}")
    for v in scenario.vehicles:
        print(f"  v{v.id}: {scenario.manoeuvres[v.id]:6s}  route={v.route}  "
              f"arrival={v.arrival_time:.2f}s  vel={v.velocity:.1f}m/s")

    env = IntersectionEnv()
    modes = ["hgt", "igreedy"] if args.mode == "both" else [args.mode]

    for mode in modes:
        if mode == "hgt":
            policy = SchedulingPolicy(hidden_dim=128, num_heads=4, num_layers=3)
            ckpt = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
            if isinstance(ckpt, dict) and "policy" in ckpt:
                policy.load_state_dict(ckpt["policy"])
            else:
                policy.load_state_dict(ckpt)
            policy.eval()
            print(f"\nLoaded checkpoint: {args.checkpoint}")
            schedule = run_hgt_episode(policy, env, scenario)
        else:
            schedule = run_igreedy_episode(env, scenario)

        print_schedule(schedule, scenario, mode, env)

        if not args.no_gui:
            wt = episode_waiting_time(env)
            ms = episode_makespan(env)
            title = (
                f"{'HGT Policy' if mode == 'hgt' else 'iGreedy'}  —  "
                f"{args.difficulty} ({n} vehicles)  |  "
                f"wt={wt:.2f}s  mkspan={ms:.2f}s"
            )
            snapshots = build_snapshots(schedule, scenario)
            animate(snapshots, scenario, title, playback_speed=args.speed)


if __name__ == "__main__":
    main()
