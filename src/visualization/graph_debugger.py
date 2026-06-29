"""Visual debugging runner for the SUMO -> JSSP -> scheduler control loop.

Run this script to see two synchronized views:

1. `sumo-gui`, which shows traffic moving through the unsignalized intersection.
2. A live matplotlib window, which shows the current JSSP/timing-conflict graph.

The scheduler is still the simple FCFS policy from `run_traci_fcfs_controller`.
This visual layer exists to verify that the graph state matches what is visible
in SUMO. That check will be important before replacing FCFS with a PPO-GNN
scheduler, because a learned policy is only meaningful if its graph observation
faithfully represents the live intersection state.
"""

from __future__ import annotations

import argparse
import csv
import json
import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
except ModuleNotFoundError:
    plt = None
    Line2D = None

try:
    import networkx as nx
except ModuleNotFoundError:
    nx = None

from src.sumo_interface.traci_runner import (
    ControllerState,
    DEFAULT_CONFLICT_ZONE_CLEARANCE_SECONDS,
    DEFAULT_RELEASE_LOOKAHEAD_SECONDS,
    DEFAULT_SAME_LANE_CLEARANCE_SECONDS,
    DEFAULT_SUMO_CONFIG,
    annotate_graph_with_schedule,
    apply_sumo_control,
    build_jssp_graph,
    close_traci_connection,
    edge_type_counts,
    get_approaching_vehicles,
    jsonable_graph_step,
    print_debug_step,
    require_dependencies,
    resolve_sumo_binary,
    scheduler_from_args,
    traci,
)


DEFAULT_OUTPUT_DIR = Path("outputs") / "visual_debug_controller"

ZONE_COLORS = {
    "z1": "#4c78a8",
    "z2": "#f58518",
    "z3": "#54a24b",
    "z4": "#b279a2",
}
TYPE1_COLOR = "#2f2f2f"
TYPE2_COLOR = "#1f77b4"
TYPE3_COLOR = "#d62728"
ACTIVE_COLOR = "#f2c94c"
STOPPED_COLOR = "#d94f45"
RELEASED_COLOR = "#2ca02c"
DEFAULT_NODE_EDGE = "#2f2f2f"


class InteractiveRunState:
    """Controller-owned pause/step state.

    With TraCI, the Python client is the component that advances SUMO. These
    controls pause that client instead of relying on the sumo-gui toolbar, so
    sumo-gui remains an ordinary responsive visualization window while the
    controller is paused or single-stepped.
    """

    def __init__(self, start_paused: bool = False) -> None:
        self._lock = threading.Lock()
        self.paused = start_paused
        self.single_step_requested = False
        self.quit_requested = False

    def toggle_pause(self) -> None:
        with self._lock:
            self.paused = not self.paused
            state = "paused" if self.paused else "running"
        print(f"Controller {state}.")

    def request_single_step(self) -> None:
        with self._lock:
            self.paused = True
            self.single_step_requested = True
        print("Controller single-step requested.")

    def request_quit(self) -> None:
        with self._lock:
            self.quit_requested = True
        print("Controller quit requested.")

    def consume_single_step_request(self) -> bool:
        with self._lock:
            if self.single_step_requested:
                self.single_step_requested = False
                return True
            return False

    def snapshot(self) -> Tuple[bool, bool, bool]:
        with self._lock:
            return self.paused, self.single_step_requested, self.quit_requested


def require_visual_dependencies() -> None:
    if nx is None:
        raise SystemExit(
            "Missing dependency: networkx. Install dependencies with "
            "`python -m pip install -r requirements.txt`."
        )
    if plt is None or Line2D is None:
        raise SystemExit(
            "Missing dependency: matplotlib. Install dependencies with "
            "`python -m pip install -r requirements.txt`."
        )


def vehicle_order_from_schedule(vehicle_records: Sequence[Any], schedule: Sequence[str]) -> List[str]:
    """Return a stable row order for graph drawing."""

    if hasattr(schedule, "vehicle_order"):
        schedule = schedule.vehicle_order

    seen = set()
    ordered = []
    for vehicle_id in schedule:
        if vehicle_id not in seen:
            ordered.append(vehicle_id)
            seen.add(vehicle_id)

    remaining = sorted(
        [record for record in vehicle_records if record.vehicle_id not in seen],
        key=lambda record: (
            record.estimated_arrival_time,
            record.first_detected_time,
            record.source_lane,
            record.vehicle_id,
        ),
    )
    for record in remaining:
        ordered.append(record.vehicle_id)
    return ordered


def graph_positions(
    graph: Any,
    vehicle_records: Sequence[Any],
    schedule: Sequence[str],
) -> Dict[str, Tuple[float, float]]:
    """Place each vehicle on one row and operations in route order."""

    ordered_vehicle_ids = vehicle_order_from_schedule(vehicle_records, schedule)
    row_by_vehicle = {
        vehicle_id: -float(index) for index, vehicle_id in enumerate(ordered_vehicle_ids)
    }

    positions: Dict[str, Tuple[float, float]] = {}
    for node_id, attrs in graph.nodes(data=True):
        positions[node_id] = (
            float(attrs["operation_index"]),
            row_by_vehicle.get(attrs["vehicle_id"], 0.0),
        )
    return positions


def edge_lists(graph: Any, graph_records: Dict[str, Any]) -> Dict[str, List[Tuple[str, str]]]:
    """Split graph edges into the three visual edge classes."""

    type1 = []
    type2 = []
    for source, target, attrs in graph.edges(data=True):
        if attrs["edge_type"] == "type1_precedence":
            type1.append((source, target))
        elif attrs["edge_type"] == "type2_same_lane_order":
            type2.append((source, target))

    # Type-3 conflict candidates are stored in the graph as paired directed
    # alternatives. For visual debugging, draw one undirected dotted line per
    # candidate pair so the window stays readable.
    type3 = [(edge.source, edge.target) for edge in graph_records["type3_edges"]]
    return {"type1": type1, "type2": type2, "type3": type3}


class LiveJSSPGraphVisualizer:
    """Small matplotlib view for the live timing-conflict graph."""

    def __init__(
        self,
        pause_seconds: float,
        interactive_state: InteractiveRunState,
    ) -> None:
        require_visual_dependencies()
        self.pause_seconds = pause_seconds
        self.interactive_state = interactive_state

        plt.ion()
        self.figure = plt.figure(figsize=(15, 8))
        if hasattr(self.figure.canvas.manager, "set_window_title"):
            self.figure.canvas.manager.set_window_title("Live JSSP Graph Debugger")

        grid = self.figure.add_gridspec(1, 2, width_ratios=[3.0, 1.15])
        self.graph_ax = self.figure.add_subplot(grid[0, 0])
        self.panel_ax = self.figure.add_subplot(grid[0, 1])
        self.figure.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.figure.canvas.mpl_connect("close_event", self.on_close)

    def on_key_press(self, event: Any) -> None:
        key = (event.key or "").lower()
        if key in {" ", "space"}:
            self.interactive_state.toggle_pause()
        elif key in {"n", "right"}:
            self.interactive_state.request_single_step()
        elif key in {"r", "enter"}:
            is_paused, _, _ = self.interactive_state.snapshot()
            if is_paused:
                self.interactive_state.toggle_pause()
        elif key in {"q", "escape"}:
            self.interactive_state.request_quit()

    def on_close(self, _event: Any) -> None:
        self.interactive_state.request_quit()

    def is_open(self) -> bool:
        return bool(plt.fignum_exists(self.figure.number))

    def update(
        self,
        sim_time: float,
        graph: Any,
        graph_records: Dict[str, Any],
        vehicle_records: Sequence[Any],
        schedule: Sequence[str],
        control_result: Dict[str, Any],
    ) -> None:
        """Redraw the graph and debug panel for the current decision step."""

        if not self.is_open():
            return

        self.graph_ax.clear()
        self.panel_ax.clear()

        self.draw_graph(
            sim_time=sim_time,
            graph=graph,
            graph_records=graph_records,
            vehicle_records=vehicle_records,
            schedule=schedule,
            control_result=control_result,
        )
        self.update_debug_panel(
            sim_time=sim_time,
            graph=graph,
            vehicle_records=vehicle_records,
            schedule=schedule,
            control_result=control_result,
        )

        self.figure.tight_layout()
        self.figure.canvas.draw_idle()

    def run(
        self,
        snapshots: "queue.Queue[Dict[str, Any]]",
        worker_done: threading.Event,
    ) -> None:
        """Run the matplotlib event loop on the main thread.

        All graph drawing stays in this thread. The TraCI worker only pushes
        immutable-ish snapshots into the queue, which avoids TkAgg reentrancy
        problems while the user drags, resizes, pauses, or steps the window.
        """

        interval_ms = max(20, int(max(self.pause_seconds, 0.02) * 1000))

        def drain_snapshots() -> bool:
            latest = None
            while True:
                try:
                    latest = snapshots.get_nowait()
                except queue.Empty:
                    break

            if latest is not None:
                if "error" in latest:
                    self.graph_ax.clear()
                    self.panel_ax.clear()
                    self.graph_ax.text(
                        0.5,
                        0.5,
                        latest["error"],
                        ha="center",
                        va="center",
                        wrap=True,
                        transform=self.graph_ax.transAxes,
                    )
                    self.graph_ax.axis("off")
                    self.panel_ax.axis("off")
                    self.figure.canvas.draw_idle()
                else:
                    self.update(
                        sim_time=latest["sim_time"],
                        graph=latest["graph"],
                        graph_records=latest["graph_records"],
                        vehicle_records=latest["vehicle_records"],
                        schedule=latest["schedule"],
                        control_result=latest["control_result"],
                    )

            if worker_done.is_set() and snapshots.empty():
                return self.is_open()
            return self.is_open()

        timer = self.figure.canvas.new_timer(interval=interval_ms)
        timer.add_callback(drain_snapshots)
        timer.start()
        plt.show(block=True)

    def draw_graph(
        self,
        sim_time: float,
        graph: Any,
        graph_records: Dict[str, Any],
        vehicle_records: Sequence[Any],
        schedule: Sequence[str],
        control_result: Dict[str, Any],
    ) -> None:
        allowed_vehicles = set(control_result.get("allowed_vehicle_ids", []))
        active_vehicle = control_result.get("allowed_vehicle_id")
        if active_vehicle:
            allowed_vehicles.add(active_vehicle)
        stopped_vehicles = set(control_result.get("currently_stopped", []))
        released_now = set(control_result.get("released_now", []))

        self.graph_ax.set_title(
            f"Live JSSP Timing-Conflict Graph, t={sim_time:.1f}s",
            fontsize=13,
        )

        if graph.number_of_nodes() == 0:
            self.graph_ax.text(
                0.5,
                0.5,
                "No approaching vehicles detected",
                ha="center",
                va="center",
                fontsize=12,
                transform=self.graph_ax.transAxes,
            )
            self.graph_ax.axis("off")
            self.draw_legend()
            return

        positions = graph_positions(graph, vehicle_records, schedule)
        edges = edge_lists(graph, graph_records)

        node_colors = []
        node_edge_colors = []
        node_widths = []
        node_sizes = []
        labels = {}

        for node_id, attrs in graph.nodes(data=True):
            vehicle_id = attrs["vehicle_id"]
            node_colors.append(ZONE_COLORS.get(attrs["conflict_zone"], "#999999"))
            timing_label = ""
            if "reservation_enter_time" in attrs and "reservation_exit_time" in attrs:
                timing_label = (
                    f"\n{attrs['reservation_enter_time']:.1f}-"
                    f"{attrs['reservation_exit_time']:.1f}s"
                )
            labels[node_id] = (
                f"{vehicle_id}\n"
                f"{attrs['conflict_zone']}  {attrs['vehicle_type']}\n"
                f"p={attrs['processing_time']:.1f}s"
                f"{timing_label}"
            )

            if vehicle_id in allowed_vehicles:
                node_edge_colors.append(ACTIVE_COLOR)
                node_widths.append(4.0)
                node_sizes.append(2350)
            elif vehicle_id in released_now:
                node_edge_colors.append(RELEASED_COLOR)
                node_widths.append(3.2)
                node_sizes.append(2200)
            elif vehicle_id in stopped_vehicles:
                node_edge_colors.append(STOPPED_COLOR)
                node_widths.append(3.2)
                node_sizes.append(2200)
            else:
                node_edge_colors.append(DEFAULT_NODE_EDGE)
                node_widths.append(1.2)
                node_sizes.append(1900)

        nx.draw_networkx_edges(
            graph,
            positions,
            edgelist=edges["type1"],
            edge_color=TYPE1_COLOR,
            arrows=True,
            arrowstyle="-|>",
            width=2.0,
            connectionstyle="arc3,rad=0.05",
            ax=self.graph_ax,
        )
        nx.draw_networkx_edges(
            graph,
            positions,
            edgelist=edges["type2"],
            edge_color=TYPE2_COLOR,
            arrows=True,
            arrowstyle="-|>",
            style="dashed",
            width=1.8,
            connectionstyle="arc3,rad=0.18",
            ax=self.graph_ax,
        )
        nx.draw_networkx_edges(
            graph,
            positions,
            edgelist=edges["type3"],
            edge_color=TYPE3_COLOR,
            arrows=False,
            style="dotted",
            width=1.5,
            alpha=0.75,
            ax=self.graph_ax,
        )
        nx.draw_networkx_nodes(
            graph,
            positions,
            node_color=node_colors,
            node_size=node_sizes,
            edgecolors=node_edge_colors,
            linewidths=node_widths,
            ax=self.graph_ax,
        )
        nx.draw_networkx_labels(
            graph,
            positions,
            labels=labels,
            font_size=7,
            font_color="white",
            ax=self.graph_ax,
        )

        self.graph_ax.margins(x=0.20, y=0.18)
        self.graph_ax.axis("off")
        self.draw_legend()

    def draw_legend(self) -> None:
        legend_items = [
            Line2D([0], [0], color=TYPE1_COLOR, lw=2.2, label="Type-1 route order"),
            Line2D([0], [0], color=TYPE2_COLOR, lw=2.2, ls="--", label="Type-2 lane order"),
            Line2D([0], [0], color=TYPE3_COLOR, lw=2.2, ls=":", label="Type-3 conflict"),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#999999",
                markeredgecolor=ACTIVE_COLOR,
                markeredgewidth=3.0,
                markersize=10,
                label="Scheduled/released",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#999999",
                markeredgecolor=STOPPED_COLOR,
                markeredgewidth=3.0,
                markersize=10,
                label="Stopped/waiting",
            ),
        ]
        self.graph_ax.legend(handles=legend_items, loc="upper left", frameon=True, fontsize=8)

    def update_debug_panel(
        self,
        sim_time: float,
        graph: Any,
        vehicle_records: Sequence[Any],
        schedule: Sequence[str],
        control_result: Dict[str, Any],
    ) -> None:
        self.panel_ax.axis("off")
        edge_counts = edge_type_counts(graph)

        schedule_lines = [
            f"{index + 1}. {vehicle_id}"
            for index, vehicle_id in enumerate(vehicle_order_from_schedule(vehicle_records, schedule)[:12])
        ]
        schedule_ids = vehicle_order_from_schedule(vehicle_records, schedule)
        if len(schedule_ids) > 12:
            schedule_lines.append(f"... +{len(schedule_ids) - 12} more")

        reservation_lines = []
        if hasattr(schedule, "operation_reservations"):
            for reservation in schedule.operation_reservations[:12]:
                reservation_lines.append(
                    f"{reservation.vehicle_id:>8} {reservation.conflict_zone} "
                    f"{reservation.enter_time:5.1f}-{reservation.exit_time:5.1f}"
                )
            if len(schedule.operation_reservations) > 12:
                reservation_lines.append(
                    f"... +{len(schedule.operation_reservations) - 12} more"
                )

        detected_lines = []
        for record in vehicle_records[:12]:
            detected_lines.append(
                f"{record.vehicle_id:>8} {record.route_id:<6} "
                f"d={record.distance_to_intersection:5.1f} "
                f"v={record.speed:4.1f} eta={record.estimated_arrival_time:5.1f}"
            )
        if len(vehicle_records) > 12:
            detected_lines.append(f"... +{len(vehicle_records) - 12} more")

        is_paused, _, _ = self.interactive_state.snapshot()
        panel_text = (
            "Live Debug State\n"
            "----------------\n"
            f"controller: {'PAUSED' if is_paused else 'RUNNING'}\n"
            "TraCI clock keys: Space pause/resume, N step, Q quit\n"
            f"time: {sim_time:.1f}s\n"
            f"detected vehicles: {len(vehicle_records)}\n"
            f"graph nodes: {graph.number_of_nodes()}\n"
            f"graph edges: {graph.number_of_edges()}\n"
            f"T1/T2/T3: "
            f"{edge_counts.get('type1_precedence', 0)}/"
            f"{edge_counts.get('type2_same_lane_order', 0)}/"
            f"{edge_counts.get('type3_conflict_candidate', 0)}\n"
            f"allowed: {', '.join(control_result.get('allowed_vehicle_ids', [])) or '-'}\n"
            f"active: {', '.join(control_result.get('active_vehicle_ids', [])) or '-'}\n"
            f"stopped: {', '.join(control_result.get('currently_stopped', [])) or '-'}\n"
            f"released now: {', '.join(control_result.get('released_now', [])) or '-'}\n"
            "\nFCFS order\n"
            "----------\n"
            f"{chr(10).join(schedule_lines) or '-'}\n"
            "\nReservations\n"
            "------------\n"
            f"{chr(10).join(reservation_lines) or '-'}\n"
            "\nDetected vehicles\n"
            "-----------------\n"
            f"{chr(10).join(detected_lines) or '-'}"
        )

        self.panel_ax.text(
            0.0,
            1.0,
            panel_text,
            ha="left",
            va="top",
            family="monospace",
            fontsize=9,
            transform=self.panel_ax.transAxes,
        )


def update_debug_panel(
    visualizer: Optional[LiveJSSPGraphVisualizer],
    sim_time: float,
    graph: Any,
    vehicle_records: Sequence[Any],
    schedule: Sequence[str],
    control_result: Dict[str, Any],
) -> None:
    """Small wrapper matching the suggested interface in the task."""

    if visualizer is not None:
        visualizer.update_debug_panel(
            sim_time=sim_time,
            graph=graph,
            vehicle_records=vehicle_records,
            schedule=schedule,
            control_result=control_result,
        )


def visualize_jssp_graph(
    visualizer: Optional[LiveJSSPGraphVisualizer],
    sim_time: float,
    graph: Any,
    graph_records: Dict[str, Any],
    vehicle_records: Sequence[Any],
    schedule: Sequence[str],
    stopped_vehicles: Sequence[str],
    released_vehicles: Sequence[str],
    active_vehicle_id: Optional[str],
) -> None:
    """Update the live graph window.

    The parameters are intentionally scheduler-agnostic. A PPO-GNN policy can
    later pass its selected vehicle and action metadata here instead of FCFS.
    """

    if visualizer is None:
        return

    control_result = {
        "allowed_vehicle_id": active_vehicle_id,
        "currently_stopped": list(stopped_vehicles),
        "released_now": list(released_vehicles),
    }
    visualizer.update(
        sim_time=sim_time,
        graph=graph,
        graph_records=graph_records,
        vehicle_records=vehicle_records,
        schedule=schedule,
        control_result=control_result,
    )


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        json.dump(payload, handle)
        handle.write("\n")


def pump_ui(
    visualizer: Optional[LiveJSSPGraphVisualizer],
    seconds: float,
) -> None:
    """Yield wall-clock time from the TraCI worker.

    The matplotlib event loop runs on the main thread. The worker must not call
    `plt.pause()` or any TkAgg drawing function; it only sleeps between SUMO
    steps so sumo-gui has time to process native window events.
    """

    delay = max(0.001, seconds)
    time.sleep(delay)


def wait_until_step_allowed(
    interactive_state: InteractiveRunState,
    visualizer: Optional[LiveJSSPGraphVisualizer],
    idle_sleep: float,
) -> bool:
    """Block simulation stepping while the controller is paused.

    Returns False when the user requested quit. While paused, the function keeps
    pumping matplotlib events so the graph window can receive Space/N/Q keys.
    """

    while True:
        is_paused, single_step_requested, quit_requested = interactive_state.snapshot()
        if quit_requested:
            return False
        if not is_paused or single_step_requested:
            break
        pump_ui(visualizer, idle_sleep)

    interactive_state.consume_single_step_request()
    _, _, quit_requested = interactive_state.snapshot()
    return not quit_requested


def pace_after_sumo_step(
    step_seconds: float,
    realtime_factor: float,
    interactive_state: InteractiveRunState,
    visualizer: Optional[LiveJSSPGraphVisualizer],
    idle_sleep: float,
) -> None:
    """Run the TraCI client at a human-debuggable wall-clock pace."""

    if realtime_factor <= 0:
        return

    deadline = time.monotonic() + (step_seconds / realtime_factor)
    while time.monotonic() < deadline:
        _, _, quit_requested = interactive_state.snapshot()
        if quit_requested:
            return
        remaining = deadline - time.monotonic()
        pump_ui(visualizer, min(idle_sleep, max(0.001, remaining)))


def write_csv_summary(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "time",
                "detected_count",
                "node_count",
                "directed_edge_count",
                "type1_edges",
                "type2_edges",
                "type3_edges",
                "active_vehicle",
                "allowed_vehicles",
                "active_vehicles",
                "reservation_count",
                "fcfs_order",
                "stopped_now",
                "released_now",
                "currently_stopped",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def publish_snapshot(
    snapshots: Optional["queue.Queue[Dict[str, Any]]"],
    payload: Dict[str, Any],
) -> None:
    """Publish the newest UI snapshot without blocking the TraCI worker."""

    if snapshots is None:
        return

    while True:
        try:
            snapshots.put_nowait(payload)
            return
        except queue.Full:
            try:
                snapshots.get_nowait()
            except queue.Empty:
                return


def build_sumo_command(args: argparse.Namespace) -> List[str]:
    sumo_binary = resolve_sumo_binary(gui=not args.headless_sumo, explicit_binary=args.sumo_binary)
    sumo_cmd = [
        sumo_binary,
        "-c",
        str(args.config),
        "--duration-log.disable",
        "true",
    ]
    if not args.headless_sumo and not args.no_gui_start:
        sumo_cmd.append("--start")
    if not args.headless_sumo and args.gui_delay >= 0:
        sumo_cmd.extend(["--delay", f"{args.gui_delay:.1f}"])
    if getattr(args, "seed", None) is not None:
        sumo_cmd.extend(["--seed", str(args.seed)])
    if args.no_warnings:
        sumo_cmd.extend(["--no-warnings", "true"])
    return sumo_cmd


def run_controller_loop(
    args: argparse.Namespace,
    interactive_state: InteractiveRunState,
    snapshots: Optional["queue.Queue[Dict[str, Any]]"] = None,
) -> None:
    """Run the TraCI/SUMO controller loop.

    This function intentionally contains no matplotlib calls. In visual mode it
    runs in a background thread and publishes snapshots to the main-thread GUI.
    """

    sumo_cmd = build_sumo_command(args)
    jsonl_path = args.output_dir / "visual_debug_steps.jsonl"
    csv_path = args.output_dir / "visual_debug_summary.csv"
    if not args.no_logs:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path.write_text("", encoding="utf-8")

    state = ControllerState(
        first_eta_by_vehicle={},
        first_detected_by_vehicle={},
        stopped_vehicles=set(),
        released_vehicles=set(),
        cleared_vehicles=set(),
        warned_unmapped_routes=set(),
    )
    csv_rows: List[Dict[str, Any]] = []
    scheduler = scheduler_from_args(args)

    print(f"Starting visual debug controller with: {' '.join(sumo_cmd)}")
    traci.start(sumo_cmd)
    try:
        next_decision_time = 0.0
        step_seconds = traci.simulation.getDeltaT()
        while traci.simulation.getMinExpectedNumber() > 0:
            if not wait_until_step_allowed(
                interactive_state=interactive_state,
                visualizer=None,
                idle_sleep=args.idle_sleep,
            ):
                break

            traci.simulationStep()
            sim_time = traci.simulation.getTime()
            if args.until is not None and sim_time > args.until:
                break

            vehicle_records = get_approaching_vehicles(
                sim_time=sim_time,
                state=state,
                detection_distance=args.detection_distance,
                min_eta_speed=args.min_eta_speed,
            )
            graph, graph_records = build_jssp_graph(vehicle_records)
            schedule = scheduler.schedule(
                vehicle_records,
                graph,
                sim_time,
            )
            control_result = apply_sumo_control(
                vehicle_records=vehicle_records,
                schedule=schedule,
                state=state,
                stop_distance=args.stop_distance,
                hold_distance=args.hold_distance,
                sim_time=sim_time,
                release_lookahead=args.release_lookahead,
            )

            should_update = sim_time + 1e-9 >= next_decision_time
            if should_update:
                annotate_graph_with_schedule(graph, schedule)
                print_debug_step(
                    sim_time=sim_time,
                    vehicle_records=vehicle_records,
                    graph=graph,
                    graph_records=graph_records,
                    schedule=schedule,
                    control_result=control_result,
                )
                publish_snapshot(
                    snapshots,
                    {
                        "sim_time": sim_time,
                        "graph": graph,
                        "graph_records": graph_records,
                        "vehicle_records": vehicle_records,
                        "schedule": schedule,
                        "control_result": control_result,
                    },
                )

                if not args.no_logs:
                    step_record = jsonable_graph_step(
                        sim_time=sim_time,
                        vehicle_records=vehicle_records,
                        graph=graph,
                        graph_records=graph_records,
                        schedule=schedule,
                        control_result=control_result,
                    )
                    append_jsonl(jsonl_path, step_record)
                    edge_counts = edge_type_counts(graph)
                    csv_rows.append(
                        {
                            "time": f"{sim_time:.1f}",
                            "detected_count": len(vehicle_records),
                            "node_count": graph.number_of_nodes(),
                            "directed_edge_count": graph.number_of_edges(),
                            "type1_edges": edge_counts.get("type1_precedence", 0),
                            "type2_edges": edge_counts.get("type2_same_lane_order", 0),
                            "type3_edges": edge_counts.get("type3_conflict_candidate", 0),
                            "active_vehicle": control_result["allowed_vehicle_id"] or "",
                            "allowed_vehicles": " ".join(control_result["allowed_vehicle_ids"]),
                            "active_vehicles": " ".join(control_result["active_vehicle_ids"]),
                            "reservation_count": len(schedule.operation_reservations),
                            "fcfs_order": " ".join(schedule.vehicle_order),
                            "stopped_now": " ".join(control_result["stopped_now"]),
                            "released_now": " ".join(control_result["released_now"]),
                            "currently_stopped": " ".join(control_result["currently_stopped"]),
                        }
                    )

                next_decision_time = sim_time + args.decision_period

            pace_after_sumo_step(
                step_seconds=step_seconds,
                realtime_factor=args.realtime_factor,
                interactive_state=interactive_state,
                visualizer=None,
                idle_sleep=args.idle_sleep,
            )
    finally:
        close_traci_connection()

    if not args.no_logs:
        write_csv_summary(csv_path, csv_rows)
        print(f"Wrote JSONL visual debug log: {jsonl_path}")
        print(f"Wrote CSV visual debug summary: {csv_path}")


def run_controller_with_visualization(args: argparse.Namespace) -> None:
    """Run SUMO, FCFS control, terminal logging, and live graph visualization."""

    require_dependencies()
    interactive_state = InteractiveRunState(start_paused=args.start_paused)

    if args.no_graph_window:
        run_controller_loop(args, interactive_state, snapshots=None)
        return

    require_visual_dependencies()
    snapshots: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=8)
    worker_done = threading.Event()
    visualizer = LiveJSSPGraphVisualizer(
        pause_seconds=args.pause,
        interactive_state=interactive_state,
    )

    # Keep matplotlib/Tk on the main thread. Running TraCI in the same thread
    # works for batch output but causes Windows to mark the graph window as
    # unresponsive while the user drags, resizes, pauses, or single-steps.
    print("Graph window controls: Space=pause/resume, N=single step while paused, Q=quit.")
    print("Use these controls to pause the TraCI clock; sumo-gui remains the traffic view.")
    print("The graph window owns the GUI event loop, so dragging/resizing it stays responsive.")

    def worker_main() -> None:
        try:
            run_controller_loop(args, interactive_state, snapshots=snapshots)
        except Exception as exc:
            publish_snapshot(snapshots, {"error": f"Controller worker failed:\n{exc!r}"})
            raise
        finally:
            worker_done.set()

    worker = threading.Thread(target=worker_main, name="traci-controller", daemon=True)
    worker.start()
    try:
        visualizer.run(snapshots=snapshots, worker_done=worker_done)
    finally:
        interactive_state.request_quit()
        worker.join(timeout=10.0)
        if worker.is_alive():
            print("Warning: TraCI controller worker did not stop within 10 seconds.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SUMO GUI plus a live JSSP graph debugger."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_SUMO_CONFIG,
        help=f"SUMO config path (default: {DEFAULT_SUMO_CONFIG})",
    )
    parser.add_argument(
        "--headless-sumo",
        action="store_true",
        help="Use headless sumo instead of sumo-gui. Useful for automated checks.",
    )
    parser.add_argument(
        "--sumo-binary",
        default=None,
        help="Explicit SUMO binary path.",
    )
    parser.add_argument(
        "--no-gui-start",
        action="store_true",
        help="Do not pass --start to sumo-gui. Most TraCI GUI runs should keep the default.",
    )
    parser.add_argument(
        "--gui-delay",
        type=float,
        default=40.0,
        help="Delay in ms passed to sumo-gui with --delay. Use -1 to disable.",
    )
    parser.add_argument(
        "--start-paused",
        action="store_true",
        help="Open both windows but do not advance SUMO until Space or N is pressed.",
    )
    parser.add_argument(
        "--until",
        type=float,
        default=120.0,
        help="Stop after this simulation time in seconds. Use -1 for full config duration.",
    )
    parser.add_argument(
        "--detection-distance",
        type=float,
        default=180.0,
        help="Detect inbound vehicles within this many meters of the intersection.",
    )
    parser.add_argument(
        "--hold-distance",
        type=float,
        default=90.0,
        help="Issue stop commands to non-selected vehicles within this distance.",
    )
    parser.add_argument(
        "--stop-distance",
        type=float,
        default=8.0,
        help="Place the hold stop this many meters before the intersection.",
    )
    parser.add_argument(
        "--min-eta-speed",
        type=float,
        default=0.1,
        help="Minimum speed used for first ETA estimates.",
    )
    parser.add_argument(
        "--decision-period",
        type=float,
        default=1.0,
        help="Seconds between graph/debug updates.",
    )
    parser.add_argument(
        "--conflict-zone-clearance",
        type=float,
        default=DEFAULT_CONFLICT_ZONE_CLEARANCE_SECONDS,
        help="Safety clearance in seconds after a conflict-zone operation.",
    )
    parser.add_argument(
        "--same-lane-clearance",
        type=float,
        default=DEFAULT_SAME_LANE_CLEARANCE_SECONDS,
        help="FIFO headway in seconds between same-lane vehicle entries.",
    )
    parser.add_argument(
        "--release-lookahead",
        type=float,
        default=DEFAULT_RELEASE_LOOKAHEAD_SECONDS,
        help="Release vehicles this many seconds before their first reserved zone entry.",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=0.05,
        help="matplotlib pause duration after each visual update.",
    )
    parser.add_argument(
        "--realtime-factor",
        type=float,
        default=None,
        help=(
            "Wall-clock pacing factor. 1.0 means about real time, 2.0 is 2x, "
            "0 disables pacing. Default is 1.0 with windows and 0 with --no-graph-window."
        ),
    )
    parser.add_argument(
        "--idle-sleep",
        type=float,
        default=0.02,
        help="Small wall-clock sleep used while paused and while pacing UI events.",
    )
    parser.add_argument(
        "--no-graph-window",
        action="store_true",
        help="Disable the matplotlib graph window and keep terminal/log output only.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for JSONL/CSV logs (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--no-logs",
        action="store_true",
        help="Disable JSONL and CSV visual-debug logs.",
    )
    parser.add_argument(
        "--no-warnings",
        action="store_true",
        help="Suppress SUMO warning output.",
    )
    args = parser.parse_args()
    if args.until is not None and args.until < 0:
        args.until = None
    if args.decision_period <= 0:
        raise ValueError("--decision-period must be positive")
    if args.conflict_zone_clearance < 0:
        raise ValueError("--conflict-zone-clearance must be non-negative")
    if args.same_lane_clearance < 0:
        raise ValueError("--same-lane-clearance must be non-negative")
    if args.release_lookahead < 0:
        raise ValueError("--release-lookahead must be non-negative")
    if args.pause < 0:
        raise ValueError("--pause must be non-negative")
    if args.realtime_factor is None:
        args.realtime_factor = 0.0 if args.no_graph_window else 1.0
    if args.realtime_factor < 0:
        raise ValueError("--realtime-factor must be non-negative")
    if args.idle_sleep <= 0:
        raise ValueError("--idle-sleep must be positive")
    if args.start_paused and args.no_graph_window:
        parser.error("--start-paused requires the graph window for keyboard controls")
    return args


if __name__ == "__main__":
    run_controller_with_visualization(parse_args())
