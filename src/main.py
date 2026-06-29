"""Canonical experiment entrypoint for the FCFS intersection pipeline."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from src.config.load_config import (
    load_experiment_config,
    load_intersection_config,
    load_vehicle_types,
    resolve_project_path,
)
from src.evaluation.episode_logger import EpisodeLogger
from src.evaluation.metrics import compute_episode_metrics, write_metrics_summary
from src.evaluation.safety_validator import SafetyValidator
from src.graph.graph_validator import validate_jssp_graph, validate_route_mapping
from src.graph.jssp_graph_builder import build_jssp_graph
from src.schedulers.fcfs_scheduler import FCFSScheduler
from src.schedulers.greedy_scheduler import GreedyScheduler
from src.sumo_interface.traci_runner import (
    ControllerState,
    LiveVehicleRecord,
    TraCIRunner,
)
from src.visualization.graph_debugger import run_controller_with_visualization


def parse_seed_list(raw: str | Sequence[int]) -> List[int]:
    if isinstance(raw, str):
        return [int(part.strip()) for part in raw.split(",") if part.strip()]
    return [int(seed) for seed in raw]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run modular FCFS SUMO -> JSSP -> scheduler experiments."
    )
    parser.add_argument(
        "--mode",
        choices=["headless", "debug", "test-scenarios"],
        default=None,
        help="Run mode. Defaults to configs/experiment_config.json.",
    )
    parser.add_argument(
        "--scheduler",
        choices=["fcfs", "greedy"],
        default=None,
        help="Scheduler backend. Greedy is a placeholder and does not run yet.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs") / "experiment_config.json",
        help="Experiment JSON config path.",
    )
    parser.add_argument(
        "--sumo-config",
        type=Path,
        default=None,
        help="Override SUMO .sumocfg path.",
    )
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seeds", default=None, help="Comma-separated seed list, e.g. 1,2,3.")
    parser.add_argument("--until", type=float, default=None, help="Simulation horizon in seconds.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--log-steps",
        action="store_true",
        help="Write verbose JSONL controller step logs.",
    )
    parser.add_argument("--no-warnings", action="store_true", help="Suppress SUMO warnings.")
    return parser


def choose_scheduler(name: str, intersection_config: Dict[str, Any]):
    scheduling = intersection_config.get("scheduling", {})
    if name == "fcfs":
        return FCFSScheduler(
            conflict_zone_clearance=float(scheduling.get("conflict_zone_clearance_s", 1.0)),
            same_lane_clearance=float(scheduling.get("same_lane_clearance_s", 1.0)),
        )
    if name == "greedy":
        return GreedyScheduler()
    raise ValueError(f"Unknown scheduler: {name}")


def resolve_sumo_config(
    cli_sumo_config: Path | None,
    experiment_config: Dict[str, Any],
    intersection_config: Dict[str, Any],
) -> Path:
    if cli_sumo_config is not None:
        return resolve_project_path(cli_sumo_config)

    scenario = experiment_config.get("scenario", "baseline")
    sumo_configs = intersection_config.get("sumo_configs", {})
    if scenario not in sumo_configs:
        available = ", ".join(sorted(sumo_configs))
        raise SystemExit(f"Unknown SUMO scenario {scenario!r}. Available: {available}")
    return resolve_project_path(sumo_configs[scenario])


def controller_args_for_episode(
    *,
    sumo_config: Path,
    output_dir: Path,
    until: float | None,
    intersection_config: Dict[str, Any],
    experiment_config: Dict[str, Any],
    gui: bool,
    log_steps: bool,
    no_warnings: bool,
    safety_validator: Any = None,
    seed: int | None = None,
    scheduler: Any = None,
) -> argparse.Namespace:
    control = intersection_config.get("control", {})
    scheduling = intersection_config.get("scheduling", {})
    return argparse.Namespace(
        config=sumo_config,
        gui=gui,
        headless_sumo=not gui,
        no_gui_start=False,
        gui_delay=40.0,
        start_paused=False,
        sumo_binary=None,
        until=until,
        detection_distance=float(control.get("detection_distance_m", 180.0)),
        hold_distance=float(control.get("hold_distance_m", 90.0)),
        stop_distance=float(control.get("stop_distance_m", 8.0)),
        min_eta_speed=float(control.get("min_eta_speed_mps", 0.1)),
        decision_period=float(experiment_config.get("decision_period_s", 1.0)),
        conflict_zone_clearance=float(scheduling.get("conflict_zone_clearance_s", 1.0)),
        same_lane_clearance=float(scheduling.get("same_lane_clearance_s", 1.0)),
        release_lookahead=float(control.get("release_lookahead_s", 0.0)),
        realtime_factor=1.0 if gui else 0.0,
        idle_sleep=0.02,
        pause=0.05,
        no_graph_window=False,
        output_dir=output_dir,
        no_logs=not log_steps,
        no_warnings=no_warnings,
        safety_validator=safety_validator,
        junction_id=intersection_config.get("junction_id", "J0"),
        seed=seed,
        scheduler=scheduler,
    )


def vehicle_record(vehicle_id: str, route_id: str, eta: float, first_detected: float = 0.0) -> LiveVehicleRecord:
    source = route_id.split("_to_", maxsplit=1)[0]
    return LiveVehicleRecord(
        vehicle_id=vehicle_id,
        route_id=route_id,
        vehicle_type="passenger",
        lane_id=f"{source}_in_0",
        source_lane=f"{source}_in",
        source_direction=source,
        lane_position=150.0,
        speed=10.0,
        distance_to_intersection=50.0,
        estimated_arrival_time=eta,
        first_detected_time=first_detected,
    )


def deterministic_vehicle_records() -> List[LiveVehicleRecord]:
    return [
        vehicle_record("veh_disjoint_e", "E_to_W", 10.0),
        vehicle_record("veh_disjoint_w", "W_to_E", 10.0),
        vehicle_record("veh_same_lane", "E_to_N", 10.1),
        vehicle_record("veh_shared_s", "S_to_N", 10.2),
    ]


def write_schedule_as_zone_occupancy(logger: EpisodeLogger, schedule: Any, safety: SafetyValidator) -> List[Any]:
    for reservation in getattr(schedule, "operation_reservations", []):
        safety.record_interval(
            vehicle_id=reservation.vehicle_id,
            route_id=reservation.route_id,
            conflict_zone=reservation.conflict_zone,
            entry_time=reservation.enter_time,
            exit_time=reservation.exit_time,
        )
    violations = safety.safety_violations()
    safety.write_zone_occupancy_csv(logger.path("zone_occupancy.csv"))
    safety.write_violations_json(logger.path("safety_violations.json"), violations)
    return violations


def run_test_scenarios(
    *,
    output_root: Path,
    scheduler_name: str,
    intersection_config: Dict[str, Any],
) -> int:
    scheduler = choose_scheduler(scheduler_name, intersection_config)
    if isinstance(scheduler, GreedyScheduler):
        print("Greedy scheduler is not implemented yet.")
        return 2

    route_result = validate_route_mapping()
    if not route_result.valid:
        for error in route_result.errors:
            print(f"Route mapping error: {error}")
        return 1

    episode_id = "test_scenarios_fcfs"
    logger = EpisodeLogger(output_root, episode_id)
    records = deterministic_vehicle_records()
    graph, graph_records = build_jssp_graph(records)
    graph_result = validate_jssp_graph(graph)
    if not graph_result.valid:
        for error in graph_result.errors:
            print(f"Graph error: {error}")
        return 1

    schedule = scheduler.schedule(records, graph, current_time=0.0)
    safety = SafetyValidator()
    violations = write_schedule_as_zone_occupancy(logger, schedule, safety)
    logger.write_reservations(schedule)
    feasibility_report = getattr(schedule, "feasibility_report", [])
    logger.write_json("constraint_report.json", feasibility_report)
    report_rows = []
    for entry in feasibility_report:
        for node_id in entry.get("feasible_operations", []):
            report_rows.append(
                {
                    "stage": entry.get("stage", ""),
                    "vehicle_id": entry.get("vehicle_id", ""),
                    "node_id": node_id,
                    "feasible": True,
                    "reasons": "",
                }
            )
        for item in entry.get("infeasible_operations", []):
            report_rows.append(
                {
                    "stage": entry.get("stage", ""),
                    "vehicle_id": entry.get("vehicle_id", ""),
                    "node_id": item.get("node_id", ""),
                    "feasible": False,
                    "reasons": " ".join(item.get("reasons", [])),
                }
            )
        for item in entry.get("results", []):
            report_rows.append(
                {
                    "stage": entry.get("stage", ""),
                    "vehicle_id": entry.get("vehicle_id", ""),
                    "node_id": item.get("node_id", ""),
                    "feasible": item.get("feasible", False),
                    "reasons": " ".join(item.get("reasons", [])),
                }
            )
    logger.write_csv("constraint_report.csv", report_rows)
    logger.write_json(
        "test_scenarios_graph.json",
        {
            "nodes": [{"node_id": node_id, **attrs} for node_id, attrs in graph.nodes(data=True)],
            "edges": [
                {"source": source, "target": target, **attrs}
                for source, target, attrs in graph.edges(data=True)
            ],
            "graph_validation": asdict(graph_result),
            "graph_record_counts": {
                "operations": len(graph_records["operations"]),
                "type1_edges": len(graph_records["type1_edges"]),
                "type2_edges": len(graph_records["type2_edges"]),
                "type3_edges": len(graph_records["type3_edges"]),
            },
        },
    )
    metrics = compute_episode_metrics(
        episode_id=episode_id,
        sim_time=max((reservation.exit_time for reservation in schedule.operation_reservations), default=0.0),
        total_vehicles_completed=len(records),
        number_of_stops=0,
        scheduler_runtime_avg_ms=0.0,
        safety_violations=len(violations),
        sumo_collisions=0,
    )
    logger.write_metrics(metrics)
    write_metrics_summary(output_root / "metrics" / "summary.json", output_root / "metrics" / "summary.csv", [metrics])
    print(f"Deterministic test scenarios passed. Outputs: {logger.logs_dir}")
    return 0


def run_headless_batch(
    *,
    output_root: Path,
    sumo_config: Path,
    seeds: Sequence[int],
    episodes: int,
    until: float | None,
    log_steps: bool,
    no_warnings: bool,
    intersection_config: Dict[str, Any],
    experiment_config: Dict[str, Any],
    scheduler: Any,
) -> int:
    metrics_rows = []
    runner = TraCIRunner()
    for episode_index in range(episodes):
        seed = seeds[episode_index % len(seeds)] if seeds else episode_index
        episode_id = f"fcfs_seed_{seed}_episode_{episode_index:03d}"
        logger = EpisodeLogger(output_root, episode_id)
        safety = SafetyValidator(
            zone_half_size_m=float(
                intersection_config.get("conflict_zone_geometry", {}).get("zone_half_size_m", 8.0)
            ),
            overlap_tolerance_s=float(
                intersection_config.get("conflict_zone_geometry", {}).get("overlap_tolerance_s", 0.0)
            ),
        )
        controller_args = controller_args_for_episode(
            sumo_config=sumo_config,
            output_dir=logger.logs_dir,
            until=until,
            intersection_config=intersection_config,
            experiment_config=experiment_config,
            gui=False,
            log_steps=log_steps,
            no_warnings=no_warnings,
            safety_validator=safety,
            seed=seed,
            scheduler=scheduler,
        )
        result = runner.run(controller_args)
        violations = safety.safety_violations()
        safety.write_zone_occupancy_csv(logger.path("zone_occupancy.csv"))
        safety.write_violations_json(logger.path("safety_violations.json"), violations)
        logger.write_empty_standard_logs()

        metrics = compute_episode_metrics(
            episode_id=episode_id,
            sim_time=result.sim_time,
            total_vehicles_completed=result.vehicles_completed,
            number_of_stops=result.number_of_stops,
            scheduler_runtime_avg_ms=result.scheduler_runtime_avg_ms,
            safety_violations=len(violations),
            sumo_collisions=result.sumo_collisions,
            total_delay=result.total_delay,
            total_waiting_time=result.total_waiting_time,
            maximum_waiting_time=result.maximum_waiting_time,
            class_delay=result.class_delay,
        )
        logger.write_metrics(metrics)
        metrics_rows.append(metrics)
        print(
            f"Episode {episode_id}: completed={metrics.total_vehicles_completed} "
            f"throughput={metrics.throughput:.3f}/s "
            f"avg_delay={metrics.average_delay:.2f}s "
            f"stops={metrics.number_of_stops} "
            f"safety_violations={metrics.safety_violations} "
            f"sumo_collisions={metrics.sumo_collisions}"
        )

    write_metrics_summary(
        output_root / "metrics" / "summary.json",
        output_root / "metrics" / "summary.csv",
        metrics_rows,
    )
    print(f"Wrote batch metrics summary: {output_root / 'metrics' / 'summary.csv'}")
    return 0


def run_debug_mode(
    *,
    output_root: Path,
    sumo_config: Path,
    until: float | None,
    log_steps: bool,
    no_warnings: bool,
    intersection_config: Dict[str, Any],
    experiment_config: Dict[str, Any],
    scheduler: Any,
) -> int:
    logger = EpisodeLogger(output_root, "debug_fcfs")
    controller_args = controller_args_for_episode(
        sumo_config=sumo_config,
        output_dir=logger.logs_dir,
        until=until,
        intersection_config=intersection_config,
        experiment_config=experiment_config,
        gui=True,
        log_steps=log_steps,
        no_warnings=no_warnings,
        safety_validator=None,
        seed=None,
        scheduler=scheduler,
    )
    run_controller_with_visualization(controller_args)
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    experiment_config = load_experiment_config(args.config)
    intersection_config = load_intersection_config()
    load_vehicle_types()

    mode = args.mode or experiment_config.get("mode", "headless")
    scheduler_name = args.scheduler or experiment_config.get("scheduler", "fcfs")
    if scheduler_name == "greedy":
        print("Greedy scheduler is not implemented yet.")
        return 2
    scheduler = choose_scheduler(scheduler_name, intersection_config)

    output_root = resolve_project_path(args.output_dir or experiment_config.get("output_dir", "outputs"))
    sumo_config = resolve_sumo_config(args.sumo_config, experiment_config, intersection_config)
    episodes = args.episodes if args.episodes is not None else int(experiment_config.get("episodes", 1))
    seeds = parse_seed_list(args.seeds if args.seeds is not None else experiment_config.get("seeds", [42]))
    until = args.until if args.until is not None else experiment_config.get("until_s", 120.0)
    no_warnings = args.no_warnings or bool(experiment_config.get("no_warnings", False))
    log_steps = bool(args.log_steps or experiment_config.get("log_steps", False))

    if episodes <= 0:
        parser.error("--episodes must be positive")

    if mode == "test-scenarios":
        return run_test_scenarios(
            output_root=output_root,
            scheduler_name=scheduler_name,
            intersection_config=intersection_config,
        )
    if mode == "debug":
        return run_debug_mode(
            output_root=output_root,
            sumo_config=sumo_config,
            until=until,
            log_steps=log_steps,
            no_warnings=no_warnings,
            intersection_config=intersection_config,
            experiment_config=experiment_config,
            scheduler=scheduler,
        )
    if mode == "headless":
        return run_headless_batch(
            output_root=output_root,
            sumo_config=sumo_config,
            seeds=seeds,
            episodes=episodes,
            until=until,
            log_steps=log_steps,
            no_warnings=no_warnings,
            intersection_config=intersection_config,
            experiment_config=experiment_config,
            scheduler=scheduler,
        )

    parser.error(f"Unsupported mode: {mode}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
