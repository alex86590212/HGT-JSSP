"""Dump one scenario, its CP-SAT optimal schedule, and verify each constraint.

Prints the vehicles, the solved per-operation start/end times, and a PASS/FAIL
check for Type-1 (route order), Type-2 (same-lane order), Type-3 (zone
exclusivity), and arrival-time constraints. Lets you eyeball whether the
CP-SAT model actually enforces what it should.
"""

from __future__ import annotations

import argparse

from intersection_scheduler.data.scenario_generator import (
    _MANOEUVRE_TO_LANE,
    ScenarioGenerator,
)

from evaluation.optimal_solver import solve_optimal

EPS = 1e-6


def check_constraints(scenario, schedule) -> None:
    # Index schedule by (vehicle_id, route_position)
    by_op = {(s["vehicle_id"], s["route_position"]): s for s in schedule}

    passes = []
    fails = []

    def record(ok, msg):
        (passes if ok else fails).append(msg)

    # Type-1: route order within each vehicle
    for v in scenario.vehicles:
        for j in range(len(v.route) - 1):
            a = by_op[(v.id, j)]
            b = by_op[(v.id, j + 1)]
            ok = b["start"] >= a["end"] - EPS
            record(ok, f"T1 v{v.id} z{a['zone_id']}(end={a['end']:.3f}) -> "
                       f"z{b['zone_id']}(start={b['start']:.3f})")

    # Type-2: same-lane order on first operation
    by_lane = {}
    for v, m in zip(scenario.vehicles, scenario.manoeuvres):
        lane = _MANOEUVRE_TO_LANE.get(m)
        if lane is not None:
            by_lane.setdefault(lane, []).append(v)
    for lane, vs in by_lane.items():
        ordered = sorted(vs, key=lambda v: (v.arrival_time, v.id))
        for k in range(len(ordered) - 1):
            leader, follower = ordered[k], ordered[k + 1]
            la = by_op[(leader.id, 0)]
            fo = by_op[(follower.id, 0)]
            ok = fo["start"] >= la["end"] - EPS
            record(ok, f"T2 lane={lane} v{leader.id}(end={la['end']:.3f}) -> "
                       f"v{follower.id}(start={fo['start']:.3f})")

    # Type-3: zone exclusivity (no overlap in any zone)
    by_zone = {}
    for s in schedule:
        by_zone.setdefault(s["zone_id"], []).append(s)
    for zone_id, ops in by_zone.items():
        ops_sorted = sorted(ops, key=lambda s: s["start"])
        for k in range(len(ops_sorted) - 1):
            a, b = ops_sorted[k], ops_sorted[k + 1]
            ok = b["start"] >= a["end"] - EPS
            record(ok, f"T3 z{zone_id} v{a['vehicle_id']}(end={a['end']:.3f}) -> "
                       f"v{b['vehicle_id']}(start={b['start']:.3f})")

    # Arrival: first op starts no earlier than arrival
    for v in scenario.vehicles:
        s = by_op[(v.id, 0)]
        ok = s["start"] >= v.arrival_time - EPS
        record(ok, f"ARR v{v.id} arrival={v.arrival_time:.3f} start={s['start']:.3f}")

    print(f"\nConstraint checks: {len(passes)} PASS, {len(fails)} FAIL")
    if fails:
        print("FAILURES:")
        for f in fails:
            print(f"  [FAIL] {f}")
    else:
        print("  All constraints satisfied.")


def main():
    parser = argparse.ArgumentParser(description="Inspect one scenario's CP-SAT schedule")
    parser.add_argument("--tier", default="hard", choices=["easy", "medium", "hard"])
    parser.add_argument("--index", type=int, default=0,
                        help="Which scenario (0-based) in the seeded sequence")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time_limit", type=float, default=30.0)
    args = parser.parse_args()

    gen = ScenarioGenerator(seed=args.seed)
    scenario = None
    for _ in range(args.index + 1):
        scenario = getattr(gen, args.tier)()

    print(f"=== {args.tier} scenario #{args.index} (seed={args.seed}) ===")
    print(f"{len(scenario.vehicles)} vehicles")
    for v, m in zip(scenario.vehicles, scenario.manoeuvres):
        lane = _MANOEUVRE_TO_LANE.get(m, "?")
        print(f"  v{v.id}: {m:4s} lane={lane:5s} route={v.route}  "
              f"arrival={v.arrival_time:.3f}s  vel={v.velocity:.1f}m/s  "
              f"p={[round(x, 3) for x in v.processing_times]}")

    w_star, schedule = solve_optimal(
        scenario.vehicles, scenario.manoeuvres,
        time_limit_seconds=args.time_limit, return_schedule=True,
    )

    if w_star is None:
        print("\nNo optimal solution found within time limit.")
        return

    print(f"\nW* (mean waiting time) = {w_star:.4f}s")
    print("\nSolved schedule (sorted by start):")
    for s in sorted(schedule, key=lambda s: (s["start"], s["vehicle_id"])):
        print(f"  v{s['vehicle_id']} z{s['zone_id']} pos{s['route_position']}: "
              f"[{s['start']:.3f} -> {s['end']:.3f}]")

    check_constraints(scenario, schedule)


if __name__ == "__main__":
    main()
