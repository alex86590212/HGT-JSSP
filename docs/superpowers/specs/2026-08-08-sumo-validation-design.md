# SUMO Microsimulation Validation — Design

## Purpose

Every result produced this session (the corrected reward, the 5-method
comparison, the optimality-gap-to-W* numbers) was measured entirely inside
the custom `DynamicIntersectionEnv` — a lightweight, hand-built discrete-event
simulator. That's methodologically sound for comparing the five methods
against each other (all five see identical simplified physics), but it does
not demonstrate the ranking survives contact with a higher-fidelity,
third-party microsimulator with realistic vehicle dynamics.

This project builds a feasibility-proof pipeline that runs all five methods
(HGT, iGreedy, LIFO, Backpressure, EDF) through SUMO on the same scenario, and
reports each method's average per-vehicle waiting time as SUMO itself
measures it — not the custom env's number.

## Scope (this phase)

- **One scenario**, not a multi-scenario statistical study. The goal is to
  answer "does the HGT-vs-baselines ranking survive real SUMO physics" for at
  least one case, cheaply, before deciding whether a larger study is worth
  building.
- **No traffic signals.** This models a connected/autonomous-vehicle
  intersection where vehicles communicate with a central scheduler — there is
  no traffic-light layer to reconcile with. Right-of-way is resolved entirely
  by the scheduler's zone-reservation windows.
- **A new 4x4 grid SUMO network**, not the existing single-intersection one.
  The dynamic scheduler and all five baselines were trained/evaluated against
  a 16-zone 4x4 grid (`intersection_scheduler.data.scenario_generator_4x4.ZONE_POSITIONS`);
  validating against a single-intersection network would not be a like-for-like
  comparison.
- Explicitly out of scope for this phase: statistically powered multi-scenario
  runs, precise edge-length/speed calibration beyond a first-pass approximation,
  any change to the trained policy, the reward function, or the existing
  `dynamic_scheduler`/`evaluation` code.

## What "zone" means in SUMO terms

`DynamicZone` (in `dynamic_scheduler/environment/dynamic_intersection.py`) has
no geometry — it's an (x, y) point with an occupancy rule enforced by
`DynamicIntersectionEnv`: **a zone holds exactly one vehicle from
`start_time` to `earliest_finish` (= `start_time + processing_time`); no two
operations in the same zone overlap in time.**

The literal translation of that rule into SUMO: **one zone = one short SUMO
edge**, entry gated by that vehicle's reserved `start_time` for that zone
(from the scheduler's `op_timeline`). This is not an approximation of what was
evaluated — it is the same exclusivity rule, expressed in SUMO's primitives
instead of the custom env's.

## Architecture

```
sumo_validation/                          (new, top-level folder)
├── network/
│   ├── build_grid_network.py             generates the 4x4 grid .net.xml via
│   │                                       SUMO's netgenerate --grid tool,
│   │                                       strips traffic-light logic, sets
│   │                                       edge lengths/speeds calibrated
│   │                                       (first pass) against the scheduler's
│   │                                       processing_time values
│   ├── grid_4x4.net.xml                  generated output, checked in once stable
│   └── grid_4x4.sumocfg
├── translate.py                          op_timeline -> reservation-window
│                                           schedule (self-contained shape,
│                                           see Data flow)
├── sumo_control.py                       self-contained TraCI stop/release
│                                           loop (grid-specific; does not
│                                           import src/sumo_interface)
├── run_comparison.py                     entry point (see Data flow below)
└── README.md                             how to run headless vs --gui, what a
                                            "zone" means here, known limitations
```

New code imports, unchanged, from:
- `dynamic_scheduler.environment.dynamic_intersection.DynamicIntersectionEnv`
- `dynamic_scheduler.training.trainer.run_episode` (or
  `evaluation.eval_dynamic.run_episode_with`) — whichever already produces a
  usable `op_timeline`
- `evaluation.eval_dynamic.{igreedy_select, lifo_select, backpressure_select,
  edf_select, make_hgt_selector}` — the same five selectors used everywhere
  else this session

**Revised after inspection:** `src/sumo_interface/traci_runner.py`'s
`apply_sumo_control` looked reusable from its signature alone, but its
supporting functions (`vehicle_has_cleared_intersection`,
`stop_vehicle_if_needed`) depend on module-level constants
(`INCOMING_LANES`, `INCOMING_EDGES`) hardcoded to the single intersection's
exact 4 inbound edges, and `LiveVehicleRecord` construction depends on
route-naming parsers (`source_direction_from_route` etc.) specific to that
same network. None of this generalizes to a 16-zone grid, where "cleared the
intersection" has no single fixed definition — it depends on each vehicle's
own route. Rather than partially reuse a file that turns out to be more
tightly coupled to one network than its function signatures suggested,
`sumo_validation/` implements its own self-contained TraCI stop/release loop,
written fresh (informed by reading `traci_runner.py` as a reference for the
TraCI call patterns, not imported from). Nothing in `dynamic_scheduler/`,
`evaluation/`, or `src/sumo_interface/` is modified or imported.

## Data flow

1. **Scenario generation** — `TrafficGenerator.hard(episode_duration)` (or
   configurable tier), same generator used throughout this session.
2. **Per-method scheduling** — for each of the 5 methods, run the existing
   episode loop against `DynamicIntersectionEnv` with that method's selector.
   Capture the resulting `op_timeline`: per (vehicle, route_position) ->
   `{zone_id, start, finish, state}` — the same structure
   `simulation/run_simulation_dynamic.py` already extracts for its visualizer.
3. **Translation** (`translate.py`) — convert `op_timeline` into the
   reservation-window shape `apply_sumo_control` expects (per-vehicle,
   per-zone enter/exit times), using the network builder's zone-id -> SUMO
   edge-id naming convention.
4. **SUMO execution** (`run_comparison.py`, reusing `apply_sumo_control` /
   `VehicleController` unchanged) — spawn vehicles per the same arrivals,
   drive TraCI stop/release per the translated schedule, run to completion.
   Runs headless by default; `--gui` launches SUMO-GUI instead for visual
   playback of one or more methods.
5. **Metric extraction** — read SUMO's own per-vehicle waiting/delay metric
   via TraCI as each vehicle exits the network (not the custom env's number).
6. **Output** — a printed comparison table: one row per method, showing
   **average per-vehicle waiting time as measured by SUMO**, in the same
   spirit as `evaluation/eval_dynamic.py`'s summary line.

## Network generation

SUMO's `netgenerate --grid --grid.number=4 ...` produces a standard 4x4 grid
of junctions/edges matching the zone layout's topology (16 nodes, unit
spacing) in one command. Post-processed to:
- strip traffic-light logic (uncontrolled/priority junctions only — no
  signals, per the "vehicles communicate" scope above)
- set edge lengths/speeds so travel time approximates the scheduler's
  `processing_time` values (first-pass calibration, not exact correspondence)

Chosen over hand-authoring the network XML: lower risk of subtle
connectivity/geometry bugs, faster to get a working network, and precise
calibration can follow later as a smaller, separate step if the feasibility
proof shows it's needed.

## Success criteria

- All 5 methods run end-to-end through SUMO on one scenario without error.
- A printed table shows each method's average per-vehicle waiting time as
  measured by SUMO's own metric.
- At least one method's run can be watched live in SUMO-GUI via `--gui`, to
  visually sanity-check that the translated schedule produces sensible
  vehicle behavior (stopping/releasing at the right times) before trusting
  the numbers.
- README documents how to run it and states the known limitations
  (single scenario, approximate calibration) plainly.

## Known limitations (documented, not solved this phase)

- Single scenario — not a statistically powered study. If this reveals a
  ranking discrepancy worth investigating, that's the trigger for a larger
  multi-scenario version, not something to build preemptively.
- Edge length/speed calibration against `processing_time` is a first-pass
  approximation.
