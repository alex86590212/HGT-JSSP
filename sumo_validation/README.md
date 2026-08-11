# SUMO validation

Runs HGT, iGreedy, LIFO, Backpressure, and EDF through SUMO on the same
scenario and compares each method's average per-vehicle waiting time as SUMO
itself measures it, not the custom `DynamicIntersectionEnv`'s number.

No traffic lights: this models a connected/autonomous-vehicle intersection
where vehicles communicate with a central scheduler, not signal-controlled
traffic. Right-of-way is resolved entirely by each method's zone-reservation
schedule via TraCI stop/release commands.

## Setup

```
brew install sumo
```

Confirm `sumo`, `sumo-gui`, `netgenerate` are on PATH afterward. Set
`SUMO_HOME` if `brew` doesn't do it automatically (check `brew info sumo`).

```
pip install traci sumolib
```

## Run

```
PYTHONPATH=. python sumo_validation/run_comparison.py \
    --checkpoint "results_dynamic_reward_fix/checkpoint_best_30_so_far (1).pt" \
    --tier hard --seed 42
```

Watch one method live in SUMO-GUI:

```
PYTHONPATH=. python sumo_validation/run_comparison.py \
    --checkpoint "results_dynamic_reward_fix/checkpoint_best_30_so_far (1).pt" \
    --tier hard --seed 42 --gui-method hgt
```

The network is generated once (`network/grid_4x4.net.xml`) on first run and
reused afterward. Delete it to force regeneration.

## What a "zone" means here

`DynamicZone` in the custom env has no geometry — it's a point with an
exclusive-occupancy rule: one vehicle holds it from `start_time` to
`earliest_finish`, no overlaps. The literal translation into SUMO: one zone =
one short edge, entry gated by that vehicle's reserved `start_time`.

## Known limitations

- **Single scenario per run**, not a statistically powered study. This is a
  feasibility proof: does the HGT-vs-baselines ranking survive real SUMO
  physics for at least one case.
- **Edge length/speed calibration is a first-pass approximation.** Edge
  length is fixed at `EDGE_LENGTH_M = 40.0` and speed at `SPEED_MPS = 10.0`
  in `network/build_grid_network.py`, not derived from the scheduler's
  per-vehicle `processing_time` values.
- **`netgenerate`'s edge-naming convention is assumed, not confirmed.**
  `network/zone_mapping.py` assumes grid nodes are named `A0`..`D3` (row
  letter + column number) and edges are named `{from_node}{to_node}` (e.g.
  `A1B1`). This was written from SUMO's documented `--grid` behavior; nobody
  has run `netgenerate` on this machine to confirm it (SUMO isn't installed
  here). **First thing to check after installing SUMO**: run
  `network/build_grid_network.py` and inspect `grid_4x4.net.xml` for the
  actual `<edge id="...">` values — if they don't match `zone_mapping.py`'s
  assumption, fix `_node_name`/`zone_incoming_edge` there.
- **`vehicle_has_cleared` in `sumo_control.py` is a generic "no route edges
  left" check**, not verified against real TraCI behavior at a route's last
  edge (e.g. whether `getRouteIndex` behaves as expected once a vehicle is on
  its final edge). Worth confirming with `--gui-method` on a small scenario.
- **`getTimeLoss` as the reported metric.** Chosen because it's SUMO's
  standard "delay relative to free flow" measure, the same convention the
  corrected training reward uses. Not yet cross-checked against SUMO's other
  per-vehicle stats (`waitingTime`, `duration`) to confirm it's the best fit
  once real output is available.
- Compares against `dynamic_scheduler`/`evaluation` code unchanged; nothing
  in the training/reward path is touched by this project.
