# SinD Departure Scenarios for JSSP-HGT

## Data in the original dataset

SinD is a drone-derived dataset of signalized intersections in China. Its raw
recordings can contain:

- `Veh_smoothed_tracks.csv`: timestamped motor-vehicle and cycle trajectories,
  including class, ground-plane position, velocity, acceleration, heading,
  dimensions, and track ID.
- `Ped_smoothed_tracks.csv`: timestamped pedestrian position, velocity, and
  acceleration.
- `Veh_tracks_meta.csv` and `Ped_tracks_meta.csv`: track boundaries and, where
  available, crossing type and signal-violation annotations.
- `TrafficLight*.csv` or `Traffic_Lights.csv`: synchronized signal changes.
- `recording_metas.csv`: location, time period, weather, recording duration,
  frame rate, and participant counts.
- `*.osm`: the Lanelet2 HD map for the physical intersection.

The public repository describes separate intersections in Tianjin, Chongqing,
Changchun, and Xi'an. A city can contain multiple recordings, and the full
dataset can contain more than one intersection per city. These levels must not
be mixed when constructing training or evaluation splits.

## What this project keeps

This project uses SinD only as **departure demand**. A departure occurs when a
vehicle first crosses inward through the configured 25 m management boundary.
From that point onward, JSSP-HGT owns the intersection decision and timing.

Each prepared departure contains only:

| Field | Meaning |
| --- | --- |
| `departure_id` | Stable participant/event identifier |
| `departure_time_s` | Time within its 30-second scenario window |
| `source_approach` | Entry side: E, N, S, or W |
| `destination_approach` | Intended exit side inferred during preprocessing |
| `source_lane` | Current abstract incoming lane |
| `route_id` | One of the project's 12 origin-to-destination routes |
| `vehicle_type` | `passenger`, `truck`, or `bus` |
| `sequence` | Stable same-scenario departure order |

Raw trajectories are used during preprocessing only to verify a complete
traversal, infer origin/destination, and detect the boundary entry. Prepared
training files explicitly exclude raw positions, speeds, accelerations,
observed intersection motion, observed exit times, traffic-light states, and
signal-violation labels. Consequently, the policy cannot copy SinD's original
signal control or observed crossing behavior.

Cars, trucks, and buses are retained. Motorcycles, bicycles, tricycles, and
pedestrians are excluded until the graph has explicit crosswalk/cycle conflict
resources and suitable safety clearances.

## Organized output

The public snapshot is installed under `data/sind/`. The processed output is
organized as:

```text
data/processed/sind/
|-- catalog.json
`-- intersections/
    |-- chongqing_nr_ll2/
    |   `-- recordings/
    |       `-- chongqing_6_22_nr_1/
    |           |-- scenario_0000.json
    |           `-- ...
    |-- tianjin_map_relink_law_save/
    |   `-- recordings/...
    `-- xi_an_xi_an_shanglin/
        `-- recordings/...
```

`catalog.json` is an index and audit file. It contains intersection summaries,
recording status, scenario paths, conversion settings, source commit, and
inclusion/exclusion counts. Each scenario file contains departures from exactly
one physical intersection, one recording, and one time window.

The current public sample produces:

- four catalogued intersections;
- three usable recordings (Tianjin, Chongqing, and Xi'an);
- 78 separate 30-second scenarios;
- 429 accepted departures.

Changchun is kept as a distinct catalog entry with zero scenarios because its
public vehicle CSV is a Git LFS pointer rather than trajectory content.
Congestion thresholds are calculated independently within each intersection,
so a low-density site is not mislabeled using another site's distribution.

## Prepare or refresh the data

```powershell
.venv\Scripts\python.exe -m src.learning.ppo_hgt.prepare_sind
```

Useful overrides:

```powershell
.venv\Scripts\python.exe -m src.learning.ppo_hgt.prepare_sind `
  --data-root D:\datasets\SinD\Data `
  --output data\processed\sind\catalog.json `
  --window-duration-s 30 `
  --entry-radius-m 25 `
  --conflict-radius-m 18 `
  --min-departures 2
```

The result is deterministic for a fixed raw snapshot and arguments.

## Train

The default configuration points to the departure catalog:

```json
"scenario": {
  "source": "sind",
  "sind_catalog_path": "data/processed/sind/catalog.json"
}
```

Run:

```powershell
.venv\Scripts\python.exe -m src.learning.ppo_hgt.train
```

The runtime converts departures into the existing `LiveVehicleRecord` API at
the management boundary. `departure_time_s` becomes the scheduling release/ETA
time. Position and speed are fixed environment initialization constants, not
values learned from SinD. Logs include `intersection_id`, `recording_id`, and
`window_start_s`.

Set `scenario.source` to `synthetic` to return to generated Poisson demand.

## Recommendations

1. Split by `intersection_id`, not by random scenario. Adjacent windows from
   the same recording are strongly correlated.
2. Keep every evaluation intersection completely absent from training,
   including all its recordings and windows.
3. Treat the public 78 scenarios as a smoke/fine-tuning set. Mix or pretrain on
   synthetic demand for broader congestion and route coverage.
4. Validate the inferred approach mapping with explicit Lanelet2 entry polygons
   before final experiments.
5. Add VRUs later as dedicated graph resources; do not alias them to cars.
6. Keep `catalog.json` with checkpoints so every experiment retains its source
   commit, site hierarchy, and conversion audit.

## Scope

SinD supplies who departs, when, from which approach, toward which destination,
and with which supported vehicle class. It does not supply the learned policy's
crossing order, reservation time, signal decision, trajectory, or exit time.
