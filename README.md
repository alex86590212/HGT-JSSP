# HGT-JSSP

Beginner-friendly guide to this repository.

This project studies how to schedule vehicles through an unsignalized traffic
intersection by treating the intersection like a Job Shop Scheduling Problem
(JSSP). In a normal JSSP, jobs move through machines in a required order. In
this project:

- A vehicle is like a job.
- A conflict zone inside the intersection is like a machine or shared resource.
- A vehicle crossing one conflict zone is an operation.
- The scheduler decides which operation should happen next while avoiding unsafe
  overlaps and preserving vehicle route order.

The repository currently contains two related implementations:

1. `intersection_scheduler/`: the main learning-based research code. It creates
   synthetic intersection scenarios, converts them into heterogeneous graphs,
   trains a Heterogeneous Graph Transformer (HGT) policy with PPO, and evaluates
   it against an iGreedy baseline.
2. `src/` plus `sumo/`: a SUMO/TraCI traffic simulation pipeline. It observes
   live vehicles in SUMO, builds a JSSP-style graph, schedules vehicles with an
   FCFS reservation scheduler, controls SUMO stop/release behavior, and writes
   logs and safety metrics.

If you are new to this project, start with the mental model below, then read the
folder guide.

## Contents

- [Quick Start](#quick-start)
- [Project Mental Model](#project-mental-model)
- [Main Workflows](#main-workflows)
- [Folder-By-Folder Guide](#folder-by-folder-guide)
- [Technology And Methodology](#technology-and-methodology)
- [Known Notes From Training](#known-notes-from-training)
- [Dependencies](#dependencies)
- [Reference Links](#reference-links)

## Quick Start

Run commands from the repository root:

```powershell
cd "C:\Users\Travid\Documents\GitHub project\HGT-JSSP"
```

Create and activate a Python environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the unit tests:

```powershell
python -m pytest
```

Train the HGT policy:

```powershell
python train.py --config configs/default.yaml --output results
```

Resume training from a checkpoint:

```powershell
python train.py --config configs/default.yaml --output results_v2 --resume results_v2/checkpoint_30000.pt
```

Evaluate a trained checkpoint against iGreedy:

```powershell
python eval.py --checkpoint results_v2/checkpoint_best.pt --n-scenarios 200
```

Plot TensorBoard training curves:

```powershell
python plot_training.py --logdir results_v2/tb --out results_v2/training_curves.png
```

Run the HGT/iGreedy replay animation:

```powershell
python simulation/run_simulation.py --checkpoint results_v2/checkpoint_best.pt --mode both --difficulty hard
```

Run the modular SUMO/FCFS batch pipeline:

```powershell
python -m src.main --mode headless --scheduler fcfs
```

Run the deterministic controller test scenario without SUMO:

```powershell
python -m src.main --mode test-scenarios --scheduler fcfs
```

Run the live SUMO visual debugger:

```powershell
python -m src.main --mode debug --scheduler fcfs
```

The SUMO commands require SUMO tools such as `sumo`, `sumo-gui`, and
`netconvert` to be installed and available on your `PATH`.

## Project Mental Model

The intersection is divided into conflict zones. A conflict zone is a small area
where two vehicle paths can overlap. If two vehicles need the same zone at the
same time, one must wait.

For each vehicle, the route through the intersection becomes a sequence of
operations. For example, a vehicle might need zones `[2, 5, 8]`. It must finish
zone `2` before entering zone `5`, and finish zone `5` before entering zone `8`.

The scheduler repeatedly asks:

- Which operations are feasible right now?
- Which feasible operation should be scheduled next?
- How does that choice change future waiting time?

The learning-based path uses a graph neural network to answer the second
question. The graph contains:

- Operation nodes: one node per vehicle-zone operation.
- Vehicle nodes: one node per vehicle.
- Zone nodes: one node per conflict zone.
- Type-1 edges: route order inside a vehicle.
- Type-2 edges: same-lane order between vehicles.
- Type-3 edges: shared-zone conflict candidates.

The policy is trained with reinforcement learning. It receives reward when it
keeps total completion time low, and it learns to choose operations that reduce
waiting.

## Main Workflows

### 1. Train The Learning-Based Scheduler

Entry point:

```powershell
python train.py
```

This loads `configs/default.yaml`, creates a `SchedulingPolicy`, samples
curriculum scenarios, runs episodes in `IntersectionEnv`, updates the policy
with PPO, logs TensorBoard metrics, and writes checkpoints.

Important outputs:

- `results/checkpoint_5000.pt`, `checkpoint_10000.pt`, etc.
- `results/checkpoint_best.pt`
- `results/tb/events.out.tfevents...`

### 2. Evaluate HGT Against iGreedy

Entry point:

```powershell
python eval.py --checkpoint results_v2/checkpoint_best.pt
```

This compares the trained HGT policy with iGreedy across easy, medium, and hard
scenario tiers. iGreedy is a deterministic baseline that chooses the feasible
operation belonging to the earliest-arriving vehicle.

### 3. Inspect Training Curves

Entry point:

```powershell
python plot_training.py
```

This reads TensorBoard event files and plots:

- evaluation waiting time
- training waiting time
- steps per episode
- total reward
- curriculum phases

### 4. Replay A Learned Or Baseline Schedule

Entry point:

```powershell
python simulation/run_simulation.py --mode both
```

This does not run SUMO. It uses the synthetic `intersection_scheduler`
environment, solves a scenario with HGT and/or iGreedy, then animates the solved
schedule with matplotlib.

### 5. Run SUMO With FCFS Control

Entry point:

```powershell
python -m src.main --mode headless --scheduler fcfs
```

This path uses SUMO and TraCI. It observes approaching vehicles, builds a
NetworkX JSSP graph, schedules reservations with FCFS, sends stop/release
commands to SUMO, validates safety, and writes metrics.

## Folder-By-Folder Guide

### `intersection_scheduler/`

This is the main HGT/PPO research package. It is self-contained and does not
need SUMO for training or evaluation.

#### `intersection_scheduler/environment/`

This folder defines the synthetic scheduling environment.

- `intersection.py` defines `Vehicle`, `Operation`, `Zone`, and
  `IntersectionEnv`.
- `IntersectionEnv.reset()` turns a list of vehicles into a flat list of
  operations.
- `IntersectionEnv.step(action)` schedules one operation, updates zone
  availability, propagates finish times, computes reward, and reports whether
  the episode is done.
- `feasibility.py` computes the feasible action mask. It enforces vehicle
  arrival time, route predecessor completion, same-lane ordering, zone
  availability, and deadlock avoidance.
- `graph_builder.py` converts the current environment state into a PyTorch
  Geometric `HeteroData` graph with operation, vehicle, and zone node types.

The key idea: the environment behaves like a Markov Decision Process (MDP).
Each state is the current partial schedule. Each action chooses one feasible
operation to schedule next.

#### `intersection_scheduler/data/`

This folder creates synthetic intersection scenarios.

- `scenario_generator.py` defines a 3x3 conflict-zone grid and route templates.
- Routes include straight, left-turn, and right-turn movements.
- `ScenarioGenerator.easy()`, `.medium()`, and `.hard()` create different
  traffic difficulty levels.
- `get_curriculum_scenario()` changes scenario difficulty as training progresses:
  easy first, then medium, hard, and finally randomized scenarios.

#### `intersection_scheduler/model/`

This folder defines the neural network policy.

- `hgt.py` defines `IntersectionHGT`, the Heterogeneous Graph Transformer
  backbone.
- The model uses node types `operation`, `vehicle`, and `zone`.
- It uses edge types `seq`, `lane`, `conflict`, `owns`, and `hosts`.
- `policy.py` defines `SchedulingPolicy`, an actor-critic policy built on top
  of `IntersectionHGT`.
- The actor outputs logits for operation actions.
- The critic estimates the value of the current state.
- Infeasible actions are masked to `-inf` so the policy cannot choose them.

#### `intersection_scheduler/training/`

This folder contains the reinforcement learning loop.

- `trainer.py` runs episodes, logs metrics, evaluates periodically, and saves
  checkpoints.
- `run_episode()` builds graph observations, computes the feasible mask, samples
  or chooses actions, and collects transitions.
- `ppo.py` defines the `Transition` data class, generalized advantage
  estimation, and the PPO update.

#### `intersection_scheduler/utils/`

This folder contains evaluation helpers.

- `metrics.py` computes mean waiting time and makespan.
- It also implements `igreedy()`, the deterministic baseline scheduler.
- `evaluate_hgt_vs_igreedy()` compares the learned policy against iGreedy and
  can optionally write per-scenario CSV results.

#### `intersection_scheduler/tests/`

This folder contains pytest tests for the learning package.

- `test_environment.py` checks scheduling completeness, reward behavior, finish
  time monotonicity, and graph consistency.
- `test_feasibility.py` checks arrival constraints, predecessor constraints,
  feasible-set behavior, and deadlock checks.
- `test_model.py` checks HGT forward passes, policy masking, scalar value
  output, and PPO gradient sanity.

### `src/`

This is the modular SUMO/TraCI FCFS controller pipeline. It is older or
auxiliary compared with `intersection_scheduler/`, but it is important because
it connects the scheduling idea to live traffic simulation.

#### `src/main.py`

Canonical entry point for the SUMO/FCFS experiment pipeline.

It supports:

- `--mode headless`: run batch SUMO experiments without GUI.
- `--mode debug`: run SUMO GUI plus live JSSP graph visualization.
- `--mode test-scenarios`: run deterministic graph/scheduler tests without SUMO.
- `--scheduler fcfs`: use the implemented FCFS scheduler.
- `--scheduler greedy`: currently a placeholder and intentionally not
  implemented.

#### `src/common/`

Shared model aliases and small data classes.

- `models.py` re-exports key TraCI runner records and defines shared
  `ControlResult`, `SafetyViolation`, and `EpisodeMetrics` structures.

#### `src/config/`

Configuration loading helpers.

- `load_config.py` loads JSON files from `configs/`.
- It also resolves paths relative to the project root.

#### `src/constraints/`

Scheduler-independent feasibility logic for the SUMO controller.

- `feasibility.py` defines `FeasibilityChecker`.
- It checks Type-1 route order, Type-2 same-lane order, Type-3 conflict-zone
  occupancy, arrival time, blocking behavior, and deadlock risk.
- It defines a mutable `ScheduleState` used while building reservations.

#### `src/evaluation/`

Logging, metrics, and safety validation.

- `metrics.py` computes throughput, delay, waiting time, stops, runtime, safety
  violations, and collision metrics.
- `episode_logger.py` writes per-episode CSV/JSON artifacts.
- `safety_validator.py` records conflict-zone occupancy intervals and reports
  unsafe overlaps.

#### `src/graph/`

JSSP graph construction and validation.

- `conflict_zones.py` defines four named conflict zones and route-to-zone
  mappings for the single intersection.
- `jssp_graph_builder.py` re-exports live graph-building helpers from the TraCI
  runner.
- `graph_validator.py` checks graph node fields, edge fields, and route-zone
  consistency.
- `static_jssp_example.py` builds a deterministic example graph without SUMO and
  can export JSON/PNG artifacts.

#### `src/schedulers/`

Scheduler interfaces and implementations.

- `base.py` defines the scheduler interface.
- `fcfs_scheduler.py` wraps the conflict-zone-aware FCFS reservation scheduler.
- `greedy_scheduler.py` is a placeholder and raises `NotImplementedError`.

#### `src/sumo_interface/`

SUMO and TraCI integration.

- `traci_runner.py` is the largest file in this path. It starts SUMO, observes
  vehicles, builds JSSP graphs, schedules reservations, applies stop/release
  commands, logs graph steps, and returns run summaries.
- `vehicle_observer.py` wraps vehicle observation helpers.
- `vehicle_controller.py` wraps vehicle stop/release helpers.

#### `src/visualization/`

Live graph debugging for SUMO runs.

- `graph_debugger.py` opens a matplotlib view beside SUMO GUI.
- It shows the current JSSP timing-conflict graph, FCFS order, reservations,
  stopped vehicles, released vehicles, and edge counts.

### `configs/`

Configuration files used by both major paths.

- `default.yaml` configures PPO, the HGT model, training episode count, logging
  interval, evaluation interval, and checkpoint interval.
- `experiment_config.json` configures the SUMO experiment mode, scheduler,
  scenario, seeds, simulation horizon, output directory, and logging behavior.
- `intersection_config.json` maps scenario names to SUMO config files, defines
  incoming lanes/edges, control distances, scheduling clearances, conflict-zone
  geometry, and route-to-conflict-zone sequences.
- `vehicle_types.json` defines nominal per-zone processing times for passenger,
  delivery, truck, and bus vehicle types.

### `scripts/`

Helper scripts and compatibility wrappers.

- `generate_single_intersection.py` generates the SUMO single-intersection
  network, route files, baseline demand, hard demand, detectors, `.sumocfg`
  files, and the local SUMO README. It can also call `netconvert` if SUMO is
  installed.
- `build_intersection_jssp_graph.py` is a compatibility wrapper around
  `src.graph.static_jssp_example`.
- `run_traci_fcfs_controller.py` is a compatibility wrapper around the modular
  TraCI runner. New runs should usually use `python -m src.main`.
- `run_visual_debug_controller.py` is a compatibility wrapper around the live
  graph debugger. New debug runs should usually use `python -m src.main
  --mode debug`.

### `simulation/`

Synthetic schedule replay and visualization.

- `run_simulation.py` runs either the trained HGT policy, iGreedy, or both on a
  generated scenario.
- It then animates the solved schedule with a 3x3 intersection map, optional
  JSSP graph panel, and debug statistics.
- This is useful for understanding how the learned policy and iGreedy differ
  without starting SUMO.

### `sumo/single_intersection/`

Generated SUMO files for a single unsignalized four-way intersection.

- `single_intersection.nod.xml`: node definitions.
- `single_intersection.edg.xml`: edge definitions.
- `single_intersection.con.xml`: allowed turning connections.
- `single_intersection.net.xml`: compiled SUMO network.
- `routes.rou.xml`: vehicle types and route definitions.
- `demand.rou.xml`: baseline deterministic demand.
- `hard_demand.rou.xml`: bursty stress-test demand.
- `additional.add.xml`: optional inbound lane detectors.
- `single_intersection.sumocfg`: baseline runnable SUMO configuration.
- `single_intersection_hard.sumocfg`: hard SUMO scenario configuration.
- `README.md`: local explanation of the SUMO network and how to regenerate it.
- `detectors.out.xml`: detector output generated by SUMO.

### `results/` And `results_v2/`

Saved experiment artifacts.

- `checkpoint_*.pt`: saved training checkpoints.
- `checkpoint_best.pt`: best checkpoint according to periodic evaluation.
- `tb/events.out.tfevents...`: TensorBoard scalar logs.
- `results/intersection_graph.png`: existing graph/image artifact.

The notes in `NOTES.md` say the existing checkpoints were trained before some
bug fixes, so their numbers should be interpreted with that context.

### `pyHGT/`

This directory is present but currently empty in the checked-in repository. It
may have been intended for a separate or vendored HGT implementation.

### Root Files

- `README.md`: this beginner guide.
- `NOTES.md`: training notes, bug fixes, observations, and future improvement
  ideas.
- `requirements.txt`: Python dependencies for the main HGT/PPO path.
- `train.py`: CLI entry point for training the HGT scheduling policy.
- `eval.py`: CLI entry point for evaluating HGT against iGreedy.
- `plot_training.py`: CLI entry point for plotting TensorBoard training curves.

## Technology And Methodology

### SUMO

SUMO stands for Simulation of Urban Mobility. It is an open-source microscopic
traffic simulator. "Microscopic" means it simulates individual vehicles rather
than only aggregate traffic flow.

In this repo, SUMO is used by the `src/` pipeline to create a real traffic
simulation around a single intersection.

### TraCI

TraCI means Traffic Control Interface. It lets Python control a running SUMO
simulation. The code uses TraCI to:

- read vehicle positions, lanes, speeds, and route IDs
- estimate vehicle arrival times
- stop vehicles before the intersection
- release selected vehicles
- detect collisions and simulation completion

### JSSP

JSSP means Job Shop Scheduling Problem. A classical JSSP has jobs, machines,
ordered tasks, and a goal such as minimizing makespan. In this repo:

- vehicles are jobs
- conflict zones are machines/resources
- zone crossings are tasks/operations
- route order is a precedence constraint
- shared conflict zones create no-overlap constraints

### Conflict Zones

A conflict zone is an area inside the intersection where routes may collide or
overlap. If two vehicles need the same conflict zone, the scheduler must choose
an order.

The SUMO controller path uses four zones named `z1`, `z2`, `z3`, and `z4`.
The synthetic HGT path uses a 3x3 grid of numbered zones.

### Heterogeneous Graph

A graph has nodes and edges. A heterogeneous graph has different kinds of nodes
and different kinds of edges. This project uses heterogeneous graphs because
"vehicle", "operation", and "zone" are different concepts, and route order,
lane order, and conflict relations mean different things.

### HGT

HGT means Heterogeneous Graph Transformer. It is a graph neural network designed
for heterogeneous graphs. Here, HGT reads the current scheduling graph and
produces embeddings for operation nodes. Those embeddings feed the actor and
critic networks.

### PyTorch

PyTorch is the deep learning library used for neural network modules, tensors,
automatic differentiation, optimization, checkpoint saving, and model loading.

### PyTorch Geometric

PyTorch Geometric is a graph neural network library built on PyTorch. This
project uses its `HeteroData` structure and `HGTConv` layer.

### Actor-Critic

Actor-critic is a reinforcement learning architecture with two parts:

- The actor chooses an action.
- The critic estimates how good the current state is.

In this repo, the actor chooses the next operation to schedule. The critic
estimates expected future reward from the current partial schedule.

### PPO

PPO means Proximal Policy Optimization. It is a reinforcement learning algorithm
that updates a policy while clipping the update size, which helps avoid unstable
jumps in behavior.

In this project, PPO uses collected scheduling transitions to update the HGT
policy after each episode.

### GAE

GAE means Generalized Advantage Estimation. It estimates how much better or
worse an action was compared with what the critic expected. PPO uses these
advantage values to train the actor.

### Feasible Action Mask

The feasible action mask is a boolean vector saying which operations are legal
to schedule now. If an operation violates arrival time, route order, same-lane
order, zone availability, or deadlock rules, it is masked out.

Masking matters because the neural policy should choose between valid scheduling
decisions, not learn by crashing into invalid actions.

### iGreedy

iGreedy is the baseline scheduler in `intersection_scheduler/utils/metrics.py`.
At each step, it chooses the feasible operation from the earliest-arriving
vehicle, with tie-breaks by route position and vehicle ID.

It is simpler than HGT and provides a comparison point for learning.

### FCFS

FCFS means First Come, First Served. The SUMO controller path uses FCFS as a
reservation scheduler. Vehicles are prioritized by estimated arrival time, but
the schedule reserves individual conflict zones rather than locking the whole
intersection.

### TensorBoard

TensorBoard is a tool for viewing training logs. The trainer writes scalar
metrics such as waiting time, reward, actor loss, critic loss, and entropy into
`results*/tb/`.

### Checkpoints

Checkpoints are `.pt` files saved by PyTorch. They store model weights and,
for interval checkpoints, optimizer state and episode number. `checkpoint_best.pt`
stores the best policy observed during periodic evaluation.

### Waiting Time

Waiting time measures idle delay. In this project it is computed as:

```text
actual finish time - arrival time - minimum processing time
```

Lower waiting time is better.

### Makespan

Makespan is the time when the last operation finishes. In traffic terms, it is
when the last scheduled vehicle operation clears.

## Known Notes From Training

`NOTES.md` records several important details about the current training history.

### Fixed Bugs

- `zone.occupied` was stale in the HGT graph builder. The feature used to stay
  true after a zone had ever been used. The code now derives occupancy from
  whether `zone.time_free > env.current_time`.
- `igreedy()` had an early-break bug. When no operation was feasible, it used to
  stop instead of advancing time. It now uses `next_feasible_time()`.
- `run_episode()` had a similar temporal-gap bug. It used to break early when no
  actions were currently feasible. It now advances time to the next feasible
  event and continues.

### Impact On Existing Checkpoints

The notes say existing `results/` and `results_v2/` checkpoints were trained
with the stale zone-occupation feature. That means the saved weights may partly
compensate for an incorrect input signal. Runs started after the fix receive the
correct signal from the beginning.

### Observed Results

According to `NOTES.md`, `results_v2/checkpoint_best.pt` evaluated on 200
scenarios had these mean waiting-time comparisons:

| Tier | iGreedy (s) | HGT (s) | Improvement |
| --- | ---: | ---: | ---: |
| easy | 0.111 | 0.082 | +25.9% |
| medium | 0.620 | 0.387 | +37.6% |
| hard | 2.083 | 1.062 | +49.0% |
| overall | 0.938 | 0.510 | +45.6% |

The hard tier is where the graph structure appears most useful, because there
are more route conflicts and more scheduling choices.

### Future Improvement Ideas

`NOTES.md` suggests:

- collect batched PPO rollouts instead of updating after every single episode
- normalize rewards by episode size to reduce critic target scale changes
- add a time-skip penalty so idle gaps affect learning
- schedule the entropy coefficient so exploration pressure decreases over time

## Dependencies

The checked-in `requirements.txt` currently lists:

```text
torch>=2.1.0
torch-geometric>=2.4.0
pyyaml
numpy
pytest
tensorboard
```

These cover the main HGT/PPO training path.

Some scripts import additional tools that are not currently listed in
`requirements.txt`:

- `matplotlib`: used by `plot_training.py`, `simulation/run_simulation.py`, and
  visual graph debugging.
- `networkx`: used for NetworkX JSSP graphs in the SUMO/debug path.
- `traci`: used to control SUMO from Python.
- SUMO command-line programs: `sumo`, `sumo-gui`, and `netconvert`.

If the SUMO or visualization commands fail with missing imports or missing
executables, install those optional tools in the active environment and make
sure SUMO is on your system `PATH`.

## Reference Links

- [SUMO documentation](https://sumo.dlr.de/docs/index.html)
- [TraCI documentation](https://sumo.dlr.de/docs/TraCI/index.html)
- [PyTorch documentation](https://docs.pytorch.org/docs/2.12/index.html)
- [PyTorch Geometric HGTConv](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.HGTConv.html)
- [Proximal Policy Optimization paper](https://arxiv.org/abs/1707.06347)
- [Heterogeneous Graph Transformer paper](https://arxiv.org/abs/2003.01332)
- [Generalized Advantage Estimation paper](https://arxiv.org/abs/1506.02438)
- [OR-Tools Job Shop explanation](https://developers.google.com/optimization/scheduling/job_shop)
- [NetworkX DiGraph documentation](https://networkx.org/documentation/stable/reference/classes/digraph.html)

## Suggested Reading Order

If this is your first time opening the project, read files in this order:

1. `README.md` for the overview.
2. `NOTES.md` for training history and caveats.
3. `configs/default.yaml` for the HGT/PPO settings.
4. `intersection_scheduler/data/scenario_generator.py` to understand scenarios.
5. `intersection_scheduler/environment/intersection.py` to understand the MDP.
6. `intersection_scheduler/environment/graph_builder.py` to understand the graph
   observation.
7. `intersection_scheduler/model/policy.py` and `model/hgt.py` to understand the
   neural policy.
8. `intersection_scheduler/training/trainer.py` and `training/ppo.py` to
   understand learning.
9. `eval.py` and `intersection_scheduler/utils/metrics.py` to understand
   evaluation.
10. `src/main.py` if you want the SUMO/TraCI controller path.

## Current Limitations

- The HGT/PPO training environment is synthetic and separate from the SUMO
  controller path.
- The SUMO controller currently implements FCFS. The `GreedyScheduler` in
  `src/schedulers/greedy_scheduler.py` is only a placeholder.
- Existing checkpoints should be interpreted with the bug-fix caveats in
  `NOTES.md`.
- Optional visualization and SUMO dependencies may need to be installed
  separately from `requirements.txt`.
