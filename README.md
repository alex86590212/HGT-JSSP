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

1. `intersection_scheduler/`: learning-based research code. It creates
   synthetic intersection scenarios, converts them into heterogeneous graphs,
   trains a Heterogeneous Graph Transformer (HGT) policy with PPO, and evaluates
   it against an iGreedy baseline.
2. `admission_rgcn/`: the R-GCN admission policy (the focus of the current
   paper, see `paper_rgcn/`). Instead of choosing individual operations, the
   controller reserves every conflict zone along a vehicle's whole route before
   entry (gap-free reservations). A Relational GCN policy trained with PPO
   decides which pending vehicle to admit, or whether to defer, and is compared
   with rule-based admission baselines (iGreedy, EDF, Backpressure, LIFO) and a
   CP-SAT reference `W*`. See [admission_rgcn/README.md](admission_rgcn/README.md)
   for training, evaluation, and ablation commands.

The gap-free admission environment and the baselines shared by the HGT
(`admission_rl/`) and R-GCN policies live in `dynamic_scheduler/`.

If you are new to this project, start with the mental model below, then read the
folder guide.

## Contents

- [Quick Start](#quick-start)
- [R-GCN Admission Policy](#r-gcn-admission-policy)
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

## R-GCN Admission Policy

Train, evaluate against the baselines, and run the deferral ablation (run from
the repository root; details in [admission_rgcn/README.md](admission_rgcn/README.md)):

```bash
PYTHONPATH=. python -m admission_rgcn.train --output runs/admission_rgcn
PYTHONPATH=. python -m admission_rgcn.evaluate \
    --checkpoint runs/admission_rgcn/checkpoint_best.pt \
    --output-csv runs/admission_rgcn/eval_final.csv
PYTHONPATH=. python -m admission_rgcn.run_ablation --out runs/ablation --episodes 4000
PYTHONPATH=. python -m admission_rgcn.summarize_ablation --dir runs/ablation
```

Paper figures and tables are produced from `paper_rgcn/`
(`make_scaling_plot.py`, `build_tables.py`).

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

It uses the synthetic `intersection_scheduler`
environment, solves a scenario with HGT and/or iGreedy, then animates the solved
schedule with matplotlib.

## Folder-By-Folder Guide

### `intersection_scheduler/`

This is the main HGT/PPO research package. It is self-contained.

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

### `configs/`

Configuration files.

- `default.yaml` configures PPO, the HGT model, training episode count, logging
  interval, evaluation interval, and checkpoint interval.
- `vehicle_types.json` defines nominal per-zone processing times for passenger,
  delivery, truck, and bus vehicle types.

### `scripts/`

Helper scripts and compatibility wrappers.

### `simulation/`

Synthetic schedule replay and visualization.

- `run_simulation.py` runs either the trained HGT policy, iGreedy, or both on a
  generated scenario.
- It then animates the solved schedule with a 3x3 intersection map, optional
  JSSP graph panel, and debug statistics.
- This is useful for understanding how the learned policy and iGreedy differ

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
  the replay animation.

## Reference Links

- [PyTorch documentation](https://docs.pytorch.org/docs/2.12/index.html)
- [PyTorch Geometric HGTConv](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.HGTConv.html)
- [Proximal Policy Optimization paper](https://arxiv.org/abs/1707.06347)
- [Heterogeneous Graph Transformer paper](https://arxiv.org/abs/2003.01332)
- [Generalized Advantage Estimation paper](https://arxiv.org/abs/1506.02438)
- [OR-Tools Job Shop explanation](https://developers.google.com/optimization/scheduling/job_shop)

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

## Current Limitations

- Existing checkpoints should be interpreted with the bug-fix caveats in
  `NOTES.md`.
- Optional visualization dependencies such as `matplotlib` may need to be
  installed separately from `requirements.txt`.
