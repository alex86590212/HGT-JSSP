# HGT-JSSP

A learned scheduling policy for 4-way intersection control, formulated as a Job Shop
Scheduling Problem (JSSP) and solved with a Heterogeneous Graph Transformer (HGT)
trained via PPO.

Given a set of approaching vehicles (arrival time, route, velocity), the trained
policy outputs an exact conflict-free timetable — the entry and exit time for every
vehicle in every zone it crosses — while minimizing total waiting time.

## How it works

The intersection is modelled as a 3×3 grid of 9 conflict zones. Each vehicle's route
is a sequence of zones; scheduling is a sequence of decisions, one per (vehicle, zone)
operation. A heterogeneous graph — operation, vehicle, and zone nodes connected by
route-order, same-lane, and zone-conflict edges — is rebuilt at every decision step
and fed to a 3-layer HGT actor-critic. The policy picks the next operation to
schedule; the environment computes its exact start/end time from arrival time,
route predecessor, and zone availability.

Training uses PPO with curriculum learning (2 → 8 vehicles) over 50,000 episodes.

## Repo layout

```
intersection_scheduler/
  data/scenario_generator.py     Procedural scenario generator (easy/medium/hard tiers)
  environment/
    intersection.py              IntersectionEnv — the scheduling MDP
    feasibility.py                Feasible-set + deadlock-free action computation
    graph_builder.py              Builds the heterogeneous PyG graph each step
  model/
    hgt.py                        HGT graph backbone
    policy.py                     Actor-critic head
  training/
    trainer.py                    Episode loop, curriculum, PPO orchestration
    ppo.py                        GAE + PPO update (batched)
  utils/metrics.py                Waiting time, makespan, FCFS/iGreedy baselines

evaluation/
  eval.py                         HGT vs iGreedy vs FCFS, per difficulty tier
  eval_gap.py                     HGT/iGreedy vs exact OR-Tools optimum (CP-SAT)
  optimal_solver.py               CP-SAT model (same constraints as the MDP)
  inspect_scenario.py             Dump one scenario + solved schedule + constraint checks
  verify_solver.py                Brute-force check that CP-SAT matches env's achievable set

simulation/
  run_simulation.py               Runs a full episode, then animates the resulting schedule

configs/default.yaml              Model + PPO + training hyperparameters
train.py                          Training entry point
```

## Training

```bash
PYTHONPATH=. python train.py --config configs/default.yaml --output results_v3
```

Resume from a checkpoint:

```bash
PYTHONPATH=. python train.py --output results_v3 --resume results_v3/checkpoint_30000.pt
```

Monitor with TensorBoard:

```bash
tensorboard --logdir results_v3/tb
```

## Evaluation

Fast comparison (HGT vs iGreedy vs FCFS, no exact solver):

```bash
PYTHONPATH=. python -m evaluation.eval --checkpoint results_v3/checkpoint_best.pt --n-scenarios 100
```

Optimality gap against the exact CP-SAT solution:

```bash
PYTHONPATH=. python -m evaluation.eval_gap --checkpoint results_v3/checkpoint_best.pt --n_scenarios 100 --time_limit 30 --seed 42
```

Inspect one scenario's exact solved schedule and verify constraints hold:

```bash
PYTHONPATH=. python -m evaluation.inspect_scenario --tier hard --index 0 --seed 42
```

## Simulation

Run and animate a scenario (HGT and/or iGreedy):

```bash
PYTHONPATH=. python simulation/run_simulation.py --mode both --difficulty hard --seed 42 --checkpoint results_v3/checkpoint_best.pt
```

Use `--no-gui` to print the schedule and metrics without opening the animation window,
and `--n_vehicles N` to control scenario size.

## Results

Mean seconds above the exact optimum W\* (CP-SAT, same constraints as the MDP), over
100 scenarios per tier:

| Tier    | HGT above W\* | iGreedy above W\* | HGT improvement |
|---------|---------------|--------------------|------------------|
| Easy    | 0.032s        | 0.079s             | 60%              |
| Medium  | 0.199s        | 0.582s             | 66%              |
| Hard    | 0.434s        | 1.913s             | 77%              |
| Overall | 0.222s        | 0.858s             | 74%              |

See `NOTES.md` for training diagnostics, bugs found and fixed, and open items.
