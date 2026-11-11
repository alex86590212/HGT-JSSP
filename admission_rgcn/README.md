# admission_rgcn

R-GCN policy for gap-free, whole-route admission at an unsignalized 4x4 intersection. The controller reserves every conflict zone along a vehicle's route before it enters. At each controller event the policy admits one feasible pending vehicle or defers (`DEFER`: wait 0.5 s and look again). It is trained with PPO to minimize the mean per-vehicle waiting time `W` over a 60 s episode with Poisson arrivals, and compared with iGreedy, Backpressure and LIFO and with a CP-SAT reference `W*`.

Installation, training, evaluation and ablation commands are in the [root README](../README.md). Run everything from the repository root with `PYTHONPATH=.`.

## Layout

| File | Purpose |
|---|---|
| `config.yaml` | Environment, model, PPO and evaluation settings |
| `graph.py` | Builds the heterogeneous state graph (pending, reserved and zone nodes; 7 edge types) |
| `policy.py` | R-GCN encoder and actor-critic policy (vehicle logits plus a defer logit) |
| `rollout.py` | Runs one episode with a policy and records transitions for PPO |
| `ppo.py` | GAE and the PPO update |
| `train.py` | Parallel-rollout PPO training |
| `evaluate.py` | Paired evaluation against the baselines (and optionally `W*`) |
| `run_ablation.py` | Seed sweep with and without `DEFER` |
| `summarize_ablation.py` | Aggregates the sweep over seeds |
| `common.py` | Shared helpers |
| `tests/` | Rollout tests |

The environment, the baselines and the `W*` solver live in `../dynamic_scheduler/`.

## Notes

- The graph and the policy input sizes are tied to the 4x4 layout (16 zones). A different intersection needs changes in `graph.py` and `dynamic_scheduler/data/traffic_generator.py`.
- Checkpoints store their config, so `evaluate.py` rebuilds the model from the checkpoint and ignores `config.yaml`.
- `checkpoint_best.pt` is selected on scenario seed 0. Evaluate on seeds 1 to 3 (as `run_ablation.py` does) for numbers that are independent of checkpoint selection.
