# admission_rgcn

R-GCN policy for gap-free, whole-route admission at an unsignalized 4x4 intersection. The controller reserves every conflict zone along a vehicle's route before it enters. This package trains the policy with PPO and compares it with rule-based admission baselines and the CP-SAT reference `W*`. The paper source is in `../paper_rgcn/`.

The policy chooses, at each controller event, one feasible pending vehicle to admit or `DEFER` (wait 0.5 s and look again). The objective is the mean per-vehicle waiting time `W` over an episode (60 s, Poisson arrivals at rate `lambda`).

## Setup

Run everything from the repository root with `PYTHONPATH=.`.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`torch-geometric` is required (`RGCNConv`). Check that the setup works:

```bash
PYTHONPATH=. python -m pytest admission_rgcn/tests
```

## Layout

| File | Purpose |
|---|---|
| `config.yaml` | Environment, model, PPO and evaluation settings |
| `graph.py` | Builds the heterogeneous state graph (`pending`, `committed`, `zone` nodes; 7 edge types) |
| `policy.py` | `AdmissionRGCN` encoder and actor-critic `AdmissionPolicy` (vehicle logits plus a defer logit) |
| `rollout.py` | Runs one episode with a policy and records transitions for PPO |
| `ppo.py` | GAE and the PPO update |
| `train.py` | Parallel-rollout PPO training |
| `evaluate.py` | Paired evaluation against baselines (and optionally `W*`) |
| `run_ablation.py` | Seed sweep with and without `DEFER` |
| `summarize_ablation.py` | Aggregates the sweep over seeds |

The environment itself (`AdmissionEnv`, the baselines, and the `W*` solver) lives in `dynamic_scheduler/`.

## Training

```bash
PYTHONPATH=. python -m admission_rgcn.train \
    --config admission_rgcn/config.yaml --output runs/admission_rgcn
```

Options:

| Flag | Meaning |
|---|---|
| `--config` | YAML file (default `admission_rgcn/config.yaml`) |
| `--output` | Output directory (default `runs/admission_rgcn`) |
| `--resume PATH` | Continue from a `checkpoint_last.pt` |
| `--episodes N` | Override `train.num_episodes` (default 15000) |
| `--workers N` | Rollout processes (default 4) |
| `--no-defer` | Order-only policy, i.e. `env.allow_defer=false` |
| `--seed N` | Seed for initialization and the training scenario stream |

Resume an interrupted run:

```bash
PYTHONPATH=. python -m admission_rgcn.train \
    --resume runs/admission_rgcn/checkpoint_last.pt --output runs/admission_rgcn
```

Each PPO update collects `rollout_episodes` episodes with arrival rates drawn from `[rate_min, rate_max]` = [0.1, 1.2]. The return is `reward_scale * (W_igreedy - W_policy)` on the same arrivals, so the policy is rewarded for beating iGreedy.

Output directory:

- `checkpoint_last.pt` and `checkpoint_best.pt`. Best means the lowest in-training evaluation wait on scenario seed 0.
- `train_log.csv` and `eval_log.csv`.
- `tb/` for TensorBoard: `tensorboard --logdir runs/admission_rgcn/tb`.

## Evaluation

```bash
PYTHONPATH=. python -m admission_rgcn.evaluate \
    --checkpoint runs/admission_rgcn/checkpoint_best.pt \
    --output-csv runs/admission_rgcn/eval_final.csv
```

The default protocol uses 8 arrival rates (0.10 to 1.20), seeds 0 to 2, and 15 scenarios per seed and rate. The policy runs deterministically, and each scenario is shared with every baseline, so comparisons are paired.

| Flag | Meaning |
|---|---|
| `--rates`, `--seeds`, `--n-scenarios` | Evaluation grid |
| `--baselines` | Any of `igreedy edf backpressure lifo edf_prio_inherit edf_commitment_exchange` (iGreedy is always included) |
| `--wstar` | Also solve the restricted-information optimum `W*` with CP-SAT (slow) |
| `--wstar-time-limit` | Seconds per CP-SAT solve (default 5) |
| `--workers` | Evaluation processes (default 4) |
| `--output-csv` | Per-rate results |

Evaluation with `W*`:

```bash
PYTHONPATH=. python -m admission_rgcn.evaluate \
    --checkpoint runs/admission_rgcn/checkpoint_best.pt \
    --wstar --output-csv runs/admission_rgcn/eval_wstar.csv
```

Evaluate the policy against the HGT policy in `admission_rl/` on identical scenarios:

```bash
PYTHONPATH=. python compare_eval.py \
    --hgt runs/admission_rl_v2/checkpoint_best.pt \
    --rgcn runs/admission_rgcn_v1/checkpoint_best.pt \
    --output-csv runs/compare_eval.csv
```

Baselines alone, without a trained model:

```bash
PYTHONPATH=. python evaluation/eval_gapfree_admission.py --output-csv gapfree_admission_results.csv
```

## Ablation (with and without DEFER)

Train and evaluate every (variant, seed) pair. Finished runs are skipped and interrupted runs resume, so the command can be re-run.

```bash
PYTHONPATH=. python -m admission_rgcn.run_ablation --out runs/ablation \
    --seeds 0 1 2 3 4 5 6 7 8 9 --episodes 4000 --parallel 2 --workers 4
```

Each run directory is `runs/ablation/<defer|nodefer>_s<seed>/` and holds the checkpoints, logs and `eval_final.csv`. Use `--ckpt last` to evaluate the final checkpoint instead of the best one.

Aggregate over seeds (mean and std per rate):

```bash
PYTHONPATH=. python -m admission_rgcn.summarize_ablation \
    --dir runs/ablation --output-csv runs/ablation/summary.csv
```

## Paper figures and tables

Run these from `paper_rgcn/` (the scripts use relative paths):

```bash
cd paper_rgcn
python make_scaling_plot.py   # scaling_plot.pdf: waiting time vs arrival rate, bands over seeds
python build_tables.py        # LaTeX tables from ../runs/compare_eval_wstar.csv
```

## Notes

- The state graph and the policy input sizes are tied to the 4x4 layout (16 zones), so a different intersection needs changes in `graph.py` and `dynamic_scheduler/data/traffic_generator.py`.
- Checkpoints store their config, so `evaluate.py` rebuilds the model from the checkpoint and ignores `config.yaml`.
- Training uses CPU rollout workers; `train.num_workers` and `train.torch_threads` control CPU use.
- `checkpoint_best.pt` is selected on scenario seed 0, which the final evaluation also uses for its seed 0. Use seeds 1 to 3 (as `run_ablation.py` does) for numbers that are independent of checkpoint selection.
