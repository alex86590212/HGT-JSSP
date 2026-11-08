# R-GCN Admission Control for Gap-Free Intersection Management

This is the source code of the research work on gap-free whole-route admission at an unsignalized intersection with a Relational GCN (R-GCN) policy trained with PPO. The controller reserves every conflict zone along a vehicle's route before the vehicle enters, and the policy decides which pending vehicle to admit, or whether to defer. The paper source is in `paper_rgcn/`; the details of the R-GCN package are in [admission_rgcn/README.md](admission_rgcn/README.md).

## Installation

Run everything from the repository root.
```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. python3 -m pytest admission_rgcn/tests
```
`torch-geometric` is required (`RGCNConv`).

## Intersection Manager Training

Use the module `admission_rgcn.train` to run the PPO training of the intersection manager. Traffic is generated on the fly (4x4 grid of 16 conflict zones, Poisson arrivals, 60 s episodes), so no data needs to be prepared. For instance, you can run a basic training with the default config using the following command,
```
PYTHONPATH=. python3 -m admission_rgcn.train \
  --config admission_rgcn/config.yaml \
  --output runs/admission_rgcn
```
Checkpoints (`checkpoint_last.pt`, `checkpoint_best.pt`) and logs (`train_log.csv`, `eval_log.csv`, `tb/`) are saved in the output directory. Experiment-related arguments are listed below.

| Argument | Type | Description |
| :------- | :--- | :---------- |
| `config` | str | YAML file with environment, model, PPO and evaluation settings (default: `admission_rgcn/config.yaml`) |
| `output` | str | directory to store checkpoints and logs (default: `runs/admission_rgcn`) |
| `resume` | str or None | if specified, will continue training from this `checkpoint_last.pt` (default: None) |
| `episodes` | int | overrides `train.num_episodes` (default: 4000) |
| `workers` | int | number of rollout processes (default: 4) |
| `no-defer` | flag | train an order-only policy without the defer action |
| `seed` | int | seed for initialization and the training scenario stream |

The training return is `reward_scale * (W_igreedy - W_policy)` on the same arrivals, with arrival rates drawn from [0.1, 1.2]. View the training curves with `tensorboard --logdir runs/admission_rgcn/tb`.

## Evaluation

Evaluate a trained checkpoint against the baselines with `admission_rgcn.evaluate`. The policy runs deterministically on scenarios shared with every baseline, so comparisons are paired. For example,
```
PYTHONPATH=. python3 -m admission_rgcn.evaluate \
  --checkpoint runs/admission_rgcn/checkpoint_best.pt \
  --output-csv runs/admission_rgcn/eval_final.csv
```
The default protocol uses 8 arrival rates (0.10 to 1.20), seeds 0 to 2, and 15 scenarios per seed and rate.

| Argument | Type | Description |
| :------- | :--- | :---------- |
| `checkpoint` | str | trained checkpoint to evaluate |
| `rates`, `seeds`, `n-scenarios` | list, list, int | evaluation grid |
| `baselines` | list | any of `igreedy backpressure lifo` (iGreedy is always included) |
| `wstar` | flag | also solve the restricted-information optimum `W*` with CP-SAT (slow) |
| `wstar-time-limit` | float | seconds per CP-SAT solve (default: 5) |
| `workers` | int | number of evaluation processes (default: 4) |
| `output-csv` | str | per-rate results |

## Baseline and Optimal Solutions

The baselines (iGreedy, Backpressure and LIFO) run in the same environment as the learned policy. To evaluate the baselines alone, without a trained model,
```
PYTHONPATH=. python3 evaluation/eval_gapfree_admission.py \
  --output-csv gapfree_admission_results.csv
```
The optimal solution `W*` is computed with CP-SAT by passing `--wstar` to `admission_rgcn.evaluate`.

## Ablation Study

The ablation compares policies trained with and without the defer action, over several training seeds. Finished runs are skipped and interrupted runs resume, so the command can be re-run.
```
PYTHONPATH=. python3 -m admission_rgcn.run_ablation \
  --out runs/ablation \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --episodes 4000 --parallel 2 --workers 4
PYTHONPATH=. python3 -m admission_rgcn.summarize_ablation \
  --dir runs/ablation --output-csv runs/ablation/summary.csv
```
Each run is stored in `runs/ablation/<defer|nodefer>_s<seed>/`. Best checkpoints are selected on scenario seed 0, so the sweep evaluates on seeds 1 to 3.

## Paper Figures and Tables

Run these from `paper_rgcn/`, since the scripts use relative paths.
```
cd paper_rgcn
python3 make_scaling_plot.py
python3 build_tables.py
```

## Repository Layout

* `admission_rgcn/`: R-GCN policy, PPO training, evaluation and ablation.
* `dynamic_scheduler/`: gap-free admission environment, traffic generator, baselines and the CP-SAT solver for `W*`.
* `evaluation/`: evaluation scripts for the baselines.
* `paper_rgcn/`: paper source, scaling plot and table scripts.
