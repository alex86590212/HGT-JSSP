#!/bin/bash
#SBATCH --job-name=hgt-jssp-dynamic-train
#SBATCH --time=50:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16000
#SBATCH --output=hgt-jssp-dynamic-train-%j.out
#SBATCH --error=hgt-jssp-dynamic-train-%j.err

set -euo pipefail

module purge
module load Python/3.10.4-GCCcore-11.3.0

PROJECT_DIR="${HOME}/test/HGT-JSSP"
VENV_DIR="${HOME}/test/.env-cuda"

source "${VENV_DIR}/bin/activate"
cd "${PROJECT_DIR}"
export PYTHONPATH=.
export PYTHONUNBUFFERED=1

# CPU-only: dynamic rollout is ~1200 sequential small forwards/episode; GPU
# launch/transfer overhead dominates. Hide GPUs so PyTorch never touches CUDA.
export CUDA_VISIBLE_DEVICES=

# Single-thread BLAS/OpenMP, ALWAYS, regardless of --cpus-per-task above.
# Confirmed on-cluster: bumping OMP_NUM_THREADS (and the other three) to
# match --cpus-per-task made training SLOWER, not faster — dynamic rollout
# is ~1000 sequential tiny GNN forwards/episode (~188-node graph), and each
# individual op is too small for multi-threading to pay for its own
# coordination overhead. Do NOT bump these four to use more cores.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# The extra cores requested via --cpus-per-task above are instead used via
# training.num_workers in configs/default_dynamic.yaml: the trainer spawns
# that many separate single-threaded PROCESSES, each collecting one whole
# episode concurrently (episodes within a PPO rollout batch are independent
# given fixed policy weights). Keep --cpus-per-task matched to num_workers.
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Working dir: $(pwd)"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-?}"
python3 --version
python3 -c "import torch; print(f'torch={torch.__version__}, cuda={torch.cuda.is_available()}')"

python3 -u train_dynamic.py --output results_dynamic

deactivate

echo "Training finished: $(date)"
