#!/usr/bin/env bash
# Deprecated compatibility wrapper. The EVEF-MVD mainline is
# scripts/stage_evef_mvd_pipeline.slurm; keep this entry point so old notes do
# not accidentally run stale v1 feature-cache paths.
#SBATCH --job-name=evef_mvd_downstream
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm_evef_mvd_downstream_%j.out
#SBATCH --error=logs/slurm_evef_mvd_downstream_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

set +u
source ~/.bashrc || true
conda activate correlascan_a40 || true
set -u

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export CONDA_ENV="${CONDA_ENV:-correlascan_a40}"

bash scripts/stage_evef_mvd_pipeline.sh --mode downstream --dataset both "$@"
