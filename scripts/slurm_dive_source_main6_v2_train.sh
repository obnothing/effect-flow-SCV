#!/usr/bin/env bash
#SBATCH --job-name=source_m6_v2_train
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/source_m6_v2_train_%j.out
#SBATCH --error=logs/source_m6_v2_train_%j.err
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
source ~/.bashrc
set -u
conda activate correlascan_a40
mkdir -p logs
bash scripts/run_dive_source_main6_v2.sh train
