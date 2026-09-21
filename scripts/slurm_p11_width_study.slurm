#!/usr/bin/env bash
#SBATCH --job-name=p11-width-study
#SBATCH --partition=gpu-l20
#SBATCH --nodelist=gpu-l20-9
#SBATCH --gres=gpu:l20:1
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/p11_width_study_%j.out
#SBATCH --error=logs/p11_width_study_%j.err

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
source "${HOME}/.bashrc"
conda activate correlascan_a40
set -u
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
mkdir -p logs
python -u scripts/test_p11_width_study.py
python -u scripts/test_polarity_queries.py
python -u scripts/run_p11_width_study.py
