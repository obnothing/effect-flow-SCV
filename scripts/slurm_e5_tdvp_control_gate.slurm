#!/usr/bin/env bash
#SBATCH --job-name=e5-tdvp-control
#SBATCH --output=logs/e5_tdvp_control_%j.out
#SBATCH --error=logs/e5_tdvp_control_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
source "${HOME}/.bashrc"
conda activate pytorch-2.1.1
set -u

bash scripts/run_e5_tdvp_control_gate.sh

