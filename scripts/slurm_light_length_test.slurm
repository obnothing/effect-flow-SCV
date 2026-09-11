#!/usr/bin/env bash
#SBATCH --job-name=main6-light-length
#SBATCH --output=logs/main6_light_length_%j.out
#SBATCH --error=logs/main6_light_length_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
source "${HOME}/.bashrc"
conda activate pytorch-2.1.1
set -u

bash scripts/run_light_length_test.sh

