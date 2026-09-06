#!/usr/bin/env bash
#SBATCH --job-name=main6-label-cal
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/main6_label_calibration_%j.out
#SBATCH --error=logs/main6_label_calibration_%j.err

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source "${HOME}/.bashrc"
conda activate correlascan_a40
set -u
mkdir -p logs
test "${ALLOW_TEST:-0}" != "1"
bash scripts/run_main6_label_calibration.sh
