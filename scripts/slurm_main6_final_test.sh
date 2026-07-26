#!/usr/bin/env bash
#SBATCH --job-name=main6_final_test
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/main6_final_test_%j.out
#SBATCH --error=logs/main6_final_test_%j.err

cd "${SLURM_SUBMIT_DIR:?Submit this job with sbatch from the project root}"
source ~/.bashrc
conda activate correlascan_a40
set -euo pipefail
mkdir -p logs

test -f results/main6_random_090_mlm8/validation_selection.json
ALLOW_TEST=1 bash scripts/run_main6_random_090.sh final
