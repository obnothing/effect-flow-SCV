#!/usr/bin/env bash
#SBATCH --job-name=main6_csdg_final
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/main6_csdg_final_%j.out
#SBATCH --error=logs/main6_csdg_final_%j.err

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc
conda activate correlascan_a40
set -u
mkdir -p logs
test "${ALLOW_TEST:-0}" = "1"
test -f results/main6_opcode_csdg/selection.json
bash scripts/run_main6_opcode_csdg.sh final
