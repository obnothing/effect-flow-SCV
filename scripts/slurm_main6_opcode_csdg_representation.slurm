#!/usr/bin/env bash
#SBATCH --job-name=main6_csdg_repr
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/main6_csdg_repr_%j.out
#SBATCH --error=logs/main6_csdg_repr_%j.err

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc
conda activate correlascan_a40
set -u
mkdir -p logs
bash scripts/run_main6_opcode_csdg.sh extract-local
bash scripts/run_main6_opcode_csdg.sh extract-instruction-value
bash scripts/run_main6_opcode_csdg.sh train-local
