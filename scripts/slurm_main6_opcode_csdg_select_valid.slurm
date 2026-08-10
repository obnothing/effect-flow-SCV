#!/usr/bin/env bash
#SBATCH --job-name=main6_csdg_select
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/main6_csdg_select_%j.out
#SBATCH --error=logs/main6_csdg_select_%j.err

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc
conda activate correlascan_a40
set -u
mkdir -p logs
python scripts/select_main6_opcode_csdg.py \
  --config configs/train_main6_opcode_csdg.yaml \
  --require-ablation
