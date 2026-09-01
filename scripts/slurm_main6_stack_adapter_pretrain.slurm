#!/usr/bin/env bash
#SBATCH --job-name=main6-stack-adapter-mlm
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/main6_stack_adapter_pretrain_%j.out
#SBATCH --error=logs/main6_stack_adapter_pretrain_%j.err

set -eo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "${HOME}/.bashrc"
conda activate correlascan_a40
set -u

python scripts/pretrain_main6_stack_adapter_mlm.py \
  --config configs/train_main6_stack_adapter.yaml
