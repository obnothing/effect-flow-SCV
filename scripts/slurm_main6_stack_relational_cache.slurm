#!/usr/bin/env bash
#SBATCH --job-name=main6-stack-cache
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/main6_stack_relational_cache_%j.out
#SBATCH --error=logs/main6_stack_relational_cache_%j.err

set -eo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "${HOME}/.bashrc"
conda activate correlascan_a40
set -u

python scripts/extract_main6_stack_relational_features.py \
  --config configs/train_main6_stack_relational.yaml \
  --splits train valid

python scripts/check_main6_stack_relational_feature_cache.py \
  --config configs/train_main6_stack_relational.yaml
