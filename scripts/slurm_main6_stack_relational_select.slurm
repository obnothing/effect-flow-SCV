#!/usr/bin/env bash
#SBATCH --job-name=main6-stack-select
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=logs/main6_stack_relational_select_%j.out
#SBATCH --error=logs/main6_stack_relational_select_%j.err

set -eo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "${HOME}/.bashrc"
conda activate correlascan_a40
set -u

python scripts/select_main6_stack_relational_valid.py \
  --config configs/train_main6_stack_relational.yaml
