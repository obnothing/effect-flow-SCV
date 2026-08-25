#!/usr/bin/env bash
#SBATCH --job-name=main6-stack-pretrain
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/main6_stack_relational_pretrain_%j.out
#SBATCH --error=logs/main6_stack_relational_pretrain_%j.err

# Do not enable nounset before sourcing bashrc: some cluster bashrc files
# reference BASHRCSOURCED before defining it.
set -eo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "${HOME}/.bashrc"
conda activate correlascan_a40
set -u

python scripts/pretrain_main6_stack_aware_mlm.py \
  --config configs/train_main6_stack_relational.yaml
