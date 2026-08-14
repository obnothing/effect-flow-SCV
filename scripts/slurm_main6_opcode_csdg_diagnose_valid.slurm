#!/usr/bin/env bash
#SBATCH --job-name=main6_csdg_diagnose
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/main6_csdg_diagnose_%j.out
#SBATCH --error=logs/main6_csdg_diagnose_%j.err

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc
conda activate correlascan_a40
set -u
mkdir -p logs

for variant in cfg_residual cfg_stack_residual cfg_stack_storage_residual graph_mlp_control cfg_stack_sparse_topk; do
  checkpoint="checkpoints/main6_opcode_csdg/${variant}/best_macro_f1.pt"
  if [[ -f "$checkpoint" ]]; then
    python scripts/diagnose_main6_opcode_csdg.py \
      --config configs/train_main6_opcode_csdg.yaml \
      --variant "$variant"
  fi
done
