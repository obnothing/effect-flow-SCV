#!/usr/bin/env bash
#SBATCH --job-name=mil_eval
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/slurm_mil_eval_%j.out
#SBATCH --error=logs/slurm_mil_eval_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc
conda activate correlascan_a40
mkdir -p logs

python src/evaluate_chunk_mil.py \
  --config configs/train_evm_bert_chunk_mil_weighted_server.yaml \
  --checkpoint checkpoints/train_evm_bert_chunk_mil_weighted/best_macro_f1.pt \
  --split test \
  --threshold auto

