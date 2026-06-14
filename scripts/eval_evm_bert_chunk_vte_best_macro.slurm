#!/usr/bin/env bash
#SBATCH --job-name=eval_evmbert_chunk_vte
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/slurm_eval_evmbert_chunk_vte_%j.out
#SBATCH --error=logs/slurm_eval_evmbert_chunk_vte_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc
conda activate correlascan_a40
mkdir -p logs

python src/evaluate.py \
  --config configs/train_evm_bert_chunk_vte_server.yaml \
  --checkpoint checkpoints/train_evm_bert_chunk_vte/best_macro_f1.pt \
  --split test \
  --threshold auto
