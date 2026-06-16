#!/usr/bin/env bash
#SBATCH --job-name=evmbert_weighted
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/slurm_evmbert_weighted_all_%j.out
#SBATCH --error=logs/slurm_evmbert_weighted_all_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc
conda activate correlascan_a40
mkdir -p logs

echo "[STAGE 9A] finalize pretraining report"
python scripts/finalize_pretrain_report.py

echo "[STAGE 9A] check pretrained EVM-BERT artifacts"
python scripts/check_evm_bert_pretrained_model.py \
  --config configs/train_evm_bert_weighted_server.yaml

echo "[STAGE 9A] sanity fine-tuning"
python src/train.py \
  --config configs/sanity_evm_bert_weighted_server.yaml

echo "[STAGE 9A] full DDP fine-tuning"
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=2 \
  src/train.py \
  --config configs/train_evm_bert_weighted_server.yaml

echo "[STAGE 9A] strict test evaluation with validation-selected best_macro checkpoint"
python src/evaluate.py \
  --config configs/train_evm_bert_weighted_server.yaml \
  --checkpoint checkpoints/train_evm_bert_weighted/best_macro_f1.pt \
  --split test \
  --threshold auto

echo "[STAGE 9A] done"
