#!/usr/bin/env bash
#SBATCH --job-name=mil_all
#SBATCH --partition=gpu-a
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/slurm_mil_all_%j.out
#SBATCH --error=logs/slurm_mil_all_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc
conda activate correlascan_a40
mkdir -p logs

echo "[Stage 10A] check pretrained EVM-BERT"
python scripts/check_evm_bert_pretrained_model.py \
  --config configs/train_evm_bert_weighted_server.yaml

echo "[Stage 10A] extract chunk feature cache"
python scripts/extract_evm_bert_chunk_features.py \
  --config configs/extract_evm_bert_chunk_features.yaml

echo "[Stage 10A] check chunk feature cache"
python scripts/check_evm_bert_chunk_features.py \
  --feature_dir data/features/evm_bert_chunks

echo "[Stage 10A] sanity MIL training"
python src/train_chunk_mil.py \
  --config configs/sanity_evm_bert_chunk_mil_weighted_server.yaml

echo "[Stage 10A] full MIL training"
python src/train_chunk_mil.py \
  --config configs/train_evm_bert_chunk_mil_weighted_server.yaml

echo "[Stage 10A] strict test evaluation"
python src/evaluate_chunk_mil.py \
  --config configs/train_evm_bert_chunk_mil_weighted_server.yaml \
  --checkpoint checkpoints/train_evm_bert_chunk_mil_weighted/best_macro_f1.pt \
  --split test \
  --threshold auto

echo "[Stage 10A] done"

