#!/usr/bin/env bash
#SBATCH --job-name=stage10b_mil
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/slurm_stage10b_mil_%j.out
#SBATCH --error=logs/slurm_stage10b_mil_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc
conda activate correlascan_a40
mkdir -p logs

echo "[Stage 10B] extract masked-mean chunk features"
python scripts/extract_evm_bert_chunk_features.py \
  --config configs/extract_evm_bert_chunk_features_mean.yaml

echo "[Stage 10B] check masked-mean feature cache"
python scripts/check_evm_bert_chunk_features.py \
  --feature_dir data/features/evm_bert_chunks_mean \
  --output data/reports/check_evm_bert_chunk_features_mean_report.txt

echo "[Stage 10B] train masked-mean top-k MIL"
python src/train_chunk_mil.py \
  --config configs/train_evm_bert_chunk_mil_mean_topk_weighted_server.yaml

echo "[Stage 10B] evaluate masked-mean top-k MIL"
python src/evaluate_chunk_mil.py \
  --config configs/train_evm_bert_chunk_mil_mean_topk_weighted_server.yaml \
  --checkpoint checkpoints/train_evm_bert_chunk_mil_mean_topk_weighted/best_macro_f1.pt \
  --split test \
  --threshold auto

echo "[Stage 10B] train label-wise gated attention MIL sanity"
python src/train_chunk_mil.py \
  --config configs/sanity_evm_bert_chunk_mil_mean_labelattn_weighted_server.yaml

echo "[Stage 10B] train label-wise gated attention MIL full"
python src/train_chunk_mil.py \
  --config configs/train_evm_bert_chunk_mil_mean_labelattn_weighted_server.yaml

echo "[Stage 10B] evaluate label-wise gated attention MIL"
python src/evaluate_chunk_mil.py \
  --config configs/train_evm_bert_chunk_mil_mean_labelattn_weighted_server.yaml \
  --checkpoint checkpoints/train_evm_bert_chunk_mil_mean_labelattn_weighted/best_macro_f1.pt \
  --split test \
  --threshold auto

echo "[Stage 10B] summarize ablation"
python scripts/summarize_stage10_ablation.py

echo "[Stage 10B] done"
