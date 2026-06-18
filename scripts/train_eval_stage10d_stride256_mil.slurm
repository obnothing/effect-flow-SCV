#!/usr/bin/env bash
#SBATCH --job-name=stage10d_stride256
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/slurm_stage10d_stride256_%j.out
#SBATCH --error=logs/slurm_stage10d_stride256_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc
conda activate correlascan_a40
mkdir -p logs

echo "[Stage 10D] extract stride=256 masked-mean chunk features"
python scripts/extract_evm_bert_chunk_features.py \
  --config configs/extract_evm_bert_chunk_features_mean_stride256.yaml

echo "[Stage 10D] check stride=256 feature cache"
python scripts/check_evm_bert_chunk_features.py \
  --feature_dir data/features/evm_bert_chunks_mean_stride256 \
  --coverage_baseline_dir data/features/evm_bert_chunks_mean \
  --expected_max_chunks 32 \
  --expected_feature_dim 768 \
  --output data/reports/check_evm_bert_chunk_features_mean_stride256_report.txt

echo "[Stage 10D] sanity train stride=256 label-wise gated attention MIL"
python src/train_chunk_mil.py \
  --config configs/sanity_evm_bert_chunk_mil_mean_stride256_labelattn_weighted_server.yaml

echo "[Stage 10D] full train stride=256 label-wise gated attention MIL"
python src/train_chunk_mil.py \
  --config configs/train_evm_bert_chunk_mil_mean_stride256_labelattn_weighted_server.yaml

echo "[Stage 10D] validation-selected threshold calibration"
python src/evaluate_chunk_mil.py \
  --config configs/train_evm_bert_chunk_mil_mean_stride256_labelattn_weighted_server.yaml \
  --checkpoint checkpoints/train_evm_bert_chunk_mil_mean_stride256_labelattn_weighted/best_macro_f1.pt \
  --split valid \
  --threshold_search global_and_per_label \
  --output_prefix threshold_calibration

python src/evaluate_chunk_mil.py \
  --config configs/train_evm_bert_chunk_mil_mean_stride256_labelattn_weighted_server.yaml \
  --checkpoint checkpoints/train_evm_bert_chunk_mil_mean_stride256_labelattn_weighted/best_macro_f1.pt \
  --split test \
  --threshold_file results/train_evm_bert_chunk_mil_mean_stride256_labelattn_weighted/per_label_thresholds_valid.json \
  --global_threshold_file results/train_evm_bert_chunk_mil_mean_stride256_labelattn_weighted/valid_global_threshold_scan.json \
  --output_prefix threshold_calibration

echo "[Stage 10D] summarize Stage 10 ablation"
python scripts/summarize_stage10_ablation.py

echo "[Stage 10D] done"
