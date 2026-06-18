#!/usr/bin/env bash
#SBATCH --job-name=stage11_random_stride256
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/slurm_stage11_random_stride256_%j.out
#SBATCH --error=logs/slurm_stage11_random_stride256_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc
conda activate correlascan_a40
mkdir -p logs

echo "[Stage 11] create random BJUT split"
python scripts/create_bjut_random_split.py \
  --source_data_dir data/processed/BJUT_SC01 \
  --output_data_dir data/processed/BJUT_SC01_random_split \
  --seed 42 \
  --report_path data/reports/bjut_sc01_random_split_report.txt

echo "[Stage 11] check random split leakage"
python scripts/check_split_leakage.py \
  --data_dir data/processed/BJUT_SC01_random_split \
  --output data/reports/split_leakage_random_split_report.txt

echo "[Stage 11] apply MLSMOTE-compatible oversampling on random train"
python scripts/apply_mlsmote.py \
  --config configs/mlsmote_random_split.yaml

echo "[Stage 11] extract random-split stride=256 masked-mean chunk features"
python scripts/extract_evm_bert_chunk_features.py \
  --config configs/extract_evm_bert_chunk_features_mean_stride256_random_split.yaml

echo "[Stage 11] check random-split feature cache"
python scripts/check_evm_bert_chunk_features.py \
  --feature_dir data/features/evm_bert_chunks_mean_stride256_random_split \
  --data_dir data/processed/BJUT_SC01_random_split \
  --expected_max_chunks 32 \
  --expected_feature_dim 768 \
  --output data/reports/check_evm_bert_chunk_features_mean_stride256_random_split_report.txt

echo "[Stage 11] sanity train random-split stride=256 MIL"
python src/train_chunk_mil.py \
  --config configs/sanity_evm_bert_chunk_mil_mean_stride256_labelattn_weighted_random_split_server.yaml

echo "[Stage 11] full train random-split stride=256 MIL"
python src/train_chunk_mil.py \
  --config configs/train_evm_bert_chunk_mil_mean_stride256_labelattn_weighted_random_split_server.yaml

echo "[Stage 11] validation threshold search"
python src/evaluate_chunk_mil.py \
  --config configs/train_evm_bert_chunk_mil_mean_stride256_labelattn_weighted_random_split_server.yaml \
  --checkpoint checkpoints/train_evm_bert_chunk_mil_mean_stride256_labelattn_weighted_random_split/best_macro_f1.pt \
  --split valid \
  --threshold_search global_and_per_label \
  --output_prefix threshold_calibration

echo "[Stage 11] strict test on random split"
python src/evaluate_chunk_mil.py \
  --config configs/train_evm_bert_chunk_mil_mean_stride256_labelattn_weighted_random_split_server.yaml \
  --checkpoint checkpoints/train_evm_bert_chunk_mil_mean_stride256_labelattn_weighted_random_split/best_macro_f1.pt \
  --split test \
  --threshold_file results/train_evm_bert_chunk_mil_mean_stride256_labelattn_weighted_random_split/per_label_thresholds_valid.json \
  --global_threshold_file results/train_evm_bert_chunk_mil_mean_stride256_labelattn_weighted_random_split/valid_global_threshold_scan.json \
  --output_prefix threshold_calibration

echo "[Stage 11] done"

