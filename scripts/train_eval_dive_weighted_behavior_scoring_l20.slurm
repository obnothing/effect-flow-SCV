#!/usr/bin/env bash
#SBATCH --job-name=dive_beh_score
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=72:00:00
#SBATCH --output=logs/slurm_dive_beh_score_%j.out
#SBATCH --error=logs/slurm_dive_beh_score_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs data/reports

set +u
source ~/.bashrc || true
conda activate correlascan_a40
set -u

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

CONFIG="configs/train_dive_evef_mvd_v2_weighted_behavior_scoring.yaml"
SANITY_CONFIG="configs/sanity_dive_evef_mvd_v2_weighted_behavior_scoring.yaml"
CHECKPOINT="checkpoints/train_dive_evef_mvd_v2_weighted_behavior_scoring/best_macro_f1.pt"
RESULT_DIR="results/train_dive_evef_mvd_v2_weighted_behavior_scoring"

echo "[INFO] current dir: $(pwd)"
echo "[INFO] python: $(which python)"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
nvidia-smi || true

echo "[DIVE 1/6] check strong base chunk cache"
python scripts/check_evm_bert_chunk_features.py \
  --feature_dir data/features/continued_dive_evm_bert_dive_random_stride256_max64 \
  --data_dir data/processed/DIVE_random_split \
  --expected_max_chunks 64 \
  --expected_feature_dim 768 \
  --expected_num_labels 8 \
  --output data/reports/check_dive_weighted_behavior_base_features.txt

echo "[DIVE 2/6] check effect-flow semantic cache"
python scripts/check_effect_flow_semantic_features.py \
  --semantic_dir data/features/effect_flow_semantics/dive_random_stride256_max64 \
  --feature_dir data/features/continued_dive_evm_bert_dive_random_stride256_max64 \
  --data_dir data/processed/DIVE_random_split \
  --expected_max_chunks 64 \
  --expected_efpp_dim 22 \
  --expected_etp_dim 16 \
  --expected_relation_dim 6 \
  --expected_global_template_dim 17 \
  --expected_num_labels 8 \
  --output data/reports/check_dive_weighted_behavior_semantic_features.txt

echo "[DIVE 3/6] sanity train"
python src/train_chunk_mil.py \
  --config "${SANITY_CONFIG}"

echo "[DIVE 4/6] full train"
python src/train_chunk_mil.py \
  --config "${CONFIG}"

echo "[DIVE 5/6] validation threshold calibration and strict test"
python src/evaluate_chunk_mil.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --split valid \
  --threshold_search global_and_per_label

python src/evaluate_chunk_mil.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --split test \
  --threshold_file "${RESULT_DIR}/per_label_thresholds_valid.json" \
  --global_threshold_file "${RESULT_DIR}/valid_global_threshold_scan.json" \
  --save_predictions

echo "[DIVE 6/6] behavior contribution summary"
python scripts/summarize_behavior_contributions.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --split test \
  --threshold_file "${RESULT_DIR}/per_label_thresholds_valid.json" \
  --global_threshold_file "${RESULT_DIR}/valid_global_threshold_scan.json" \
  --output_prefix "${RESULT_DIR}/behavior_contribution_summary"

echo "[OK] DIVE weighted behavior scoring pipeline finished."
