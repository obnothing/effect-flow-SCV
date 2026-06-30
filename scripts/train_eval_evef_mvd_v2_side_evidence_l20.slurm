#!/usr/bin/env bash
#SBATCH --job-name=evef_v2_l20
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=72:00:00
#SBATCH --output=logs/slurm_evef_v2_l20_%j.out
#SBATCH --error=logs/slurm_evef_v2_l20_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs data/reports

set +u
source ~/.bashrc || true
conda activate correlascan_a40
set -u

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

echo "[INFO] current dir: $(pwd)"
echo "[INFO] python: $(which python)"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
nvidia-smi || true

echo "[BJUT 1/4] check strong base chunk cache"
python scripts/check_evm_bert_chunk_features.py \
  --feature_dir data/features/continued_dive_evm_bert_bjut_random_stride256 \
  --data_dir data/processed/BJUT_SC01_random_split \
  --expected_max_chunks 32 \
  --expected_feature_dim 768 \
  --expected_num_labels 10 \
  --output data/reports/check_bjut_evef_mvd_v2_base_features.txt

echo "[BJUT 2/4] check effect-flow semantic cache"
python scripts/check_effect_flow_semantic_features.py \
  --semantic_dir data/features/effect_flow_semantics/bjut_random_stride256_max32 \
  --feature_dir data/features/continued_dive_evm_bert_bjut_random_stride256 \
  --data_dir data/processed/BJUT_SC01_random_split \
  --expected_max_chunks 32 \
  --expected_efpp_dim 22 \
  --expected_etp_dim 16 \
  --expected_relation_dim 6 \
  --expected_global_template_dim 17 \
  --expected_num_labels 10 \
  --output data/reports/check_bjut_evef_mvd_v2_semantic_features.txt

echo "[BJUT 3/4] single-GPU sanity"
python src/train_chunk_mil.py \
  --config configs/sanity_bjut_evef_mvd_v2_side_evidence.yaml

echo "[BJUT 4/4] full train + threshold calibration + test"
python src/train_chunk_mil.py \
  --config configs/train_bjut_evef_mvd_v2_side_evidence.yaml
python src/evaluate_chunk_mil.py \
  --config configs/train_bjut_evef_mvd_v2_side_evidence.yaml \
  --checkpoint checkpoints/train_bjut_evef_mvd_v2_side_evidence/best_macro_f1.pt \
  --split valid \
  --threshold_search global_and_per_label
python src/evaluate_chunk_mil.py \
  --config configs/train_bjut_evef_mvd_v2_side_evidence.yaml \
  --checkpoint checkpoints/train_bjut_evef_mvd_v2_side_evidence/best_macro_f1.pt \
  --split test \
  --threshold_file results/train_bjut_evef_mvd_v2_side_evidence/per_label_thresholds_valid.json \
  --global_threshold_file results/train_bjut_evef_mvd_v2_side_evidence/valid_global_threshold_scan.json \
  --save_predictions

echo "[DIVE 1/4] check strong base chunk cache"
python scripts/check_evm_bert_chunk_features.py \
  --feature_dir data/features/continued_dive_evm_bert_dive_random_stride256_max64 \
  --data_dir data/processed/DIVE_random_split \
  --expected_max_chunks 64 \
  --expected_feature_dim 768 \
  --expected_num_labels 8 \
  --output data/reports/check_dive_evef_mvd_v2_base_features.txt

echo "[DIVE 2/4] check effect-flow semantic cache"
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
  --output data/reports/check_dive_evef_mvd_v2_semantic_features.txt

echo "[DIVE 3/4] single-GPU sanity"
python src/train_chunk_mil.py \
  --config configs/sanity_dive_evef_mvd_v2_side_evidence.yaml

echo "[DIVE 4/4] full train + threshold calibration + test"
python src/train_chunk_mil.py \
  --config configs/train_dive_evef_mvd_v2_side_evidence.yaml
python src/evaluate_chunk_mil.py \
  --config configs/train_dive_evef_mvd_v2_side_evidence.yaml \
  --checkpoint checkpoints/train_dive_evef_mvd_v2_side_evidence/best_macro_f1.pt \
  --split valid \
  --threshold_search global_and_per_label
python src/evaluate_chunk_mil.py \
  --config configs/train_dive_evef_mvd_v2_side_evidence.yaml \
  --checkpoint checkpoints/train_dive_evef_mvd_v2_side_evidence/best_macro_f1.pt \
  --split test \
  --threshold_file results/train_dive_evef_mvd_v2_side_evidence/per_label_thresholds_valid.json \
  --global_threshold_file results/train_dive_evef_mvd_v2_side_evidence/valid_global_threshold_scan.json \
  --save_predictions

echo "[OK] EVEF-MVD v2.1 single-GPU L20 pipeline finished."
