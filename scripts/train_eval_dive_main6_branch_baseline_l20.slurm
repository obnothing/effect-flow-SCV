#!/usr/bin/env bash
#SBATCH --job-name=dive_main6_base
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm_dive_main6_branch_baseline_%j.out
#SBATCH --error=logs/slurm_dive_main6_branch_baseline_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs data/reports results/dive_main6_branch_baseline

set +u
source ~/.bashrc || true
conda activate correlascan_a40
set -u

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

RUN_SANITY="${RUN_SANITY:-0}"
CONFIG="configs/train_dive_main6_branch_baseline.yaml"
CHECKPOINT_DIR="checkpoints/dive_main6_branch_baseline"
RESULT_DIR="results/dive_main6_branch_baseline"

echo "[INFO] current dir: $(pwd)"
echo "[INFO] RUN_SANITY=${RUN_SANITY}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
nvidia-smi || true

echo "[1/5] derive and audit DIVE Main-6 data view"
python scripts/build_dive_main6_dataset.py --overwrite

echo "[2/5] validate source feature cache"
python scripts/check_evm_bert_chunk_features.py \
  --feature_dir data/features/continued_dive_evm_bert_dive_random_stride256_max64 \
  --data_dir data/processed/DIVE_random_split \
  --expected_max_chunks 64 \
  --expected_feature_dim 768 \
  --expected_num_labels 8 \
  --output data/reports/dive_main6_source_feature_cache.txt

echo "[3/5] validate source semantic cache"
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
  --output data/reports/dive_main6_source_semantic_cache.txt

TRAIN_ARGS=(--config "${CONFIG}")
if [[ "${RUN_SANITY}" == "1" ]]; then
  TRAIN_ARGS+=(
    --override "checkpoint_dir=checkpoints/dive_main6_branch_baseline_sanity"
    --override "result_dir=results/dive_main6_branch_baseline_sanity"
    --override "experiment_name=dive_main6_branch_baseline_sanity"
    --override "epochs=2"
    --override "early_stopping_patience=2"
    --override "debug_num_train_samples=512"
    --override "debug_num_valid_samples=512"
  )
  CHECKPOINT_DIR="checkpoints/dive_main6_branch_baseline_sanity"
  RESULT_DIR="results/dive_main6_branch_baseline_sanity"
fi

echo "[4/5] train Main-6 baseline"
python src/train_chunk_mil.py "${TRAIN_ARGS[@]}"

echo "[5/5] validation calibration and strict test evaluation"
python src/evaluate_chunk_mil.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT_DIR}/best_macro_f1.pt" \
  --split valid \
  --threshold_search global_and_per_label \
  --result_dir_override "${RESULT_DIR}/best_macro_f1"

python src/evaluate_chunk_mil.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT_DIR}/best_macro_f1.pt" \
  --split test \
  --threshold_file "${RESULT_DIR}/best_macro_f1/per_label_thresholds_valid.json" \
  --global_threshold_file "${RESULT_DIR}/best_macro_f1/valid_global_threshold_scan.json" \
  --result_dir_override "${RESULT_DIR}/best_macro_f1" \
  --save_predictions

echo "[OK] Main-6 baseline finished: ${RESULT_DIR}/best_macro_f1"
