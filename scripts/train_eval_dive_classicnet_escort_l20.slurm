#!/usr/bin/env bash
#SBATCH --job-name=dive_classic_escort
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=168:00:00
#SBATCH --output=logs/slurm_dive_classicnet_escort_%j.out
#SBATCH --error=logs/slurm_dive_classicnet_escort_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs data/reports

set +u
source ~/.bashrc || true
conda activate correlascan_a40
set -u

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

CONFIG="configs/dive_classicnet_escort.yaml"
RUN_SANITY="${RUN_SANITY:-0}"
MAIN6_VARIANTS="${MAIN6_VARIANTS:-main6_current_lr100 main6_residual_lr100 main6_localglobal_lr050 main6_localglobal_lr100 main6_localglobal_lr200}"
TRANSFER_MAIN6_VARIANTS="${TRANSFER_MAIN6_VARIANTS:-main6_residual_lr100 main6_localglobal_lr050 main6_localglobal_lr100 main6_localglobal_lr200}"
FRONT_VARIANTS="${FRONT_VARIANTS:-front_r1_lr100 front_r1_lr300 front_r3_lr100 front_r3_lr300}"
BAD_VARIANTS="${BAD_VARIANTS:-bad_r1_lr100 bad_r1_lr300 bad_r3_lr100 bad_r3_lr300}"
JOINT_VARIANT="${JOINT_VARIANT:-joint8_bs128_control}"

if [[ "${RUN_SANITY}" == "1" ]]; then
  CHECKPOINT_ROOT="checkpoints/dive_classicnet_escort_sanity"
  RESULT_ROOT="results/dive_classicnet_escort_sanity"
else
  CHECKPOINT_ROOT="checkpoints/dive_classicnet_escort"
  RESULT_ROOT="results/dive_classicnet_escort"
fi
VALIDATION_ROOT="${RESULT_ROOT}/validation"
FINAL_ROOT="${RESULT_ROOT}/final"
mkdir -p "${CHECKPOINT_ROOT}" "${RESULT_ROOT}" "${VALIDATION_ROOT}" "${FINAL_ROOT}"

echo "[INFO] current dir: $(pwd)"
echo "[INFO] python: $(which python)"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] RUN_SANITY=${RUN_SANITY}"
nvidia-smi || true

python scripts/describe_classicnet_escort_model.py \
  --config "${CONFIG}" \
  --variant main6_localglobal_lr100 \
  --output data/reports/dive_classicnet_escort_model_architecture.json

echo "[1/8] check DIVE feature caches"
python scripts/check_evm_bert_chunk_features.py \
  --feature_dir data/features/continued_dive_evm_bert_dive_random_stride256_max64 \
  --data_dir data/processed/DIVE_random_split \
  --expected_max_chunks 64 \
  --expected_feature_dim 768 \
  --expected_num_labels 8 \
  --output data/reports/dive_classicnet_escort_base_features.txt

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
  --output data/reports/dive_classicnet_escort_semantic_features.txt

train_variant() {
  local variant="$1"
  local transfer_stage="${2:-none}"
  local checkpoint_dir="${CHECKPOINT_ROOT}/${variant}"
  local result_dir="${RESULT_ROOT}/members/${variant}"
  local checkpoint="${checkpoint_dir}/best_selection.pt"
  if [[ -f "${checkpoint}" ]]; then
    echo "[SKIP] ${variant}: ${checkpoint}"
  else
    mkdir -p "${checkpoint_dir}" "${result_dir}"
    local args=(
      --config "${CONFIG}"
      --variant "${variant}"
      --override "checkpoint_dir=${checkpoint_dir}"
      --override "result_dir=${result_dir}"
      --override "experiment_name=dive_classicnet_escort_${variant}"
    )
    if [[ "${RUN_SANITY}" == "1" ]]; then
      args+=(
        --override "epochs=2"
        --override "early_stopping_patience=2"
        --override "debug_num_train_samples=512"
        --override "debug_num_valid_samples=512"
        --override "stable_calibration_bootstrap=50"
      )
    fi
    if [[ "${transfer_stage}" == "front" ]]; then
      args+=(--override "transfer_init_checkpoint=${CHECKPOINT_ROOT}/selected_main6/best_selection.pt")
    elif [[ "${transfer_stage}" == "bad" ]]; then
      args+=(--override "transfer_init_checkpoint=${CHECKPOINT_ROOT}/selected_front/best_selection.pt")
    fi
    if [[ "${transfer_stage}" != "none" && -n "${SELECTED_ENCODER_TYPE:-}" ]]; then
      args+=(--override "chunk_context_encoder_type=${SELECTED_ENCODER_TYPE}")
    fi
    echo "[TRAIN] ${variant}"
    python src/train_chunk_mil.py "${args[@]}"
  fi
  if [[ -f "${result_dir}/epoch_history.json" ]]; then
    mapfile -t curve_labels < <(python - "${CONFIG}" "${variant}" <<'PY'
import sys
sys.path.insert(0, "src")
from train_chunk_mil import load_config
for label in load_config(sys.argv[1], sys.argv[2])["label_names"]:
    print(label)
PY
)
    python scripts/plot_dive_epoch_f1_curves.py \
      --history "${result_dir}/epoch_history.json" \
      --output_dir "${result_dir}/curves" \
      --prefix "${variant}" \
      --labels "${curve_labels[@]}"
  fi
}

eval_valid() {
  local variant="$1"
  local checkpoint="${CHECKPOINT_ROOT}/${variant}/best_selection.pt"
  local eval_dir="${VALIDATION_ROOT}/${variant}"
  local args=(
    --config "${CONFIG}"
    --variant "${variant}"
    --checkpoint "${checkpoint}"
    --split valid
    --threshold_search global_and_per_label
    --result_dir_override "${eval_dir}"
  )
  if [[ "${RUN_SANITY}" == "1" ]]; then
    args+=(
      --override "debug_num_valid_samples=512"
      --override "stable_calibration_bootstrap=50"
    )
  fi
  if [[ "${variant}" != main6_* && -n "${SELECTED_ENCODER_TYPE:-}" ]]; then
    args+=(--override "chunk_context_encoder_type=${SELECTED_ENCODER_TYPE}")
  fi
  python src/evaluate_chunk_mil.py "${args[@]}"
}

eval_test() {
  local variant="$1"
  local checkpoint="$2"
  local eval_dir="$3"
  local valid_dir="$4"
  local args=(
    --config "${CONFIG}" \
    --variant "${variant}" \
    --checkpoint "${checkpoint}" \
    --split test \
    --threshold_file "${valid_dir}/per_label_thresholds_valid.json" \
    --global_threshold_file "${valid_dir}/valid_global_threshold_scan.json" \
    --result_dir_override "${eval_dir}" \
    --save_predictions
  )
  if [[ -n "${SELECTED_ENCODER_TYPE:-}" ]]; then
    args+=(--override "chunk_context_encoder_type=${SELECTED_ENCODER_TYPE}")
  fi
  python src/evaluate_chunk_mil.py "${args[@]}"
}

echo "[2/8] train Main-6 architecture and learning-rate variants"
for variant in ${MAIN6_VARIANTS}; do
  train_variant "${variant}"
  eval_valid "${variant}"
done

python scripts/select_classicnet_escort_checkpoint.py \
  --config "${CONFIG}" \
  --variants ${TRANSFER_MAIN6_VARIANTS} \
  --selection checkpoint \
  --checkpoint_root "${CHECKPOINT_ROOT}" \
  --output_dir "${CHECKPOINT_ROOT}/selected_main6" \
  --report_prefix "${RESULT_ROOT}/main6_selection"

SELECTED_ENCODER_TYPE="$(python - "${CHECKPOINT_ROOT}/selected_main6/best_selection.pt" <<'PY'
import sys
import torch
state = torch.load(sys.argv[1], map_location="cpu")
print(state["config"].get("chunk_context_encoder_type", "transformer"))
PY
)"
echo "[INFO] selected Main-6 encoder: ${SELECTED_ENCODER_TYPE}"

SELECTED_MAIN6_VALID_DIR="${VALIDATION_ROOT}/selected_main6"
python src/evaluate_chunk_mil.py \
  --config "${CONFIG}" \
  --variant main6_localglobal_lr100 \
  --checkpoint "${CHECKPOINT_ROOT}/selected_main6/best_selection.pt" \
  --split valid \
  --threshold_search global_and_per_label \
  --override "chunk_context_encoder_type=${SELECTED_ENCODER_TYPE}" \
  --result_dir_override "${SELECTED_MAIN6_VALID_DIR}"

eval_test \
  main6_localglobal_lr100 \
  "${CHECKPOINT_ROOT}/selected_main6/best_selection.pt" \
  "${FINAL_ROOT}/main6_selected" \
  "${SELECTED_MAIN6_VALID_DIR}"

echo "[3/8] train and validate Front Running transfer variants"
for variant in ${FRONT_VARIANTS}; do
  train_variant "${variant}" front
  eval_valid "${variant}"
done

python scripts/select_classicnet_escort_checkpoint.py \
  --config "${CONFIG}" \
  --variants ${FRONT_VARIANTS} \
  --selection valid_label \
  --label_name "Front Running" \
  --validation_root "${VALIDATION_ROOT}" \
  --checkpoint_root "${CHECKPOINT_ROOT}" \
  --output_dir "${CHECKPOINT_ROOT}/selected_front" \
  --report_prefix "${RESULT_ROOT}/front_selection"

echo "[4/8] audit Main-6 -> Front invariance"
python scripts/audit_classicnet_escort_transfer.py \
  --source_checkpoint "${CHECKPOINT_ROOT}/selected_main6/best_selection.pt" \
  --target_config "${CONFIG}" \
  --target_variant front_r1_lr100 \
  --target_override "chunk_context_encoder_type=${SELECTED_ENCODER_TYPE}" \
  --target_checkpoint "${CHECKPOINT_ROOT}/selected_front/best_selection.pt" \
  --max_samples 128 \
  --tolerance 0.00001 \
  --output "${RESULT_ROOT}/front_transfer_invariance.json"

echo "[5/8] train and validate Bad Randomness transfer variants"
for variant in ${BAD_VARIANTS}; do
  train_variant "${variant}" bad
  eval_valid "${variant}"
done

python scripts/select_classicnet_escort_checkpoint.py \
  --config "${CONFIG}" \
  --variants ${BAD_VARIANTS} \
  --selection valid_label \
  --label_name "Bad Randomness" \
  --validation_root "${VALIDATION_ROOT}" \
  --checkpoint_root "${CHECKPOINT_ROOT}" \
  --output_dir "${CHECKPOINT_ROOT}/selected_bad" \
  --report_prefix "${RESULT_ROOT}/bad_selection"

echo "[6/8] audit Front -> Bad invariance"
python scripts/audit_classicnet_escort_transfer.py \
  --source_checkpoint "${CHECKPOINT_ROOT}/selected_front/best_selection.pt" \
  --target_config "${CONFIG}" \
  --target_variant bad_r1_lr100 \
  --target_override "chunk_context_encoder_type=${SELECTED_ENCODER_TYPE}" \
  --target_checkpoint "${CHECKPOINT_ROOT}/selected_bad/best_selection.pt" \
  --max_samples 128 \
  --tolerance 0.00001 \
  --output "${RESULT_ROOT}/bad_transfer_invariance.json"

echo "[7/8] train joint-8 control and run final evaluations"
train_variant "${JOINT_VARIANT}" joint
eval_valid "${JOINT_VARIANT}"

SELECTED_BAD_VALID_DIR="${VALIDATION_ROOT}/selected_bad"
python src/evaluate_chunk_mil.py \
  --config "${CONFIG}" \
  --variant bad_r1_lr100 \
  --checkpoint "${CHECKPOINT_ROOT}/selected_bad/best_selection.pt" \
  --split valid \
  --threshold_search global_and_per_label \
  --override "chunk_context_encoder_type=${SELECTED_ENCODER_TYPE}" \
  --result_dir_override "${SELECTED_BAD_VALID_DIR}"

eval_test \
  bad_r1_lr100 \
  "${CHECKPOINT_ROOT}/selected_bad/best_selection.pt" \
  "${FINAL_ROOT}/incremental_final" \
  "${SELECTED_BAD_VALID_DIR}"

eval_test \
  "${JOINT_VARIANT}" \
  "${CHECKPOINT_ROOT}/${JOINT_VARIANT}/best_selection.pt" \
  "${FINAL_ROOT}/joint8_bs128_control" \
  "${VALIDATION_ROOT}/${JOINT_VARIANT}"

echo "[8/8] summarize"
python scripts/summarize_dive_classicnet_escort.py \
  --final_dir "${FINAL_ROOT}/incremental_final" \
  --joint_dir "${FINAL_ROOT}/joint8_bs128_control" \
  --output_prefix "${RESULT_ROOT}/summary"

cp "${RESULT_ROOT}/summary.txt" data/reports/dive_classicnet_escort_summary.txt
echo "[OK] DIVE ClassicNet-ESCORT pipeline finished"
