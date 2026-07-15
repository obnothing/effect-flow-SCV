#!/usr/bin/env bash
#SBATCH --job-name=dive_major_st
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=120:00:00
#SBATCH --output=logs/slurm_dive_major_label_v2_single_task_%j.out
#SBATCH --error=logs/slurm_dive_major_label_v2_single_task_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs data/reports results/dive_major_label_v2_single_task

set +u
source ~/.bashrc || true
conda activate correlascan_a40
set -u

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

RUN_SANITY="${RUN_SANITY:-0}"
VARIANTS="${VARIANTS:-single_task_recognition_only single_task_recognition_halfscale}"
CHECKPOINTS="${CHECKPOINTS:-best_micro_f1.pt best_macro_f1.pt}"

echo "[INFO] current dir: $(pwd)"
echo "[INFO] python: $(which python)"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] RUN_SANITY=${RUN_SANITY}"
echo "[INFO] VARIANTS=${VARIANTS}"
echo "[INFO] CHECKPOINTS=${CHECKPOINTS}"
nvidia-smi || true

echo "[DIVE 1/5] check strong base chunk cache"
python scripts/check_evm_bert_chunk_features.py \
  --feature_dir data/features/continued_dive_evm_bert_dive_random_stride256_max64 \
  --data_dir data/processed/DIVE_random_split \
  --expected_max_chunks 64 \
  --expected_feature_dim 768 \
  --expected_num_labels 8 \
  --output data/reports/dive_major_label_v2_single_task_base_features.txt

echo "[DIVE 2/5] check effect-flow semantic cache"
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
  --output data/reports/dive_major_label_v2_single_task_semantic_features.txt

GEN_ARGS=(
  --variants ${VARIANTS}
  --output_config_dir configs/generated/dive_major_label_v2_single_task
  --checkpoint_root checkpoints/dive_major_label_v2_single_task
  --result_root results/dive_major_label_v2_single_task/members
  --num_workers 4
)
if [[ "${RUN_SANITY}" == "1" ]]; then
  GEN_ARGS=(
    --variants ${VARIANTS}
    --output_config_dir configs/generated/dive_major_label_v2_single_task_sanity
    --checkpoint_root checkpoints/dive_major_label_v2_single_task_sanity
    --result_root results/dive_major_label_v2_single_task_sanity/members
    --num_workers 4
    --debug_num_train_samples 512
    --debug_num_valid_samples 512
    --epochs 2
    --early_stopping_patience 2
  )
fi

echo "[DIVE 3/5] generate single-task configs"
python scripts/generate_dive_major_label_v2_single_task_configs.py "${GEN_ARGS[@]}"

if [[ "${RUN_SANITY}" == "1" ]]; then
  MANIFEST="configs/generated/dive_major_label_v2_single_task_sanity/manifest.yaml"
  EVAL_ROOT="results/dive_major_label_v2_single_task_sanity"
else
  MANIFEST="configs/generated/dive_major_label_v2_single_task/manifest.yaml"
  EVAL_ROOT="results/dive_major_label_v2_single_task"
fi
mapfile -t CONFIGS < <(python - "${MANIFEST}" <<'PY'
import sys
from pathlib import Path

import yaml

manifest = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
for item in manifest["variants"]:
    print(item["config"])
PY
)
if [[ "${#CONFIGS[@]}" -eq 0 ]]; then
  echo "[ERROR] no configs generated" >&2
  exit 1
fi
printf '[INFO] selected configs:\n'
printf '  %s\n' "${CONFIGS[@]}"

python - "${CONFIGS[@]}" <<'PY'
import sys
from pathlib import Path

import yaml

for config_path in sys.argv[1:]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if config.get("model_type") != "evef_mvd_v2_side_evidence_mil":
        raise SystemExit(f"{config_path} must use evef_mvd_v2_side_evidence_mil")
    if config.get("task_mode") != "recognition_only":
        raise SystemExit(f"{config_path} must use task_mode=recognition_only")
    if config.get("detection_head_enabled"):
        raise SystemExit(f"{config_path} must disable detection_head_enabled")
    if float(config.get("detection_loss_weight", 1.0)) != 0.0:
        raise SystemExit(f"{config_path} must set detection_loss_weight=0.0")
    if not config.get("derived_detection_from_multilabel"):
        raise SystemExit(f"{config_path} must derive detection from multilabel")
    if config.get("major_global_residual_enabled") or config.get("major_multipool_enabled"):
        raise SystemExit(f"{config_path} must not enable major v2 pooling modules")
    print("[OK]", config_path, config["ablation_variant"])
PY

train_if_needed() {
  local config="$1"
  local checkpoint_dir
  checkpoint_dir="$(python -c "import yaml; print(yaml.safe_load(open('${config}', encoding='utf-8'))['checkpoint_dir'])")"
  local retrain="0"
  if [[ -f "${checkpoint_dir}/best_micro_f1.pt" && -f "${checkpoint_dir}/best_macro_f1.pt" ]]; then
    retrain="$(python - "${checkpoint_dir}/best_micro_f1.pt" "${RUN_SANITY}" <<'PY'
import sys

import torch

checkpoint_path = sys.argv[1]
run_sanity = sys.argv[2] == "1"
state = torch.load(checkpoint_path, map_location="cpu")
config = state.get("config", {})
has_debug = (
    config.get("debug_num_train_samples") is not None
    or config.get("debug_num_valid_samples") is not None
)
print("0" if run_sanity or not has_debug else "1")
PY
)"
  fi
  if [[ -f "${checkpoint_dir}/best_micro_f1.pt" && -f "${checkpoint_dir}/best_macro_f1.pt" && "${retrain}" == "0" ]]; then
    echo "[SKIP] checkpoints exist: ${checkpoint_dir}"
  else
    if [[ "${retrain}" == "1" ]]; then
      echo "[RETRAIN] existing checkpoint is a sanity/debug checkpoint: ${checkpoint_dir}"
    fi
    echo "[TRAIN] ${config}"
    python src/train_chunk_mil.py --config "${config}"
  fi
}

eval_checkpoint() {
  local config="$1"
  local checkpoint_name="$2"
  local checkpoint_dir
  local variant
  checkpoint_dir="$(python -c "import yaml; print(yaml.safe_load(open('${config}', encoding='utf-8'))['checkpoint_dir'])")"
  variant="$(python -c "import yaml; print(yaml.safe_load(open('${config}', encoding='utf-8'))['ablation_variant'].replace('major_label_v2_', ''))")"
  local checkpoint="${checkpoint_dir}/${checkpoint_name}"
  local checkpoint_tag="${checkpoint_name%.pt}"
  local eval_dir="${EVAL_ROOT}/${variant}/${checkpoint_tag}"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "[ERROR] missing checkpoint: ${checkpoint}" >&2
    exit 1
  fi
  echo "[EVAL] ${variant} ${checkpoint_name} -> ${eval_dir}"
  python src/evaluate_chunk_mil.py \
    --config "${config}" \
    --checkpoint "${checkpoint}" \
    --split valid \
    --threshold_search global_and_per_label \
    --result_dir_override "${eval_dir}"

  python src/evaluate_chunk_mil.py \
    --config "${config}" \
    --checkpoint "${checkpoint}" \
    --split test \
    --threshold_file "${eval_dir}/per_label_thresholds_valid.json" \
    --global_threshold_file "${eval_dir}/valid_global_threshold_scan.json" \
    --result_dir_override "${eval_dir}" \
    --save_predictions
}

echo "[DIVE 4/5] train configs"
for config in "${CONFIGS[@]}"; do
  train_if_needed "${config}"
done

echo "[DIVE 5/5] evaluate checkpoints and summarize"
for config in "${CONFIGS[@]}"; do
  for checkpoint_name in ${CHECKPOINTS}; do
    eval_checkpoint "${config}" "${checkpoint_name}"
  done
done

python scripts/summarize_dive_major_label_v2_single_task.py \
  --manifest "${MANIFEST}" \
  --result_root "${EVAL_ROOT}" \
  --output_prefix "${EVAL_ROOT}/summary"

if [[ "${RUN_SANITY}" == "1" ]]; then
  cp "${EVAL_ROOT}/summary.txt" \
    data/reports/dive_major_label_v2_single_task_sanity_summary.txt
else
  cp "${EVAL_ROOT}/summary.txt" \
    data/reports/dive_major_label_v2_single_task_summary.txt
fi

echo "[OK] DIVE major-label v2 single-task search finished."
