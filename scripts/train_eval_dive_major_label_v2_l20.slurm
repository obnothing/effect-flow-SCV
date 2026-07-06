#!/usr/bin/env bash
#SBATCH --job-name=dive_major_v2
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=120:00:00
#SBATCH --output=logs/slurm_dive_major_label_v2_%j.out
#SBATCH --error=logs/slurm_dive_major_label_v2_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs data/reports results/dive_major_label_v2

set +u
source ~/.bashrc || true
conda activate correlascan_a40
set -u

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

RUN_SANITY="${RUN_SANITY:-0}"
VARIANTS="${VARIANTS:-global_residual multipool global_residual_multipool}"
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
  --output data/reports/dive_major_label_v2_base_features.txt

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
  --output data/reports/dive_major_label_v2_semantic_features.txt

GEN_ARGS=(
  --variants ${VARIANTS}
  --output_config_dir configs/generated/dive_major_label_v2
  --checkpoint_root checkpoints/dive_major_label_v2
  --result_root results/dive_major_label_v2/members
  --num_workers 4
)
if [[ "${RUN_SANITY}" == "1" ]]; then
  GEN_ARGS+=(--debug_num_train_samples 512 --debug_num_valid_samples 512 --epochs 2 --early_stopping_patience 2)
fi

echo "[DIVE 3/5] generate major-label v2 configs"
python scripts/generate_dive_major_label_v2_configs.py "${GEN_ARGS[@]}"

MANIFEST="configs/generated/dive_major_label_v2/manifest.yaml"
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
    if config.get("front_running_special_enabled"):
        raise SystemExit(f"{config_path} must keep front_running_special disabled")
    if config.get("front_contrastive_loss_enabled"):
        raise SystemExit(f"{config_path} must keep front_contrastive disabled")
    if config.get("rare_negative_subsampling_enabled"):
        raise SystemExit(f"{config_path} must keep rare negative subsampling disabled")
    if not (config.get("major_global_residual_enabled") or config.get("major_multipool_enabled")):
        raise SystemExit(f"{config_path} must enable at least one major-label module")
    print("[OK]", config_path, config["ablation_variant"])
PY

train_if_needed() {
  local config="$1"
  local checkpoint_dir
  checkpoint_dir="$(python -c "import yaml; print(yaml.safe_load(open('${config}', encoding='utf-8'))['checkpoint_dir'])")"
  if [[ -f "${checkpoint_dir}/best_micro_f1.pt" && -f "${checkpoint_dir}/best_macro_f1.pt" ]]; then
    echo "[SKIP] checkpoints exist: ${checkpoint_dir}"
  else
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
  local eval_dir="results/dive_major_label_v2/${variant}/${checkpoint_tag}"
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

python scripts/summarize_dive_major_label_v2_search.py \
  --manifest "${MANIFEST}" \
  --result_root results/dive_major_label_v2 \
  --output_prefix results/dive_major_label_v2/summary

cp results/dive_major_label_v2/summary.txt data/reports/dive_major_label_v2_summary.txt

echo "[OK] DIVE major-label v2 search finished."
