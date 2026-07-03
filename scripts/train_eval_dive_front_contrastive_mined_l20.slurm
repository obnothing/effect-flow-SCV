#!/usr/bin/env bash
#SBATCH --job-name=dive_front_mined
#SBATCH --partition=gpu-a10
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=120:00:00
#SBATCH --output=logs/slurm_dive_front_contrastive_mined_%j.out
#SBATCH --error=logs/slurm_dive_front_contrastive_mined_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs data/reports

set +u
source ~/.bashrc || true
conda activate correlascan_a40
set -u

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

RUN_SANITY="${RUN_SANITY:-0}"
VARIANTS="${VARIANTS:-}"
CHECKPOINTS="${CHECKPOINTS:-best_micro_f1.pt best_macro_f1.pt}"
LAMBDAS="${LAMBDAS:-0.005 0.01 0.02}"

if [[ "${RUN_SANITY}" == "1" ]]; then
  OUTPUT_CONFIG_DIR="configs/generated/dive_front_contrastive_mined_search_sanity"
  CHECKPOINT_ROOT="checkpoints/dive_front_contrastive_mined_search_sanity"
  RESULT_ROOT="results/dive_front_contrastive_mined_search_sanity"
  EPOCHS=1
  EXTRA_GENERATOR_ARGS=(--debug_num_train_samples 512 --debug_num_valid_samples 512 --early_stopping_patience 1)
else
  OUTPUT_CONFIG_DIR="configs/generated/dive_front_contrastive_mined_search"
  CHECKPOINT_ROOT="checkpoints/dive_front_contrastive_mined_search"
  RESULT_ROOT="results/dive_front_contrastive_mined_search"
  EPOCHS=50
  EXTRA_GENERATOR_ARGS=()
fi
MANIFEST="${OUTPUT_CONFIG_DIR}/manifest.yaml"

mkdir -p "${RESULT_ROOT}"

echo "[INFO] current dir: $(pwd)"
echo "[INFO] python: $(which python)"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] RUN_SANITY=${RUN_SANITY}"
echo "[INFO] VARIANTS=${VARIANTS:-ALL}"
echo "[INFO] CHECKPOINTS=${CHECKPOINTS}"
echo "[INFO] LAMBDAS=${LAMBDAS}"
echo "[INFO] RESULT_ROOT=${RESULT_ROOT}"
nvidia-smi || true

echo "[DIVE 1/5] check strong base chunk cache"
python scripts/check_evm_bert_chunk_features.py \
  --feature_dir data/features/continued_dive_evm_bert_dive_random_stride256_max64 \
  --data_dir data/processed/DIVE_random_split \
  --expected_max_chunks 64 \
  --expected_feature_dim 768 \
  --expected_num_labels 8 \
  --output data/reports/check_dive_front_contrastive_mined_base_features.txt

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
  --output data/reports/check_dive_front_contrastive_mined_semantic_features.txt

echo "[DIVE 3/5] generate mined Front contrastive configs"
python scripts/generate_dive_front_contrastive_mined_configs.py \
  --output_config_dir "${OUTPUT_CONFIG_DIR}" \
  --checkpoint_root "${CHECKPOINT_ROOT}" \
  --result_root "${RESULT_ROOT}" \
  --lambdas ${LAMBDAS} \
  --epochs "${EPOCHS}" \
  --batch_size 1024 \
  --num_workers 4 \
  "${EXTRA_GENERATOR_ARGS[@]}"

mapfile -t CONFIGS < <(python - "${MANIFEST}" "${VARIANTS}" <<'PY'
import sys
from pathlib import Path

import yaml

manifest = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
variant_filter = set(sys.argv[2].split()) if sys.argv[2].strip() else None
for item in manifest["configs"]:
    if variant_filter and item["variant"] not in variant_filter:
        continue
    print(item["config"])
PY
)

if [[ "${#CONFIGS[@]}" -eq 0 ]]; then
  echo "[ERROR] no configs selected. Check VARIANTS=${VARIANTS}" >&2
  exit 1
fi

echo "[INFO] selected configs:"
printf '  %s\n' "${CONFIGS[@]}"

echo "[DIVE 4/5] validate configs and train"
python - "${CONFIGS[@]}" <<'PY'
import sys
from pathlib import Path

import yaml

expected_confounders = ["Access Control", "Reentrancy", "Time manipulation"]
for config_path in sys.argv[1:]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if config.get("model_type") != "evef_mvd_v2_side_evidence_mil":
        raise SystemExit(f"{config_path} must use evef_mvd_v2_side_evidence_mil")
    if not config.get("front_contrastive_loss_enabled"):
        raise SystemExit(f"{config_path} must enable front_contrastive_loss")
    if config.get("front_running_special_enabled"):
        raise SystemExit(f"{config_path} must keep front_running_special disabled")
    if config.get("front_hard_negative_loss_enabled"):
        raise SystemExit(f"{config_path} must keep front_hard_negative disabled")
    if config.get("front_contrastive_confounder_labels") != expected_confounders:
        raise SystemExit(f"{config_path} confounder labels mismatch")
    if config.get("front_contrastive_negative_mining") != "front_prob_topk":
        raise SystemExit(f"{config_path} must use front_prob_topk mining")
    if float(config.get("front_contrastive_hard_negative_ratio", 0.0)) != 2.0:
        raise SystemExit(f"{config_path} hard negative ratio must be 2.0")
    if int(config.get("front_contrastive_hard_negative_min_k", 0)) != 16:
        raise SystemExit(f"{config_path} hard negative min_k must be 16")
    if int(config.get("front_contrastive_hard_negative_max_k", 0)) != 128:
        raise SystemExit(f"{config_path} hard negative max_k must be 128")
    if int(config.get("batch_size", 0)) != 1024:
        raise SystemExit(f"{config_path} batch_size must be 1024")
    if int(config.get("max_chunks", 0)) != 64:
        raise SystemExit(f"{config_path} max_chunks must be 64")
    if int(config.get("num_workers", 0)) != 4:
        raise SystemExit(f"{config_path} num_workers must be 4")
    if float(config.get("coefficient_scale", 0.0)) != 1.5:
        raise SystemExit(f"{config_path} coefficient_scale must be 1.5")
    print(
        "[OK]",
        config_path,
        "lambda=",
        config.get("front_contrastive_lambda"),
        "epochs=",
        config.get("epochs"),
    )
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
  variant="$(python -c "import yaml; print(yaml.safe_load(open('${config}', encoding='utf-8'))['ablation_variant'])")"
  local checkpoint="${checkpoint_dir}/${checkpoint_name}"
  local checkpoint_tag="${checkpoint_name%.pt}"
  local eval_dir="${RESULT_ROOT}/${variant}/${checkpoint_tag}"
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

for config in "${CONFIGS[@]}"; do
  train_if_needed "${config}"
done

echo "[DIVE 5/5] evaluate selected checkpoints and summarize"
for config in "${CONFIGS[@]}"; do
  for checkpoint_name in ${CHECKPOINTS}; do
    eval_checkpoint "${config}" "${checkpoint_name}"
  done
done

python scripts/summarize_dive_front_contrastive_search.py \
  --manifest "${MANIFEST}" \
  --output_prefix "${RESULT_ROOT}/summary"

echo "[OK] DIVE mined Front conditional contrastive search finished."
