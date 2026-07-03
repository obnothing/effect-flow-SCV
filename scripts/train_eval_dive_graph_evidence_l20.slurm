#!/usr/bin/env bash
#SBATCH --job-name=dive_graph_ev
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=120:00:00
#SBATCH --output=logs/slurm_dive_graph_evidence_%j.out
#SBATCH --error=logs/slurm_dive_graph_evidence_%j.err

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
SCALES="${SCALES:-0.5 1.0 1.5}"
FORCE_GRAPH="${FORCE_GRAPH:-0}"

GRAPH_EVIDENCE_DIR="data/features/graph_evidence/dive_random_stride256_max64"
FEATURE_DIR="data/features/continued_dive_evm_bert_dive_random_stride256_max64"
SEMANTIC_DIR="data/features/effect_flow_semantics/dive_random_stride256_max64"
DATA_DIR="data/processed/DIVE_random_split"

if [[ "${RUN_SANITY}" == "1" ]]; then
  OUTPUT_CONFIG_DIR="configs/generated/dive_graph_evidence_search_sanity"
  CHECKPOINT_ROOT="checkpoints/dive_graph_evidence_search_sanity"
  RESULT_ROOT="results/dive_graph_evidence_search_sanity"
  EPOCHS=1
  EXTRA_GENERATOR_ARGS=(--debug_num_train_samples 512 --debug_num_valid_samples 512 --early_stopping_patience 1)
else
  OUTPUT_CONFIG_DIR="configs/generated/dive_graph_evidence_search"
  CHECKPOINT_ROOT="checkpoints/dive_graph_evidence_search"
  RESULT_ROOT="results/dive_graph_evidence_search"
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
echo "[INFO] SCALES=${SCALES}"
echo "[INFO] GRAPH_EVIDENCE_DIR=${GRAPH_EVIDENCE_DIR}"
nvidia-smi || true

echo "[DIVE 1/6] check strong base chunk cache"
python scripts/check_evm_bert_chunk_features.py \
  --feature_dir "${FEATURE_DIR}" \
  --data_dir "${DATA_DIR}" \
  --expected_max_chunks 64 \
  --expected_feature_dim 768 \
  --expected_num_labels 8 \
  --output data/reports/check_dive_graph_evidence_base_features.txt

echo "[DIVE 2/6] check effect-flow semantic cache"
python scripts/check_effect_flow_semantic_features.py \
  --semantic_dir "${SEMANTIC_DIR}" \
  --feature_dir "${FEATURE_DIR}" \
  --data_dir "${DATA_DIR}" \
  --expected_max_chunks 64 \
  --expected_efpp_dim 22 \
  --expected_etp_dim 16 \
  --expected_relation_dim 6 \
  --expected_global_template_dim 17 \
  --expected_num_labels 8 \
  --output data/reports/check_dive_graph_evidence_semantic_features.txt

echo "[DIVE 3/6] build contract-level graph evidence cache"
GRAPH_ARGS=()
if [[ "${FORCE_GRAPH}" == "1" ]]; then
  GRAPH_ARGS+=(--force)
fi
python scripts/build_dive_graph_evidence.py \
  --data_dir "${DATA_DIR}" \
  --feature_dir "${FEATURE_DIR}" \
  --output_dir "${GRAPH_EVIDENCE_DIR}" \
  --chunk_size 512 \
  --chunk_stride 256 \
  --max_chunks 64 \
  --num_labels 8 \
  --num_workers 4 \
  "${GRAPH_ARGS[@]}"

echo "[DIVE 4/6] generate graph evidence configs"
python scripts/generate_dive_graph_evidence_configs.py \
  --output_config_dir "${OUTPUT_CONFIG_DIR}" \
  --checkpoint_root "${CHECKPOINT_ROOT}" \
  --result_root "${RESULT_ROOT}" \
  --graph_evidence_dir "${GRAPH_EVIDENCE_DIR}" \
  --scales ${SCALES} \
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

echo "[DIVE 5/6] validate configs and train"
python - "${CONFIGS[@]}" <<'PY'
import sys
from pathlib import Path

import yaml

for config_path in sys.argv[1:]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if config.get("model_type") != "evef_mvd_v2_side_evidence_mil":
        raise SystemExit(f"{config_path} must use evef_mvd_v2_side_evidence_mil")
    if not config.get("graph_evidence_enabled"):
        raise SystemExit(f"{config_path} must enable graph_evidence")
    if not config.get("graph_evidence_dir"):
        raise SystemExit(f"{config_path} missing graph_evidence_dir")
    if int(config.get("graph_evidence_dim", 0)) != 12:
        raise SystemExit(f"{config_path} graph_evidence_dim must be 12")
    if config.get("front_running_special_enabled"):
        raise SystemExit(f"{config_path} must keep front_running_special disabled")
    if config.get("front_hard_negative_loss_enabled"):
        raise SystemExit(f"{config_path} must keep front_hard_negative disabled")
    if config.get("front_contrastive_loss_enabled"):
        raise SystemExit(f"{config_path} must keep front_contrastive disabled")
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
        "graph_scale=",
        config.get("graph_evidence_scale"),
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

echo "[DIVE 6/6] evaluate selected checkpoints and summarize"
for config in "${CONFIGS[@]}"; do
  for checkpoint_name in ${CHECKPOINTS}; do
    eval_checkpoint "${config}" "${checkpoint_name}"
  done
done

python scripts/summarize_dive_graph_evidence_search.py \
  --manifest "${MANIFEST}" \
  --output_prefix "${RESULT_ROOT}/summary"

echo "[OK] DIVE graph evidence search finished."
