#!/usr/bin/env bash
#SBATCH --job-name=dive_major_b200
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=168:00:00
#SBATCH --output=logs/slurm_dive_major_label_v2_branch_single_task_ep200_%j.out
#SBATCH --error=logs/slurm_dive_major_label_v2_branch_single_task_ep200_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs data/reports results/dive_major_label_v2_branch_single_task

set +u
source ~/.bashrc || true
conda activate correlascan_a40
set -u

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

RUN_SANITY="${RUN_SANITY:-0}"
CONFIG="${CONFIG:-configs/dive_major_label_v2_branch_mlp_recognition_only_ep200.yaml}"
MANIFEST="${MANIFEST:-configs/dive_major_label_v2_branch_single_task_ep200_manifest.yaml}"
CHECKPOINTS="${CHECKPOINTS:-best_micro_f1.pt best_macro_f1.pt}"
EVAL_ROOT="${EVAL_ROOT:-results/dive_major_label_v2_branch_single_task}"

echo "[INFO] current dir: $(pwd)"
echo "[INFO] python: $(which python)"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] RUN_SANITY=${RUN_SANITY}"
echo "[INFO] CONFIG=${CONFIG}"
echo "[INFO] CHECKPOINTS=${CHECKPOINTS}"
nvidia-smi || true

if [[ "${RUN_SANITY}" == "1" ]]; then
  echo "[ERROR] This static ep200 script is for the formal run. Use RUN_SANITY=0." >&2
  exit 1
fi

echo "[DIVE 1/5] check strong base chunk cache"
python scripts/check_evm_bert_chunk_features.py \
  --feature_dir data/features/continued_dive_evm_bert_dive_random_stride256_max64 \
  --data_dir data/processed/DIVE_random_split \
  --expected_max_chunks 64 \
  --expected_feature_dim 768 \
  --expected_num_labels 8 \
  --output data/reports/dive_major_label_v2_branch_single_task_ep200_base_features.txt

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
  --output data/reports/dive_major_label_v2_branch_single_task_ep200_semantic_features.txt

python - "${CONFIG}" <<'PY'
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
checks = {
    "model_type": config.get("model_type") == "evef_mvd_v2_side_evidence_mil",
    "task_mode": config.get("task_mode") == "recognition_only",
    "detection_head_enabled": not bool(config.get("detection_head_enabled")),
    "detection_loss_weight": float(config.get("detection_loss_weight", 1.0)) == 0.0,
    "recognition_loss_weight": float(config.get("recognition_loss_weight", 0.0)) == 1.0,
    "derived_detection_from_multilabel": bool(config.get("derived_detection_from_multilabel")),
    "recognition_head_type": config.get("recognition_head_type") == "label_branch_mlp",
    "side_evidence_enabled": bool(config.get("side_evidence_enabled")),
    "epochs": int(config.get("epochs", 0)) == 200,
    "early_stopping_patience": int(config.get("early_stopping_patience", 0)) == 100,
    "major_global_residual_enabled": not bool(config.get("major_global_residual_enabled")),
    "major_multipool_enabled": not bool(config.get("major_multipool_enabled")),
    "graph_evidence_enabled": not bool(config.get("graph_evidence_enabled")),
}
failed = [name for name, ok in checks.items() if not ok]
if failed:
    raise SystemExit(f"{config_path} failed config checks: {failed}")
print("[OK]", config_path, config["experiment_name"])
PY

checkpoint_dir="$(python -c "import yaml; print(yaml.safe_load(open('${CONFIG}', encoding='utf-8'))['checkpoint_dir'])")"
result_dir="$(python -c "import yaml; print(yaml.safe_load(open('${CONFIG}', encoding='utf-8'))['result_dir'])")"
variant="$(python -c "import yaml; v=yaml.safe_load(open('${CONFIG}', encoding='utf-8'))['ablation_variant']; print(v.replace('major_label_v2_branch_single_task_', ''))")"

echo "[DIVE 3/5] train static ep200 config"
if [[ -f "${checkpoint_dir}/best_micro_f1.pt" && -f "${checkpoint_dir}/best_macro_f1.pt" ]]; then
  echo "[SKIP] checkpoints exist: ${checkpoint_dir}"
else
  python src/train_chunk_mil.py --config "${CONFIG}"
fi

history_path="${result_dir}/epoch_history.json"
if [[ ! -f "${history_path}" ]]; then
  history_path="${result_dir}/epoch_history.txt"
fi

echo "[DIVE 4/5] plot validation F1 curves"
python scripts/plot_dive_epoch_f1_curves.py \
  --history "${history_path}" \
  --output_dir "${EVAL_ROOT}/${variant}/curves" \
  --prefix "${variant}"

eval_checkpoint() {
  local checkpoint_name="$1"
  local checkpoint="${checkpoint_dir}/${checkpoint_name}"
  local checkpoint_tag="${checkpoint_name%.pt}"
  local eval_dir="${EVAL_ROOT}/${variant}/${checkpoint_tag}"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "[ERROR] missing checkpoint: ${checkpoint}" >&2
    exit 1
  fi
  echo "[EVAL] ${variant} ${checkpoint_name} -> ${eval_dir}"
  python src/evaluate_chunk_mil.py \
    --config "${CONFIG}" \
    --checkpoint "${checkpoint}" \
    --split valid \
    --threshold_search global_and_per_label \
    --result_dir_override "${eval_dir}"

  python src/evaluate_chunk_mil.py \
    --config "${CONFIG}" \
    --checkpoint "${checkpoint}" \
    --split test \
    --threshold_file "${eval_dir}/per_label_thresholds_valid.json" \
    --global_threshold_file "${eval_dir}/valid_global_threshold_scan.json" \
    --result_dir_override "${eval_dir}" \
    --save_predictions
}

echo "[DIVE 5/5] evaluate checkpoints and summarize"
for checkpoint_name in ${CHECKPOINTS}; do
  eval_checkpoint "${checkpoint_name}"
done

python scripts/summarize_dive_major_label_v2_branch_single_task.py \
  --manifest "${MANIFEST}" \
  --result_root "${EVAL_ROOT}" \
  --output_prefix "${EVAL_ROOT}/summary_ep200"

cp "${EVAL_ROOT}/summary_ep200.txt" \
  data/reports/dive_major_label_v2_branch_single_task_ep200_summary.txt

echo "[OK] DIVE major-label v2 branch single-task ep200 run finished."
