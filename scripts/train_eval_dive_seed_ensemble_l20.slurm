#!/usr/bin/env bash
#SBATCH --job-name=dive_seed_ens
#SBATCH --partition=gpu-a10
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=120:00:00
#SBATCH --output=logs/slurm_dive_seed_ensemble_%j.out
#SBATCH --error=logs/slurm_dive_seed_ensemble_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs data/reports results/dive_seed_ensemble

set +u
source ~/.bashrc || true
conda activate correlascan_a40
set -u

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

SEEDS="${SEEDS:-42 123 777 2024 3407}"
SANITY_SEEDS="${SANITY_SEEDS:-42 123}"
RUN_SANITY="${RUN_SANITY:-1}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-best_macro_f1.pt}"
BOOTSTRAP_ITERATIONS="${BOOTSTRAP_ITERATIONS:-500}"
SANITY_BOOTSTRAP_ITERATIONS="${SANITY_BOOTSTRAP_ITERATIONS:-50}"

echo "[INFO] current dir: $(pwd)"
echo "[INFO] python: $(which python)"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] SEEDS=${SEEDS}"
echo "[INFO] RUN_SANITY=${RUN_SANITY}"
echo "[INFO] CHECKPOINT_NAME=${CHECKPOINT_NAME}"
nvidia-smi || true

echo "[DIVE 1/5] check strong base chunk cache"
python scripts/check_evm_bert_chunk_features.py \
  --feature_dir data/features/continued_dive_evm_bert_dive_random_stride256_max64 \
  --data_dir data/processed/DIVE_random_split \
  --expected_max_chunks 64 \
  --expected_feature_dim 768 \
  --expected_num_labels 8 \
  --output data/reports/check_dive_seed_ensemble_base_features.txt

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
  --output data/reports/check_dive_seed_ensemble_semantic_features.txt

manifest_configs() {
  local manifest="$1"
  python - "${manifest}" <<'PY'
import sys
from pathlib import Path

import yaml

manifest = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
for item in manifest["configs"]:
    print(item["config"])
PY
}

train_if_needed() {
  local config="$1"
  local checkpoint_dir
  checkpoint_dir="$(python -c "import yaml; print(yaml.safe_load(open('${config}', encoding='utf-8'))['checkpoint_dir'])")"
  local checkpoint="${checkpoint_dir}/${CHECKPOINT_NAME}"
  echo "[TRAIN] ${config}"
  if [[ -f "${checkpoint}" ]]; then
    echo "[SKIP] checkpoint exists: ${checkpoint}"
  else
    python src/train_chunk_mil.py --config "${config}"
  fi
}

run_manifest_training() {
  local manifest="$1"
  mapfile -t configs < <(manifest_configs "${manifest}")
  if [[ "${#configs[@]}" -eq 0 ]]; then
    echo "[ERROR] no configs in manifest ${manifest}" >&2
    exit 1
  fi
  printf '[INFO] selected configs from %s:\n' "${manifest}"
  printf '  %s\n' "${configs[@]}"
  for config in "${configs[@]}"; do
    train_if_needed "${config}"
  done
}

if [[ "${RUN_SANITY}" == "1" ]]; then
  echo "[DIVE 3/5] generate and run sanity seed ensemble"
  python scripts/generate_dive_seed_ensemble_configs.py \
    --output_config_dir configs/generated/dive_seed_ensemble_sanity \
    --checkpoint_root checkpoints/dive_seed_ensemble_sanity \
    --result_root results/dive_seed_ensemble/sanity/members \
    --experiment_prefix sanity_train_dive_side_scale_150 \
    --variant_prefix sanity_side_scale_150_ep50 \
    --ensemble_group dive_seed_ensemble_sanity \
    --seeds ${SANITY_SEEDS} \
    --debug_num_train_samples 512 \
    --debug_num_valid_samples 512 \
    --batch_size 512 \
    --epochs 1 \
    --early_stopping_patience 1 \
    --num_workers 4
  run_manifest_training configs/generated/dive_seed_ensemble_sanity/manifest.yaml
  python scripts/evaluate_dive_seed_ensemble.py \
    --manifest configs/generated/dive_seed_ensemble_sanity/manifest.yaml \
    --checkpoint_name "${CHECKPOINT_NAME}" \
    --result_dir results/dive_seed_ensemble/sanity/ensemble_best_macro \
    --bootstrap_iterations "${SANITY_BOOTSTRAP_ITERATIONS}"
else
  echo "[DIVE 3/5] sanity skipped"
fi

echo "[DIVE 4/5] generate and train full seed ensemble"
python scripts/generate_dive_seed_ensemble_configs.py \
  --output_config_dir configs/generated/dive_seed_ensemble \
  --checkpoint_root checkpoints/dive_seed_ensemble \
  --result_root results/dive_seed_ensemble/members \
  --experiment_prefix train_dive_side_scale_150 \
  --variant_prefix side_scale_150_ep50 \
  --ensemble_group dive_seed_ensemble \
  --seeds ${SEEDS} \
  --num_workers 4

run_manifest_training configs/generated/dive_seed_ensemble/manifest.yaml

CHECKPOINT_TAG="${CHECKPOINT_NAME%.pt}"

echo "[DIVE 5/5] evaluate full seed ensemble"
python scripts/evaluate_dive_seed_ensemble.py \
  --manifest configs/generated/dive_seed_ensemble/manifest.yaml \
  --checkpoint_name "${CHECKPOINT_NAME}" \
  --result_dir "results/dive_seed_ensemble/ensemble_${CHECKPOINT_TAG}" \
  --bootstrap_iterations "${BOOTSTRAP_ITERATIONS}"

echo "[OK] DIVE seed ensemble run finished."
