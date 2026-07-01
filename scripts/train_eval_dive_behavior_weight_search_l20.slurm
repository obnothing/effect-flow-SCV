#!/usr/bin/env bash
#SBATCH --job-name=dive_beh_search
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=72:00:00
#SBATCH --output=logs/slurm_dive_beh_search_%j.out
#SBATCH --error=logs/slurm_dive_beh_search_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs data/reports results/dive_behavior_weight_search

set +u
source ~/.bashrc || true
conda activate correlascan_a40
set -u

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

CONFIGS=(
  "configs/generated/dive_behavior_weight_search/train_dive_behavior_v2_relation_split.yaml"
  "configs/generated/dive_behavior_weight_search/train_dive_behavior_v2_front_conservative.yaml"
  "configs/generated/dive_behavior_weight_search/train_dive_behavior_v2_access_counter.yaml"
)

echo "[INFO] current dir: $(pwd)"
echo "[INFO] python: $(which python)"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
nvidia-smi || true

echo "[DIVE 1/5] check strong base chunk cache"
python scripts/check_evm_bert_chunk_features.py \
  --feature_dir data/features/continued_dive_evm_bert_dive_random_stride256_max64 \
  --data_dir data/processed/DIVE_random_split \
  --expected_max_chunks 64 \
  --expected_feature_dim 768 \
  --expected_num_labels 8 \
  --output data/reports/check_dive_behavior_weight_search_base_features.txt

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
  --output data/reports/check_dive_behavior_weight_search_semantic_features.txt

echo "[DIVE 3/5] validate behavior weight configs"
python - "${CONFIGS[@]}" <<'PY'
import sys
from pathlib import Path
import yaml

PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from evm_chunk_mil_model import EVEFMVDV2SideEvidenceMIL

for config_path in sys.argv[1:]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    model = EVEFMVDV2SideEvidenceMIL(config)
    print(
        "[OK]",
        config_path,
        "weighted=",
        model.use_weighted_behavior_scoring,
        "risk_rel=",
        tuple(model.behavior_risk_relation_weights.shape),
        "protective_rel=",
        tuple(model.behavior_protective_relation_weights.shape),
    )
PY

run_variant() {
  local config="$1"
  local result_dir
  local checkpoint_dir
  result_dir="$(python -c "import yaml; print(yaml.safe_load(open('${config}', encoding='utf-8'))['result_dir'])")"
  checkpoint_dir="$(python -c "import yaml; print(yaml.safe_load(open('${config}', encoding='utf-8'))['checkpoint_dir'])")"
  local checkpoint="${checkpoint_dir}/best_macro_f1.pt"

  echo "[DIVE 4/5] train/eval variant: ${config}"
  if [[ -f "${checkpoint}" ]]; then
    echo "[SKIP] checkpoint exists: ${checkpoint}"
  else
    python src/train_chunk_mil.py --config "${config}"
  fi

  python src/evaluate_chunk_mil.py \
    --config "${config}" \
    --checkpoint "${checkpoint}" \
    --split valid \
    --threshold_search global_and_per_label

  python src/evaluate_chunk_mil.py \
    --config "${config}" \
    --checkpoint "${checkpoint}" \
    --split test \
    --threshold_file "${result_dir}/per_label_thresholds_valid.json" \
    --global_threshold_file "${result_dir}/valid_global_threshold_scan.json" \
    --save_predictions

  python scripts/summarize_behavior_contributions.py \
    --config "${config}" \
    --checkpoint "${checkpoint}" \
    --split test \
    --threshold_file "${result_dir}/per_label_thresholds_valid.json" \
    --global_threshold_file "${result_dir}/valid_global_threshold_scan.json" \
    --output_prefix "${result_dir}/behavior_contribution_summary"
}

for config in "${CONFIGS[@]}"; do
  run_variant "${config}"
done

echo "[DIVE 5/5] summarize weight search"
python scripts/summarize_dive_behavior_weight_search.py \
  --configs "${CONFIGS[@]}" \
  --output_prefix results/dive_behavior_weight_search/summary

echo "[OK] DIVE behavior weight search finished."
