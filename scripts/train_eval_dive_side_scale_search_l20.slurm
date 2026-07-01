#!/usr/bin/env bash
#SBATCH --job-name=dive_side_search
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=96:00:00
#SBATCH --output=logs/slurm_dive_side_search_%j.out
#SBATCH --error=logs/slurm_dive_side_search_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs data/reports results/dive_side_scale_search

set +u
source ~/.bashrc || true
conda activate correlascan_a40
set -u

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

MATRIX="configs/evef_mvd_v2_dive_side_scale_search.yaml"
MANIFEST="configs/generated/dive_side_scale_search/manifest.yaml"
VARIANTS="${VARIANTS:-}"

echo "[INFO] current dir: $(pwd)"
echo "[INFO] python: $(which python)"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] VARIANTS=${VARIANTS:-ALL}"
nvidia-smi || true

echo "[DIVE 1/5] generate 50-epoch side-scale search configs"
python scripts/generate_evef_mvd_ablation_configs.py \
  --matrix "${MATRIX}"

echo "[DIVE 2/5] check strong base chunk cache"
python scripts/check_evm_bert_chunk_features.py \
  --feature_dir data/features/continued_dive_evm_bert_dive_random_stride256_max64 \
  --data_dir data/processed/DIVE_random_split \
  --expected_max_chunks 64 \
  --expected_feature_dim 768 \
  --expected_num_labels 8 \
  --output data/reports/check_dive_side_scale_search_base_features.txt

echo "[DIVE 3/5] check effect-flow semantic cache"
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
  --output data/reports/check_dive_side_scale_search_semantic_features.txt

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

echo "[INFO] selected configs:"
printf '  %s\n' "${CONFIGS[@]}"

echo "[DIVE 4/5] validate configs use simple side-scale scoring"
python - "${CONFIGS[@]}" <<'PY'
import sys
from pathlib import Path
import yaml

for config_path in sys.argv[1:]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if config.get("behavior_weight_path"):
        raise SystemExit(f"{config_path} unexpectedly uses behavior_weight_path")
    if config.get("epochs") != 50:
        raise SystemExit(f"{config_path} epochs must be 50")
    if int(config.get("early_stopping_patience", 0)) < 10:
        raise SystemExit(f"{config_path} early_stopping_patience must be >= 10")
    print(
        "[OK]",
        config_path,
        "scale=",
        config.get("coefficient_scale"),
        "risk=",
        config.get("risk_evidence_weight"),
        "beta=",
        config.get("beta_reliable_init"),
    )
PY

run_config() {
  local config="$1"
  local checkpoint_dir
  local result_dir
  checkpoint_dir="$(python -c "import yaml; print(yaml.safe_load(open('${config}', encoding='utf-8'))['checkpoint_dir'])")"
  result_dir="$(python -c "import yaml; print(yaml.safe_load(open('${config}', encoding='utf-8'))['result_dir'])")"
  local checkpoint="${checkpoint_dir}/best_macro_f1.pt"

  echo "[RUN] ${config}"
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
    --output_prefix "${result_dir}/behavior_contribution_summary"
}

for config in "${CONFIGS[@]}"; do
  run_config "${config}"
done

echo "[DIVE 5/5] summarize side-scale search"
python scripts/summarize_dive_side_scale_search.py \
  --manifest "${MANIFEST}" \
  --output_prefix results/dive_side_scale_search/summary

echo "[OK] DIVE side-scale search finished."
