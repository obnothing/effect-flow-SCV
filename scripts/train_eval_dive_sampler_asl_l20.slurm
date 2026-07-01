#!/usr/bin/env bash
#SBATCH --job-name=dive_sampler_asl
#SBATCH --partition=gpu-a10
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=120:00:00
#SBATCH --output=logs/slurm_dive_sampler_asl_%j.out
#SBATCH --error=logs/slurm_dive_sampler_asl_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs data/reports results/dive_sampler_asl_search

set +u
source ~/.bashrc || true
conda activate correlascan_a40
set -u

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

MATRIX="configs/evef_mvd_dive_sampler_asl_search.yaml"
MANIFEST="configs/generated/dive_sampler_asl_search/manifest.yaml"
VARIANTS="${VARIANTS:-}"
RUN_SANITY="${RUN_SANITY:-1}"

echo "[INFO] current dir: $(pwd)"
echo "[INFO] python: $(which python)"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] VARIANTS=${VARIANTS:-ALL}"
echo "[INFO] RUN_SANITY=${RUN_SANITY}"
nvidia-smi || true

echo "[DIVE 1/6] generate sampler/ASL configs"
python scripts/generate_evef_mvd_ablation_configs.py \
  --matrix "${MATRIX}"

echo "[DIVE 2/6] check strong base chunk cache"
python scripts/check_evm_bert_chunk_features.py \
  --feature_dir data/features/continued_dive_evm_bert_dive_random_stride256_max64 \
  --data_dir data/processed/DIVE_random_split \
  --expected_max_chunks 64 \
  --expected_feature_dim 768 \
  --expected_num_labels 8 \
  --output data/reports/check_dive_sampler_asl_base_features.txt

echo "[DIVE 3/6] check effect-flow semantic cache"
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
  --output data/reports/check_dive_sampler_asl_semantic_features.txt

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

echo "[DIVE 4/6] validate sampler/ASL configs"
python - "${CONFIGS[@]}" <<'PY'
import sys
from pathlib import Path
import yaml

for config_path in sys.argv[1:]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    variant = config["ablation_variant"]
    loss_type = config.get("recognition_loss_type", "bce")
    sampler = config.get("train_sampler", "shuffle")
    if config.get("behavior_weight_path"):
        raise SystemExit(f"{config_path} unexpectedly uses behavior_weight_path")
    if int(config.get("epochs", 0)) != 50:
        raise SystemExit(f"{config_path} epochs must be 50")
    if int(config.get("early_stopping_patience", 0)) < 10:
        raise SystemExit(f"{config_path} early_stopping_patience must be >= 10")
    if float(config.get("coefficient_scale", 0.0)) != 1.5:
        raise SystemExit(f"{config_path} coefficient_scale must be 1.5")
    if int(config.get("batch_size", 0)) != 1024:
        raise SystemExit(f"{config_path} batch_size must be 1024")
    if loss_type == "asl" and config.get("use_pos_weight", False):
        raise SystemExit(f"{config_path} ASL must set use_pos_weight=false")
    if variant.endswith("sampler_bce") and (sampler != "label_balanced" or loss_type != "bce"):
        raise SystemExit(f"{config_path} sampler_bce config mismatch")
    if variant.endswith("asl") and "sampler_asl" not in variant and (sampler != "shuffle" or loss_type != "asl"):
        raise SystemExit(f"{config_path} asl config mismatch")
    if variant.endswith("sampler_asl") and (sampler != "label_balanced" or loss_type != "asl"):
        raise SystemExit(f"{config_path} sampler_asl config mismatch")
    print(
        "[OK]",
        config_path,
        "sampler=",
        sampler,
        "loss=",
        loss_type,
        "pos_weight=",
        config.get("use_pos_weight", False),
    )
PY

make_sanity_config() {
  local config="$1"
  python - "${config}" <<'PY'
import sys
from pathlib import Path
import yaml

config_path = Path(sys.argv[1])
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
variant = config["ablation_variant"]
out_dir = Path("results/dive_sampler_asl_search/sanity_configs")
out_dir.mkdir(parents=True, exist_ok=True)
config["experiment_name"] = f"sanity_{config['experiment_name']}"
config["checkpoint_dir"] = f"checkpoints/dive_sampler_asl_search_sanity/{variant}"
config["result_dir"] = f"results/dive_sampler_asl_search/sanity/{variant}"
config["debug_num_train_samples"] = 512
config["debug_num_valid_samples"] = 512
config["batch_size"] = 512
config["epochs"] = 1
config["early_stopping_patience"] = 1
out_path = out_dir / config_path.name
out_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
print(out_path.as_posix())
PY
}

run_training_if_needed() {
  local config="$1"
  local checkpoint_dir
  checkpoint_dir="$(python -c "import yaml; print(yaml.safe_load(open('${config}', encoding='utf-8'))['checkpoint_dir'])")"
  local checkpoint="${checkpoint_dir}/best_macro_f1.pt"
  echo "[TRAIN] ${config}"
  if [[ -f "${checkpoint}" ]]; then
    echo "[SKIP] checkpoint exists: ${checkpoint}"
  else
    python src/train_chunk_mil.py --config "${config}"
  fi
}

run_eval() {
  local config="$1"
  local checkpoint_dir
  local result_dir
  checkpoint_dir="$(python -c "import yaml; print(yaml.safe_load(open('${config}', encoding='utf-8'))['checkpoint_dir'])")"
  result_dir="$(python -c "import yaml; print(yaml.safe_load(open('${config}', encoding='utf-8'))['result_dir'])")"
  local checkpoint="${checkpoint_dir}/best_macro_f1.pt"

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

if [[ "${RUN_SANITY}" == "1" ]]; then
  echo "[DIVE 5/6] sanity train selected configs"
  for config in "${CONFIGS[@]}"; do
    sanity_config="$(make_sanity_config "${config}")"
    run_training_if_needed "${sanity_config}"
  done
else
  echo "[DIVE 5/6] sanity skipped"
fi

echo "[DIVE 6/6] full train/evaluate selected configs"
for config in "${CONFIGS[@]}"; do
  run_training_if_needed "${config}"
  run_eval "${config}"
done

python scripts/summarize_dive_sampler_asl_search.py \
  --manifest "${MANIFEST}" \
  --output_prefix results/dive_sampler_asl_search/summary

echo "[OK] DIVE sampler/ASL search finished."
