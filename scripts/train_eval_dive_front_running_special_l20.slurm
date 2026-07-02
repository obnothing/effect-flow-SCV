#!/usr/bin/env bash
#SBATCH --job-name=dive_front_special
#SBATCH --partition=gpu-a10
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=120:00:00
#SBATCH --output=logs/slurm_dive_front_special_%j.out
#SBATCH --error=logs/slurm_dive_front_special_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs data/reports results/dive_front_running_special_search

set +u
source ~/.bashrc || true
conda activate correlascan_a40
set -u

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

MATRIX="configs/evef_mvd_dive_front_running_special_search.yaml"
MANIFEST="configs/generated/dive_front_running_special_search/manifest.yaml"
VARIANTS="${VARIANTS:-}"
RUN_SANITY="${RUN_SANITY:-1}"
RUN_AUDIT="${RUN_AUDIT:-1}"

echo "[INFO] current dir: $(pwd)"
echo "[INFO] python: $(which python)"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] VARIANTS=${VARIANTS:-ALL}"
echo "[INFO] RUN_SANITY=${RUN_SANITY}"
echo "[INFO] RUN_AUDIT=${RUN_AUDIT}"
nvidia-smi || true

echo "[DIVE 1/8] build Front Running special feature cache"
python scripts/build_front_running_special_features.py \
  --dataset DIVE \
  --num_workers 4

echo "[DIVE 2/8] check strong base chunk cache"
python scripts/check_evm_bert_chunk_features.py \
  --feature_dir data/features/continued_dive_evm_bert_dive_random_stride256_max64 \
  --data_dir data/processed/DIVE_random_split \
  --expected_max_chunks 64 \
  --expected_feature_dim 768 \
  --expected_num_labels 8 \
  --output data/reports/check_dive_front_special_base_features.txt

echo "[DIVE 3/8] check effect-flow semantic cache"
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
  --output data/reports/check_dive_front_special_semantic_features.txt

echo "[DIVE 4/8] check Front Running special cache alignment"
python scripts/check_front_running_special_features.py \
  --special_dir data/features/front_running_special/dive_random_stride256_max64 \
  --feature_dir data/features/continued_dive_evm_bert_dive_random_stride256_max64 \
  --data_dir data/processed/DIVE_random_split \
  --expected_max_chunks 64 \
  --expected_feature_dim 15 \
  --expected_num_labels 8 \
  --output data/reports/check_dive_front_running_special_features.txt

echo "[DIVE 5/8] generate experiment configs"
python scripts/generate_evef_mvd_ablation_configs.py \
  --matrix "${MATRIX}"

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

echo "[DIVE 6/8] validate Front Running special configs"
python - "${CONFIGS[@]}" <<'PY'
import sys
from pathlib import Path

import yaml

expected_features = [
    "approve_selector",
    "transfer_from_selector",
    "transfer_selector",
    "allowance_selector",
    "balance_of_selector",
    "swap_selector_family",
    "buy_sell_bid_order_text",
    "price_amount_balance_text",
    "calldata_to_state_write",
    "calldata_to_external_call",
    "external_call_then_state_write",
    "state_write_then_external_call",
    "deadline_or_time_guard_text",
    "slippage_or_min_amount_text",
    "commit_reveal_text",
]
for config_path in sys.argv[1:]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    variant = config["ablation_variant"]
    if config.get("model_type") != "evef_mvd_v2_side_evidence_mil":
        raise SystemExit(f"{config_path} must use evef_mvd_v2_side_evidence_mil")
    if not config.get("front_running_special_enabled"):
        raise SystemExit(f"{config_path} must enable front_running_special")
    if config.get("front_special_feature_names") != expected_features:
        raise SystemExit(f"{config_path} front_special_feature_names mismatch")
    if int(config.get("front_special_feature_dim", 0)) != len(expected_features):
        raise SystemExit(f"{config_path} front_special_feature_dim mismatch")
    if config.get("front_special_feature_dir") != "data/features/front_running_special/dive_random_stride256_max64":
        raise SystemExit(f"{config_path} front_special_feature_dir mismatch")
    if config.get("recognition_loss_type", "bce") != "bce":
        raise SystemExit(f"{config_path} should use BCE for this controlled experiment")
    if config.get("train_sampler", "shuffle") != "shuffle":
        raise SystemExit(f"{config_path} should keep train_sampler=shuffle")
    if int(config.get("batch_size", 0)) != 1024:
        raise SystemExit(f"{config_path} batch_size must be 1024")
    if int(config.get("max_chunks", 0)) != 64:
        raise SystemExit(f"{config_path} max_chunks must be 64")
    if int(config.get("epochs", 0)) != 50:
        raise SystemExit(f"{config_path} epochs must be 50")
    if int(config.get("early_stopping_patience", 0)) < 10:
        raise SystemExit(f"{config_path} early_stopping_patience must be >= 10")
    if float(config.get("coefficient_scale", 0.0)) != 1.5:
        raise SystemExit(f"{config_path} coefficient_scale must be 1.5")
    if variant.endswith("hn005") and float(config.get("front_hard_negative_lambda", 0.0)) != 0.05:
        raise SystemExit(f"{config_path} hard negative lambda mismatch")
    if variant.endswith("hn010") and float(config.get("front_hard_negative_lambda", 0.0)) != 0.1:
        raise SystemExit(f"{config_path} hard negative lambda mismatch")
    print(
        "[OK]",
        config_path,
        "supp=",
        config.get("front_running_confounder_suppression_weight"),
        "hn=",
        config.get("front_hard_negative_lambda"),
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
out_dir = Path("results/dive_front_running_special_search/sanity_configs")
out_dir.mkdir(parents=True, exist_ok=True)
config["experiment_name"] = f"sanity_{config['experiment_name']}"
config["checkpoint_dir"] = f"checkpoints/dive_front_running_special_search_sanity/{variant}"
config["result_dir"] = f"results/dive_front_running_special_search/sanity/{variant}"
config["debug_num_train_samples"] = 512
config["debug_num_valid_samples"] = 512
config["batch_size"] = 512
config["epochs"] = 1
config["early_stopping_patience"] = 1
out_path = out_dir / config_path.name
out_path.write_text(
    yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
    encoding="utf-8",
)
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
  echo "[DIVE 7/8] sanity train selected configs"
  for config in "${CONFIGS[@]}"; do
    sanity_config="$(make_sanity_config "${config}")"
    run_training_if_needed "${sanity_config}"
  done
else
  echo "[DIVE 7/8] sanity skipped"
fi

echo "[DIVE 8/8] full train/evaluate selected configs"
for config in "${CONFIGS[@]}"; do
  run_training_if_needed "${config}"
  run_eval "${config}"
done

python scripts/summarize_dive_front_running_special_search.py \
  --manifest "${MANIFEST}" \
  --output_prefix results/dive_front_running_special_search/summary

if [[ "${RUN_AUDIT}" == "1" ]]; then
  BEST_RESULT_DIR="$(python - "${MANIFEST}" <<'PY'
import json
import sys
from pathlib import Path

import yaml

manifest = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
best = None
for item in manifest["configs"]:
    config = yaml.safe_load(Path(item["config"]).read_text(encoding="utf-8"))
    result_dir = Path(config["result_dir"])
    report_path = result_dir / "test_threshold_calibration_metrics.json"
    if not report_path.exists():
        continue
    report = json.loads(report_path.read_text(encoding="utf-8"))
    for row in report.get("per_label_metrics", []):
        if row.get("label_name") == "Front Running":
            candidate = (float(row["f1"]), result_dir.as_posix())
            if best is None or candidate[0] > best[0]:
                best = candidate
            break
print("" if best is None else best[1])
PY
)"
  if [[ -n "${BEST_RESULT_DIR}" ]]; then
    echo "[AUDIT] best Front Running result dir: ${BEST_RESULT_DIR}"
    python scripts/audit_front_running_confusions.py \
      --predictions "${BEST_RESULT_DIR}/test_predictions_per_label.jsonl" \
      --top_chunks "${BEST_RESULT_DIR}/top_chunks_test_calibrated.jsonl" \
      --semantic_cache data/features/effect_flow_semantics/dive_random_stride256_max64/test.pt \
      --output_prefix "${BEST_RESULT_DIR}/front_running_confusion_audit"
  else
    echo "[WARN] no completed result found for Front Running confusion audit"
  fi
else
  echo "[AUDIT] skipped"
fi

echo "[OK] DIVE Front Running special evidence search finished."
