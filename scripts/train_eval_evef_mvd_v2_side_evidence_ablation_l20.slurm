#!/usr/bin/env bash
#SBATCH --job-name=evef_v2_ablation
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=96:00:00
#SBATCH --output=logs/slurm_evef_v2_ablation_%j.out
#SBATCH --error=logs/slurm_evef_v2_ablation_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs data/reports results/ablation_evef_mvd_v2

set +u
source ~/.bashrc || true
conda activate correlascan_a40
set -u

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

DATASETS="${DATASETS:-BJUT DIVE}"
VARIANTS="${VARIANTS:-no_side side_scale_025 side_scale_050 side_scale_100 side_scale_200 side_mechanism_weighted}"

echo "[INFO] current dir: $(pwd)"
echo "[INFO] python: $(which python)"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] DATASETS=${DATASETS}"
echo "[INFO] VARIANTS=${VARIANTS}"
nvidia-smi || true

echo "[STAGE] generate ablation configs"
python scripts/generate_evef_mvd_ablation_configs.py \
  --matrix configs/evef_mvd_v2_side_evidence_ablation.yaml

check_bjut() {
  echo "[BJUT] check strong base chunk cache"
  python scripts/check_evm_bert_chunk_features.py \
    --feature_dir data/features/continued_dive_evm_bert_bjut_random_stride256 \
    --data_dir data/processed/BJUT_SC01_random_split \
    --expected_max_chunks 32 \
    --expected_feature_dim 768 \
    --expected_num_labels 10 \
    --output data/reports/check_bjut_evef_mvd_v2_ablation_base_features.txt

  echo "[BJUT] check effect-flow semantic cache"
  python scripts/check_effect_flow_semantic_features.py \
    --semantic_dir data/features/effect_flow_semantics/bjut_random_stride256_max32 \
    --feature_dir data/features/continued_dive_evm_bert_bjut_random_stride256 \
    --data_dir data/processed/BJUT_SC01_random_split \
    --expected_max_chunks 32 \
    --expected_efpp_dim 22 \
    --expected_etp_dim 16 \
    --expected_relation_dim 6 \
    --expected_global_template_dim 17 \
    --expected_num_labels 10 \
    --output data/reports/check_bjut_evef_mvd_v2_ablation_semantic_features.txt
}

check_dive() {
  echo "[DIVE] check strong base chunk cache"
  python scripts/check_evm_bert_chunk_features.py \
    --feature_dir data/features/continued_dive_evm_bert_dive_random_stride256_max64 \
    --data_dir data/processed/DIVE_random_split \
    --expected_max_chunks 64 \
    --expected_feature_dim 768 \
    --expected_num_labels 8 \
    --output data/reports/check_dive_evef_mvd_v2_ablation_base_features.txt

  echo "[DIVE] check effect-flow semantic cache"
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
    --output data/reports/check_dive_evef_mvd_v2_ablation_semantic_features.txt
}

run_variant() {
  local dataset="$1"
  local variant="$2"
  local dataset_lower
  dataset_lower="$(echo "$dataset" | tr '[:upper:]' '[:lower:]')"
  local config="configs/generated/evef_mvd_v2_ablation/train_${dataset_lower}_${variant}.yaml"
  local experiment="train_${dataset_lower}_${variant}"
  local checkpoint="checkpoints/ablation_evef_mvd_v2/${experiment}/best_macro_f1.pt"
  local result_dir="results/ablation_evef_mvd_v2/${experiment}"

  echo "[RUN] dataset=${dataset} variant=${variant}"
  python src/train_chunk_mil.py --config "$config"
  python src/evaluate_chunk_mil.py \
    --config "$config" \
    --checkpoint "$checkpoint" \
    --split valid \
    --threshold_search global_and_per_label
  python src/evaluate_chunk_mil.py \
    --config "$config" \
    --checkpoint "$checkpoint" \
    --split test \
    --threshold_file "${result_dir}/per_label_thresholds_valid.json" \
    --global_threshold_file "${result_dir}/valid_global_threshold_scan.json" \
    --save_predictions

  if [[ "$variant" != "no_side" ]]; then
    python scripts/summarize_behavior_contributions.py \
      --config "$config" \
      --checkpoint "$checkpoint" \
      --split test \
      --threshold_file "${result_dir}/per_label_thresholds_valid.json" \
      --output_prefix "${result_dir}/behavior_contribution_summary"
  fi
}

for dataset in ${DATASETS}; do
  if [[ "$dataset" == "BJUT" ]]; then
    check_bjut
  elif [[ "$dataset" == "DIVE" ]]; then
    check_dive
  else
    echo "[ERROR] unsupported dataset: ${dataset}" >&2
    exit 1
  fi

  for variant in ${VARIANTS}; do
    run_variant "$dataset" "$variant"
  done
done

python scripts/summarize_side_evidence_ablation.py \
  --manifest configs/generated/evef_mvd_v2_ablation/manifest.yaml \
  --output_prefix results/ablation_evef_mvd_v2/side_evidence_ablation_summary

echo "[OK] EVEF-MVD v2 side-evidence ablation finished."
