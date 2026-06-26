#!/usr/bin/env bash
#SBATCH --job-name=ef_guided_mil
#SBATCH --partition=gpu-a10
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/slurm_ef_guided_mil_%j.out
#SBATCH --error=logs/slurm_ef_guided_mil_%j.err

set -eo pipefail

cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc
conda activate correlascan_a40
mkdir -p logs
export CUDA_VISIBLE_DEVICES=0,1

if [[ "$(python -c 'import torch; print(torch.cuda.device_count())')" -lt 2 ]]; then
  echo "[ERROR] this experiment requires two visible CUDA devices"
  exit 1
fi

check_chunk_cache() {
  local feature_dir="$1"
  local data_dir="$2"
  local max_chunks="$3"
  local num_labels="$4"
  local output="$5"
  python scripts/check_evm_bert_chunk_features.py \
    --feature_dir "$feature_dir" \
    --data_dir "$data_dir" \
    --expected_max_chunks "$max_chunks" \
    --expected_feature_dim 768 \
    --expected_num_labels "$num_labels" \
    --output "$output"
}

check_semantic_cache() {
  local semantic_dir="$1"
  local feature_dir="$2"
  local data_dir="$3"
  local max_chunks="$4"
  local num_labels="$5"
  local output="$6"
  python scripts/check_effect_flow_semantic_features.py \
    --semantic_dir "$semantic_dir" \
    --feature_dir "$feature_dir" \
    --data_dir "$data_dir" \
    --expected_max_chunks "$max_chunks" \
    --expected_efpp_dim 22 \
    --expected_etp_dim 16 \
    --expected_num_labels "$num_labels" \
    --output "$output"
}

run_train_eval() {
  local sanity_config="$1"
  local train_config="$2"
  local checkpoint_dir="$3"
  local result_dir="$4"

  python src/train_chunk_mil.py --config "$sanity_config"
  python src/train_chunk_mil.py --config "$train_config"
  python src/evaluate_chunk_mil.py \
    --config "$train_config" \
    --checkpoint "$checkpoint_dir/best_macro_f1.pt" \
    --split valid \
    --threshold_search global_and_per_label \
    --output_prefix threshold_calibration
  python src/evaluate_chunk_mil.py \
    --config "$train_config" \
    --checkpoint "$checkpoint_dir/best_macro_f1.pt" \
    --split test \
    --threshold_file "$result_dir/per_label_thresholds_valid.json" \
    --global_threshold_file "$result_dir/valid_global_threshold_scan.json" \
    --output_prefix threshold_calibration \
    --save_predictions
  python scripts/summarize_effect_flow_guided_semantics.py \
    --config "$train_config"
}

echo "[BJUT 1/5] check 32-chunk base feature cache"
check_chunk_cache \
  data/features/continued_dive_evm_bert_bjut_random_stride256 \
  data/processed/BJUT_SC01_random_split \
  32 \
  10 \
  data/reports/check_bjut_effect_flow_guided_base_features.txt

echo "[BJUT 2/5] extract 32-chunk effect-flow semantic cache"
python scripts/extract_effect_flow_semantic_features.py \
  --config configs/extract_bjut_effect_flow_semantics_stride256.yaml

echo "[BJUT 3/5] check 32-chunk semantic cache"
check_semantic_cache \
  data/features/effect_flow_semantics/bjut_random_stride256_max32 \
  data/features/continued_dive_evm_bert_bjut_random_stride256 \
  data/processed/BJUT_SC01_random_split \
  32 \
  10 \
  data/reports/check_bjut_effect_flow_guided_semantic_features.txt

echo "[BJUT 4/5] sanity + full train + eval"
run_train_eval \
  configs/sanity_bjut_effect_flow_guided_mil.yaml \
  configs/train_bjut_effect_flow_guided_mil.yaml \
  checkpoints/train_bjut_effect_flow_guided_mil \
  results/train_bjut_effect_flow_guided_mil

echo "[BJUT 5/5] complete"

echo "[DIVE 1/5] extract/check 64-chunk base feature cache"
python scripts/extract_evm_bert_chunk_features.py \
  --config configs/extract_continued_dive_evm_bert_dive_random_stride256_max64.yaml
check_chunk_cache \
  data/features/continued_dive_evm_bert_dive_random_stride256_max64 \
  data/processed/DIVE_random_split \
  64 \
  8 \
  data/reports/check_dive_effect_flow_guided_base_features.txt

echo "[DIVE 2/5] extract 64-chunk effect-flow semantic cache"
python scripts/extract_effect_flow_semantic_features.py \
  --config configs/extract_dive_effect_flow_semantics_stride256_max64.yaml

echo "[DIVE 3/5] check 64-chunk semantic cache"
check_semantic_cache \
  data/features/effect_flow_semantics/dive_random_stride256_max64 \
  data/features/continued_dive_evm_bert_dive_random_stride256_max64 \
  data/processed/DIVE_random_split \
  64 \
  8 \
  data/reports/check_dive_effect_flow_guided_semantic_features.txt

echo "[DIVE 4/5] sanity + full train + eval"
run_train_eval \
  configs/sanity_dive_effect_flow_guided_mil.yaml \
  configs/train_dive_effect_flow_guided_mil.yaml \
  checkpoints/train_dive_effect_flow_guided_mil \
  results/train_dive_effect_flow_guided_mil

echo "[DIVE 5/5] complete"
echo "[OK] Effect-flow guided MIL two-dataset experiment completed"
