#!/usr/bin/env bash
#SBATCH --job-name=two_random_chunk_context
#SBATCH --partition=gpu-a10
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/slurm_two_random_chunk_context_%j.out
#SBATCH --error=logs/slurm_two_random_chunk_context_%j.err

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

check_cache() {
  local feature_dir="$1"
  local num_labels="$2"
  local report_path="$3"
  for split in train valid test; do
    if [[ ! -f "$feature_dir/$split.pt" ]]; then
      echo "[ERROR] missing feature cache: $feature_dir/$split.pt"
      exit 1
    fi
  done
  python scripts/check_evm_bert_chunk_features.py \
    --feature_dir "$feature_dir" \
    --expected_max_chunks 32 \
    --expected_feature_dim 768 \
    --expected_num_labels "$num_labels" \
    --output "$report_path"
}

echo "[BJUT 1/4] check random-split feature cache"
check_cache \
  data/features/continued_dive_evm_bert_bjut_random_stride256 \
  10 \
  data/reports/check_bjut_random_chunk_context_features.txt

echo "[BJUT 2/4] two-GPU sanity"
python src/train_chunk_mil.py \
  --config configs/sanity_evm_bert_chunk_context_labelattn_weighted_server.yaml

echo "[BJUT 3/4] two-GPU full training"
python src/train_chunk_mil.py \
  --config configs/train_evm_bert_chunk_context_labelattn_weighted_server.yaml

BJUT_CONFIG="configs/train_evm_bert_chunk_context_labelattn_weighted_server.yaml"
BJUT_CHECKPOINT="checkpoints/train_continued_dive_evm_bert_bjut_random_stride256_context_mil/best_macro_f1.pt"
BJUT_RESULTS="results/train_continued_dive_evm_bert_bjut_random_stride256_context_mil"

echo "[BJUT 4/4] validation calibration and test application"
python src/evaluate_chunk_mil.py \
  --config "$BJUT_CONFIG" \
  --checkpoint "$BJUT_CHECKPOINT" \
  --split valid \
  --threshold_search global_and_per_label \
  --output_prefix threshold_calibration
python src/evaluate_chunk_mil.py \
  --config "$BJUT_CONFIG" \
  --checkpoint "$BJUT_CHECKPOINT" \
  --split test \
  --threshold_file "$BJUT_RESULTS/per_label_thresholds_valid.json" \
  --global_threshold_file "$BJUT_RESULTS/valid_global_threshold_scan.json" \
  --output_prefix threshold_calibration \
  --save_predictions

echo "[INFO] BJUT complete; its Python processes have exited and released both GPUs"

echo "[DIVE 1/4] check random-split feature cache"
check_cache \
  data/features/continued_dive_evm_bert_dive_random_stride256 \
  8 \
  data/reports/check_dive_random_chunk_context_features.txt

echo "[DIVE 2/4] two-GPU sanity"
python src/train_chunk_mil.py \
  --config configs/sanity_dive_evm_bert_chunk_context_labelattn_weighted_server.yaml

echo "[DIVE 3/4] two-GPU full training"
python src/train_chunk_mil.py \
  --config configs/train_dive_evm_bert_chunk_context_labelattn_weighted_server.yaml

DIVE_CONFIG="configs/train_dive_evm_bert_chunk_context_labelattn_weighted_server.yaml"
DIVE_CHECKPOINT="checkpoints/train_continued_dive_evm_bert_dive_random_stride256_context_mil/best_macro_f1.pt"
DIVE_RESULTS="results/train_continued_dive_evm_bert_dive_random_stride256_context_mil"

echo "[DIVE 4/4] validation calibration and test application"
python src/evaluate_chunk_mil.py \
  --config "$DIVE_CONFIG" \
  --checkpoint "$DIVE_CHECKPOINT" \
  --split valid \
  --threshold_search global_and_per_label \
  --output_prefix threshold_calibration
python src/evaluate_chunk_mil.py \
  --config "$DIVE_CONFIG" \
  --checkpoint "$DIVE_CHECKPOINT" \
  --split test \
  --threshold_file "$DIVE_RESULTS/per_label_thresholds_valid.json" \
  --global_threshold_file "$DIVE_RESULTS/valid_global_threshold_scan.json" \
  --output_prefix threshold_calibration \
  --save_predictions

python scripts/summarize_chunk_context_two_random_results.py

echo "[OK] sequential two-dataset, two-GPU chunk-context experiment completed"
