#!/usr/bin/env bash
#SBATCH --job-name=stage14a_fusion
#SBATCH --partition=gpu-a10
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/slurm_stage14a_fusion_%j.out
#SBATCH --error=logs/slurm_stage14a_fusion_%j.err

set -eo pipefail

cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc
conda activate correlascan_a40
mkdir -p logs

ensure_checkpoint() {
  local checkpoint="$1"
  if [[ ! -f "$checkpoint" ]]; then
    echo "[ERROR] missing checkpoint required to generate validation predictions: $checkpoint"
    echo "[ERROR] Stage 14A does not retrain models. Restore the checkpoint and resubmit."
    exit 1
  fi
}

ensure_generic_predictions() {
  local name="$1"
  local config="$2"
  local checkpoint="$3"
  local result_dir="$4"
  for split in valid test; do
    local prediction="$result_dir/${split}_predictions.jsonl"
    if [[ -f "$prediction" ]]; then
      echo "[SKIP] $name $split predictions already exist"
      continue
    fi
    ensure_checkpoint "$checkpoint"
    echo "[EVALUATE] $name $split predictions"
    python src/evaluate.py \
      --config "$config" \
      --checkpoint "$checkpoint" \
      --split "$split" \
      --threshold 0.5 \
      --save_predictions
  done
}

ensure_chunk_predictions() {
  local name="$1"
  local config="$2"
  local checkpoint="$3"
  local result_dir="$4"
  for split in valid test; do
    local prediction="$result_dir/${split}_predictions.jsonl"
    if [[ -f "$prediction" ]]; then
      echo "[SKIP] $name $split predictions already exist"
      continue
    fi
    ensure_checkpoint "$checkpoint"
    echo "[EVALUATE] $name $split predictions"
    python src/evaluate_chunk_mil.py \
      --config "$config" \
      --checkpoint "$checkpoint" \
      --split "$split" \
      --threshold 0.5 \
      --save_predictions
  done
}

echo "[ARTIFACTS] BJUT strict grouped models"
ensure_generic_predictions \
  first512 \
  configs/train_evm_bert_weighted_server.yaml \
  checkpoints/train_evm_bert_weighted/best_macro_f1.pt \
  results/train_evm_bert_weighted
ensure_chunk_predictions \
  strict_chunk_mil \
  configs/train_evm_bert_chunk_mil_mean_stride256_labelattn_weighted_server.yaml \
  checkpoints/train_evm_bert_chunk_mil_mean_stride256_labelattn_weighted/best_macro_f1.pt \
  results/train_evm_bert_chunk_mil_mean_stride256_labelattn_weighted
ensure_generic_predictions \
  codebert_weighted \
  configs/train_mlsmote_codebert_weighted_server.yaml \
  checkpoints/train_mlsmote_codebert_weighted/best_macro_f1.pt \
  results/train_mlsmote_codebert_weighted

echo "[ARTIFACTS] BJUT random models"
ensure_chunk_predictions \
  bjut_random_chunk_mil \
  configs/train_continued_dive_evm_bert_bjut_random_stride256_mil.yaml \
  checkpoints/train_continued_dive_evm_bert_bjut_random_stride256_mil/best_macro_f1.pt \
  results/train_continued_dive_evm_bert_bjut_random_stride256_mil
ensure_chunk_predictions \
  bjut_random_chunk_context \
  configs/train_evm_bert_chunk_context_labelattn_weighted_server.yaml \
  checkpoints/train_continued_dive_evm_bert_bjut_random_stride256_context_mil/best_macro_f1.pt \
  results/train_continued_dive_evm_bert_bjut_random_stride256_context_mil

echo "[ARTIFACTS] DIVE random models"
ensure_chunk_predictions \
  dive_random_chunk_mil \
  configs/train_continued_dive_evm_bert_dive_random_stride256_mil.yaml \
  checkpoints/train_continued_dive_evm_bert_dive_random_stride256_mil/best_macro_f1.pt \
  results/train_continued_dive_evm_bert_dive_random_stride256_mil
ensure_chunk_predictions \
  dive_random_chunk_context \
  configs/train_dive_evm_bert_chunk_context_labelattn_weighted_server.yaml \
  checkpoints/train_continued_dive_evm_bert_dive_random_stride256_context_mil/best_macro_f1.pt \
  results/train_continued_dive_evm_bert_dive_random_stride256_context_mil

echo "[FUSION] BJUT strict grouped"
python scripts/fuse_predictions_validation_search.py \
  --config configs/fusion_stage14a_bjut_strict.yaml

echo "[FUSION] BJUT random upper bound"
python scripts/fuse_predictions_validation_search.py \
  --config configs/fusion_stage14a_bjut_random.yaml

echo "[FUSION] DIVE random upper bound"
python scripts/fuse_predictions_validation_search.py \
  --config configs/fusion_stage14a_dive_random.yaml

echo "[OK] Stage 14A fusion searches completed"
