#!/usr/bin/env bash
#SBATCH --job-name=chunk_context_mil
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/slurm_chunk_context_mil_%j.out
#SBATCH --error=logs/slurm_chunk_context_mil_%j.err

set -eo pipefail

cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc
conda activate correlascan_a40
mkdir -p logs

FEATURE_DIR="data/features/evm_bert_chunks_mean_stride256"
for split in train valid test; do
  if [[ ! -f "$FEATURE_DIR/$split.pt" ]]; then
    echo "[ERROR] missing feature cache: $FEATURE_DIR/$split.pt"
    exit 1
  fi
done

echo "[CHECK] existing stride256 masked-mean feature cache"
python scripts/check_evm_bert_chunk_features.py \
  --feature_dir "$FEATURE_DIR" \
  --expected_max_chunks 32 \
  --expected_feature_dim 768 \
  --expected_num_labels 10 \
  --output data/reports/check_evm_bert_chunk_context_features.txt

echo "[SANITY] chunk-level Transformer context encoder"
python src/train_chunk_mil.py \
  --config configs/sanity_evm_bert_chunk_context_labelattn_weighted_server.yaml

echo "[TRAIN] full chunk-context MIL"
python src/train_chunk_mil.py \
  --config configs/train_evm_bert_chunk_context_labelattn_weighted_server.yaml

CHECKPOINT="checkpoints/train_evm_bert_chunk_context_labelattn_weighted/best_macro_f1.pt"

echo "[VALID] select global and per-label thresholds"
python src/evaluate_chunk_mil.py \
  --config configs/train_evm_bert_chunk_context_labelattn_weighted_server.yaml \
  --checkpoint "$CHECKPOINT" \
  --split valid \
  --threshold_search global_and_per_label \
  --output_prefix threshold_calibration

echo "[TEST] apply validation-selected thresholds"
python src/evaluate_chunk_mil.py \
  --config configs/train_evm_bert_chunk_context_labelattn_weighted_server.yaml \
  --checkpoint "$CHECKPOINT" \
  --split test \
  --threshold_file results/train_evm_bert_chunk_context_labelattn_weighted/per_label_thresholds_valid.json \
  --global_threshold_file results/train_evm_bert_chunk_context_labelattn_weighted/valid_global_threshold_scan.json \
  --output_prefix threshold_calibration \
  --save_predictions

echo "[OK] chunk-context MIL experiment completed"
