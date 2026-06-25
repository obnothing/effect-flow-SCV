#!/usr/bin/env bash
#SBATCH --job-name=eval_bjut_ef_mil_ctx
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/slurm_eval_bjut_ef_mil_ctx_%j.out
#SBATCH --error=logs/slurm_eval_bjut_ef_mil_ctx_%j.err

set -eo pipefail

cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc
conda activate correlascan_a40
mkdir -p logs

CONFIG="configs/train_bjut_effect_flow_chunk_mil_stride256_labelattn_ctx_weighted_server.yaml"
CHECKPOINT="checkpoints/train_bjut_effect_flow_chunk_mil_stride256_labelattn_ctx_weighted/best_macro_f1.pt"
RESULT_DIR="results/train_bjut_effect_flow_chunk_mil_stride256_labelattn_ctx_weighted"

python src/evaluate_chunk_mil.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --split valid \
  --threshold_search global_and_per_label \
  --output_prefix threshold_calibration

python src/evaluate_chunk_mil.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --split test \
  --threshold_file "$RESULT_DIR/per_label_thresholds_valid.json" \
  --global_threshold_file "$RESULT_DIR/valid_global_threshold_scan.json" \
  --output_prefix threshold_calibration \
  --save_predictions

