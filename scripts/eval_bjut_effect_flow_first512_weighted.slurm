#!/usr/bin/env bash
#SBATCH --job-name=eval_bjut_ef_first512
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/slurm_eval_bjut_ef_first512_%j.out
#SBATCH --error=logs/slurm_eval_bjut_ef_first512_%j.err

set -eo pipefail

cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc
conda activate correlascan_a40
mkdir -p logs

CONFIG="configs/train_bjut_effect_flow_first512_weighted_server.yaml"
CHECKPOINT="checkpoints/train_bjut_effect_flow_first512_weighted/best_macro_f1.pt"
RESULT_DIR="results/train_bjut_effect_flow_first512_weighted"

python src/evaluate.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --split valid \
  --threshold_search global_and_per_label \
  --global_threshold 0.5

python src/evaluate.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --split test \
  --threshold_file "$RESULT_DIR/per_label_thresholds_valid.json" \
  --global_threshold_file "$RESULT_DIR/valid_global_threshold_scan.json" \
  --global_threshold 0.5 \
  --save_predictions

