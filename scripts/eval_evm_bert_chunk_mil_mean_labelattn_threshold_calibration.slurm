#!/usr/bin/env bash
#SBATCH --job-name=eval_mil_thr_calib
#SBATCH --partition=gpu-a30
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/slurm_eval_mil_thr_calib_%j.out
#SBATCH --error=logs/slurm_eval_mil_thr_calib_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc
conda activate correlascan_a40
mkdir -p logs

python src/evaluate_chunk_mil.py \
  --config configs/train_evm_bert_chunk_mil_mean_labelattn_weighted_server.yaml \
  --checkpoint checkpoints/train_evm_bert_chunk_mil_mean_labelattn_weighted/best_macro_f1.pt \
  --split valid \
  --threshold_search global_and_per_label \
  --output_prefix threshold_calibration

python src/evaluate_chunk_mil.py \
  --config configs/train_evm_bert_chunk_mil_mean_labelattn_weighted_server.yaml \
  --checkpoint checkpoints/train_evm_bert_chunk_mil_mean_labelattn_weighted/best_macro_f1.pt \
  --split test \
  --threshold_file results/train_evm_bert_chunk_mil_mean_labelattn_weighted/per_label_thresholds_valid.json \
  --global_threshold_file results/train_evm_bert_chunk_mil_mean_labelattn_weighted/valid_global_threshold_scan.json \
  --output_prefix threshold_calibration
