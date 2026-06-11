#!/usr/bin/env bash
#SBATCH --job-name=eval_codebert_per_label_threshold
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/slurm_eval_codebert_per_label_threshold_%j.out
#SBATCH --error=logs/slurm_eval_codebert_per_label_threshold_%j.err

set -e

cd $SLURM_SUBMIT_DIR
source ~/.bashrc
conda activate correlascan_a40
mkdir -p logs

python src/evaluate.py \
  --config configs/train_mlsmote_codebert_server.yaml \
  --checkpoint checkpoints/train_mlsmote_codebert/best_macro_f1.pt \
  --split valid \
  --threshold_search per_label \
  --global_threshold 0.2

python src/evaluate.py \
  --config configs/train_mlsmote_codebert_server.yaml \
  --checkpoint checkpoints/train_mlsmote_codebert/best_macro_f1.pt \
  --split test \
  --threshold_file results/train_mlsmote_codebert/per_label_thresholds_valid.json \
  --global_threshold 0.2
