#!/usr/bin/env bash
#SBATCH --job-name=eval_weighted_plthr
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --output=logs/slurm_eval_weighted_plthr_%j.out
#SBATCH --error=logs/slurm_eval_weighted_plthr_%j.err

set -e

cd $SLURM_SUBMIT_DIR
source ~/.bashrc
conda activate correlascan_a40
mkdir -p logs

python src/evaluate.py \
  --config configs/train_mlsmote_codebert_weighted_server.yaml \
  --checkpoint checkpoints/train_mlsmote_codebert_weighted/best_macro_f1.pt \
  --split valid \
  --threshold_search per_label \
  --global_threshold 0.5 \
  --output_prefix weighted_per_label

python src/evaluate.py \
  --config configs/train_mlsmote_codebert_weighted_server.yaml \
  --checkpoint checkpoints/train_mlsmote_codebert_weighted/best_macro_f1.pt \
  --split test \
  --threshold_file results/train_mlsmote_codebert_weighted/per_label_thresholds_valid.json \
  --global_threshold 0.5 \
  --output_prefix weighted_per_label
