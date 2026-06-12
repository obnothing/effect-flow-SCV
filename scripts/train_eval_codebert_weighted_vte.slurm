#!/usr/bin/env bash
#SBATCH --job-name=codebert_vte_full
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/slurm_codebert_vte_full_%j.out
#SBATCH --error=logs/slurm_codebert_vte_full_%j.err

set -e

cd $SLURM_SUBMIT_DIR
source ~/.bashrc
conda activate correlascan_a40
mkdir -p logs

export NPROC_PER_NODE=2

echo "[STAGE 7] training weighted VTE model"
bash scripts/train_mlsmote_codebert_ddp_server.sh configs/train_mlsmote_codebert_weighted_vte_server.yaml

echo "[STAGE 7] strict test with checkpoint-selected threshold"
python src/evaluate.py \
  --config configs/train_mlsmote_codebert_weighted_vte_server.yaml \
  --checkpoint checkpoints/train_mlsmote_codebert_weighted_vte/best_macro_f1.pt \
  --split test \
  --threshold auto

echo "[STAGE 7] selecting validation per-label thresholds"
python src/evaluate.py \
  --config configs/train_mlsmote_codebert_weighted_vte_server.yaml \
  --checkpoint checkpoints/train_mlsmote_codebert_weighted_vte/best_macro_f1.pt \
  --split valid \
  --threshold_search per_label \
  --global_threshold 0.5 \
  --output_prefix weighted_vte_per_label

echo "[STAGE 7] strict test with validation-selected per-label thresholds"
python src/evaluate.py \
  --config configs/train_mlsmote_codebert_weighted_vte_server.yaml \
  --checkpoint checkpoints/train_mlsmote_codebert_weighted_vte/best_macro_f1.pt \
  --split test \
  --threshold_file results/train_mlsmote_codebert_weighted_vte/per_label_thresholds_valid.json \
  --global_threshold 0.5 \
  --output_prefix weighted_vte_per_label

echo "[OK] Stage 7 full pipeline finished"