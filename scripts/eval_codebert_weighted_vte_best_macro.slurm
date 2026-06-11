#!/usr/bin/env bash
#SBATCH --job-name=eval_weighted_vte_macro
#SBATCH --partition=gpu-a30
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --output=logs/slurm_eval_weighted_vte_macro_%j.out
#SBATCH --error=logs/slurm_eval_weighted_vte_macro_%j.err

set -e

cd $SLURM_SUBMIT_DIR
source ~/.bashrc
conda activate correlascan_a40
mkdir -p logs

python src/evaluate.py \
  --config configs/train_mlsmote_codebert_weighted_vte_server.yaml \
  --checkpoint checkpoints/train_mlsmote_codebert_weighted_vte/best_macro_f1.pt \
  --split test \
  --threshold auto
