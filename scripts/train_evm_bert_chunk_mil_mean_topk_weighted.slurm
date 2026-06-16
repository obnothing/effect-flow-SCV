#!/usr/bin/env bash
#SBATCH --job-name=mil_mean_topk
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/slurm_mil_mean_topk_%j.out
#SBATCH --error=logs/slurm_mil_mean_topk_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc
conda activate correlascan_a40
mkdir -p logs

python src/train_chunk_mil.py \
  --config configs/train_evm_bert_chunk_mil_mean_topk_weighted_server.yaml

