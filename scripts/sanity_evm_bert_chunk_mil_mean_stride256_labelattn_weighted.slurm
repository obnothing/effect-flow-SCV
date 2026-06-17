#!/usr/bin/env bash
#SBATCH --job-name=sanity_stride256_mil
#SBATCH --partition=gpu-a30
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/slurm_sanity_stride256_mil_%j.out
#SBATCH --error=logs/slurm_sanity_stride256_mil_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc
conda activate correlascan_a40
mkdir -p logs

python src/train_chunk_mil.py \
  --config configs/sanity_evm_bert_chunk_mil_mean_stride256_labelattn_weighted_server.yaml

