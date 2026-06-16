#!/usr/bin/env bash
#SBATCH --job-name=extract_evmbert_mean
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/slurm_extract_evmbert_mean_%j.out
#SBATCH --error=logs/slurm_extract_evmbert_mean_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc
conda activate correlascan_a40
mkdir -p logs

python scripts/extract_evm_bert_chunk_features.py \
  --config configs/extract_evm_bert_chunk_features_mean.yaml

python scripts/check_evm_bert_chunk_features.py \
  --feature_dir data/features/evm_bert_chunks_mean \
  --output data/reports/check_evm_bert_chunk_features_mean_report.txt

