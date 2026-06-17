#!/usr/bin/env bash
#SBATCH --job-name=extract_stride256
#SBATCH --partition=gpu-a30
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/slurm_extract_stride256_%j.out
#SBATCH --error=logs/slurm_extract_stride256_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc
conda activate correlascan_a40
mkdir -p logs

python scripts/extract_evm_bert_chunk_features.py \
  --config configs/extract_evm_bert_chunk_features_mean_stride256.yaml

python scripts/check_evm_bert_chunk_features.py \
  --feature_dir data/features/evm_bert_chunks_mean_stride256 \
  --coverage_baseline_dir data/features/evm_bert_chunks_mean \
  --expected_max_chunks 32 \
  --expected_feature_dim 768 \
  --output data/reports/check_evm_bert_chunk_features_mean_stride256_report.txt
