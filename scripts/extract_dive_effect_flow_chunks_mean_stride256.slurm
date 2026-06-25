#!/usr/bin/env bash
#SBATCH --job-name=extract_dive_ef_chunks
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/slurm_extract_dive_ef_chunks_%j.out
#SBATCH --error=logs/slurm_extract_dive_ef_chunks_%j.err

set -eo pipefail

cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc
conda activate correlascan_a40
mkdir -p logs

python scripts/extract_evm_bert_chunk_features.py \
  --config configs/extract_dive_effect_flow_chunks_mean_stride256.yaml

python scripts/check_evm_bert_chunk_features.py \
  --feature_dir data/features/effect_flow_evm_bert_chunks_mean_stride256/DIVE \
  --data_dir data/processed/DIVE_random_split \
  --expected_max_chunks 32 \
  --expected_feature_dim 768 \
  --expected_num_labels 8 \
  --output data/reports/check_effect_flow_dive_chunks_mean_stride256_report.txt

