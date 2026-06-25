#!/usr/bin/env bash
#SBATCH --job-name=extract_bjut_ef_chunks
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/slurm_extract_bjut_ef_chunks_%j.out
#SBATCH --error=logs/slurm_extract_bjut_ef_chunks_%j.err

set -eo pipefail

cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc
conda activate correlascan_a40
mkdir -p logs

python scripts/extract_evm_bert_chunk_features.py \
  --config configs/extract_bjut_effect_flow_chunks_mean_stride256.yaml

python scripts/check_evm_bert_chunk_features.py \
  --feature_dir data/features/effect_flow_evm_bert_chunks_mean_stride256/BJUT \
  --data_dir data/processed/BJUT_SC01_random_split \
  --expected_max_chunks 32 \
  --expected_feature_dim 768 \
  --expected_num_labels 10 \
  --output data/reports/check_effect_flow_bjut_chunks_mean_stride256_report.txt

