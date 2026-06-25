#!/usr/bin/env bash
#SBATCH --job-name=dive_ef_mil_ctx
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/slurm_dive_ef_mil_ctx_%j.out
#SBATCH --error=logs/slurm_dive_ef_mil_ctx_%j.err

set -eo pipefail

cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc
conda activate correlascan_a40
mkdir -p logs
export CUDA_VISIBLE_DEVICES=0,1

python scripts/check_evm_bert_chunk_features.py \
  --feature_dir data/features/effect_flow_evm_bert_chunks_mean_stride256/DIVE \
  --data_dir data/processed/DIVE_random_split \
  --expected_max_chunks 32 \
  --expected_feature_dim 768 \
  --expected_num_labels 8 \
  --output data/reports/check_effect_flow_dive_chunks_mean_stride256_report.txt

python src/train_chunk_mil.py \
  --config configs/train_dive_effect_flow_chunk_mil_stride256_labelattn_ctx_weighted_server.yaml

