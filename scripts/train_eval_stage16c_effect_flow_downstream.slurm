#!/usr/bin/env bash
#SBATCH --job-name=stage16c_ef_downstream
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --output=logs/slurm_stage16c_ef_downstream_%j.out
#SBATCH --error=logs/slurm_stage16c_ef_downstream_%j.err

set -eo pipefail

cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc
conda activate correlascan_a40
mkdir -p logs
export CUDA_VISIBLE_DEVICES=0,1

if [[ "$(python -c 'import torch; print(torch.cuda.device_count())')" -lt 2 ]]; then
  echo "[ERROR] Stage 16C expects two visible CUDA devices for training jobs."
  exit 1
fi

echo "[1/14] train BJUT effect-flow first512"
bash scripts/train_bjut_effect_flow_first512_weighted.slurm

echo "[2/14] evaluate BJUT effect-flow first512"
bash scripts/eval_bjut_effect_flow_first512_weighted.slurm

echo "[3/14] train DIVE effect-flow first512"
bash scripts/train_dive_effect_flow_first512_weighted.slurm

echo "[4/14] evaluate DIVE effect-flow first512"
bash scripts/eval_dive_effect_flow_first512_weighted.slurm

echo "[5/14] first512 summary checkpoint"
python scripts/summarize_stage16c_effect_flow_downstream.py

echo "[6/14] extract BJUT effect-flow stride256 masked_mean features"
bash scripts/extract_bjut_effect_flow_chunks_mean_stride256.slurm

echo "[7/14] check BJUT features"
python scripts/check_evm_bert_chunk_features.py \
  --feature_dir data/features/effect_flow_evm_bert_chunks_mean_stride256/BJUT \
  --data_dir data/processed/BJUT_SC01_random_split \
  --expected_max_chunks 32 \
  --expected_feature_dim 768 \
  --expected_num_labels 10 \
  --output data/reports/check_effect_flow_bjut_chunks_mean_stride256_report.txt

echo "[8/14] extract DIVE effect-flow stride256 masked_mean features"
bash scripts/extract_dive_effect_flow_chunks_mean_stride256.slurm

echo "[9/14] check DIVE features"
python scripts/check_evm_bert_chunk_features.py \
  --feature_dir data/features/effect_flow_evm_bert_chunks_mean_stride256/DIVE \
  --data_dir data/processed/DIVE_random_split \
  --expected_max_chunks 32 \
  --expected_feature_dim 768 \
  --expected_num_labels 8 \
  --output data/reports/check_effect_flow_dive_chunks_mean_stride256_report.txt

echo "[10/14] train BJUT effect-flow chunk-context MIL"
bash scripts/train_bjut_effect_flow_chunk_mil_ctx.slurm

echo "[11/14] evaluate BJUT effect-flow chunk-context MIL"
bash scripts/eval_bjut_effect_flow_chunk_mil_ctx.slurm

echo "[12/14] train DIVE effect-flow chunk-context MIL"
bash scripts/train_dive_effect_flow_chunk_mil_ctx.slurm

echo "[13/14] evaluate DIVE effect-flow chunk-context MIL"
bash scripts/eval_dive_effect_flow_chunk_mil_ctx.slurm

echo "[14/14] final Stage 16C summary and paired bootstrap"
python scripts/summarize_stage16c_effect_flow_downstream.py

echo "[OK] Stage 16C effect-flow downstream validation finished"

