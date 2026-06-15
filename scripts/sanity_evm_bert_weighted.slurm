#!/usr/bin/env bash
#SBATCH --job-name=evmbert_sanity
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/slurm_evmbert_sanity_%j.out
#SBATCH --error=logs/slurm_evmbert_sanity_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc
conda activate correlascan_a40
mkdir -p logs

python scripts/finalize_pretrain_report.py
python scripts/check_evm_bert_pretrained_model.py \
  --config configs/sanity_evm_bert_weighted_server.yaml

python src/train.py \
  --config configs/sanity_evm_bert_weighted_server.yaml
