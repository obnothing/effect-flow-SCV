#!/usr/bin/env bash
#SBATCH --job-name=evmbert_weighted_ddp
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --output=logs/slurm_evmbert_weighted_%j.out
#SBATCH --error=logs/slurm_evmbert_weighted_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc
conda activate correlascan_a40
mkdir -p logs

python scripts/finalize_pretrain_report.py
python scripts/check_evm_bert_pretrained_model.py \
  --config configs/train_evm_bert_weighted_server.yaml

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=2 \
  src/train.py \
  --config configs/train_evm_bert_weighted_server.yaml
