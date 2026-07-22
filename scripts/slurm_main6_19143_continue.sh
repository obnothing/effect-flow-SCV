#!/usr/bin/env bash
#SBATCH --job-name=main6_19143_continue
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/main6_19143_continue_%j.out
#SBATCH --error=logs/main6_19143_continue_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit this job with sbatch from the project root}"
source ~/.bashrc
conda activate correlascan_a40
mkdir -p logs

test -f checkpoints/pretrain_evm_bert_19143_base/hf_model/config.json

torchrun --standalone --nproc_per_node=2 src/pretrain_evm_bert.py \
  --config configs/pretrain_evm_bert_19143_base_continue.yaml

test -f checkpoints/pretrain_evm_bert_19143_base_continue/hf_model/config.json
test -f checkpoints/pretrain_evm_bert_19143_base_continue/evm_vocab.json
