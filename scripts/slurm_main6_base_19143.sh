#!/usr/bin/env bash
#SBATCH --job-name=main6_19143_base
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/main6_19143_base_%j.out
#SBATCH --error=logs/main6_19143_base_%j.err

cd "${SLURM_SUBMIT_DIR:?Submit this job with sbatch from the project root}"
source ~/.bashrc
conda activate correlascan_a40
set -euo pipefail
mkdir -p logs

test -f data/processed/ethereum_public_pretrain_19143_unique_runtime/runtime_opcode.jsonl
test -f data/processed/ethereum_public_pretrain_19143_unique_runtime/evm_vocab.json

torchrun --standalone --nproc_per_node=2 src/pretrain_evm_bert.py \
  --config configs/pretrain_evm_bert_19143_base.yaml

test -f checkpoints/pretrain_evm_bert_19143_base/hf_model/config.json
