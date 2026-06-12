#!/usr/bin/env bash
#SBATCH --job-name=evmbert_base_pretrain
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --output=logs/slurm_evmbert_base_pretrain_%j.out
#SBATCH --error=logs/slurm_evmbert_base_pretrain_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc
conda activate correlascan_a40
mkdir -p logs

python scripts/build_full_bjut_pretrain_corpus.py \
  --config configs/pretrain_evm_bert_base_full_bjut.yaml

torchrun --standalone --nproc_per_node=2 src/pretrain_evm_bert.py \
  --config configs/pretrain_evm_bert_base_full_bjut.yaml
