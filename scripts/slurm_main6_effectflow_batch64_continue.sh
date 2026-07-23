#!/usr/bin/env bash
#SBATCH --job-name=main6_effectflow_b64_continue
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/main6_effectflow_b64_continue_%j.out
#SBATCH --error=logs/main6_effectflow_b64_continue_%j.err

cd "${SLURM_SUBMIT_DIR:?Submit this job with sbatch from the project root}"
source ~/.bashrc
conda activate correlascan_a40
set -euo pipefail
mkdir -p logs

test -f data/reports/stage16b_effect_flow_pretraining_main6_random_train_new_report.json
test -f checkpoints/pretrain_effect_flow_evm_bert_main6_random_train_new/last.pt
test -f data/processed/effect_flow_pretrain/DIVE_main6_random_train_new/train_effect_flow_chunks.jsonl
test -f data/processed/effect_flow_pretrain/ethereum_19143/train_effect_flow_chunks.jsonl

torchrun --standalone --nproc_per_node=2 src/pretrain_effect_flow_evm_bert_main6.py \
  --config configs/pretrain_effect_flow_evm_bert_main6_random_train_batch64_continue.yaml

test -f checkpoints/pretrain_effect_flow_evm_bert_main6_random_train_batch64_continue/hf_model/config.json
