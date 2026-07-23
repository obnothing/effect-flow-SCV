#!/usr/bin/env bash
#SBATCH --job-name=main6_effectflow_pretrain
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/main6_effectflow_pretrain_%j.out
#SBATCH --error=logs/main6_effectflow_pretrain_%j.err

cd "${SLURM_SUBMIT_DIR:?Submit this job with sbatch from the project root}"
source ~/.bashrc
conda activate correlascan_a40
set -euo pipefail
mkdir -p logs

test -f checkpoints/pretrain_evm_bert_19143_base_continue/hf_model/config.json
test -f checkpoints/pretrain_evm_bert_19143_base_continue/evm_vocab.json
test -f data/processed/DIVE_main6_random_split/train.jsonl

python scripts/audit_main6_random_pretrain_protocol.py
bash scripts/run_main6_random_090.sh corpus

test -f data/processed/effect_flow_pretrain/DIVE_main6_random_train_new/train_effect_flow_chunks.jsonl
test -f data/processed/effect_flow_pretrain/ethereum_19143/train_effect_flow_chunks.jsonl

pretrain_args=(--config configs/pretrain_effect_flow_evm_bert_main6_random_train_new.yaml)
if [[ -n "${RESUME_FROM:-}" ]]; then
  pretrain_args+=(--resume "$RESUME_FROM")
fi
torchrun --standalone --nproc_per_node=2 src/pretrain_effect_flow_evm_bert_main6.py \
  "${pretrain_args[@]}"

test -f checkpoints/pretrain_effect_flow_evm_bert_main6_random_train_new/hf_model/config.json
