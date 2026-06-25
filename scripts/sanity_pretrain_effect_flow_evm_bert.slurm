#!/usr/bin/env bash
#SBATCH --job-name=eflow_sanity
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/slurm_eflow_sanity_%j.out
#SBATCH --error=logs/slurm_eflow_sanity_%j.err

set -eo pipefail

cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc
conda activate correlascan_a40
mkdir -p logs

for required in \
  checkpoints/pretrain_evm_bert_base_full_bjut/hf_model/config.json \
  data/processed/BJUT_SC01/evm_vocab.json \
  data/processed/effect_flow_pretrain/BJUT/train_effect_flow_chunks.jsonl \
  data/processed/effect_flow_pretrain/DIVE/train_effect_flow_chunks.jsonl \
  configs/effect_flow_efpp_conservative_22.json; do
  if [[ ! -f "$required" ]]; then
    echo "[ERROR] missing required file: $required"
    exit 1
  fi
done

BASE=checkpoints/pretrain_evm_bert_base_full_bjut/hf_model
if [[ ! -f "$BASE/model.safetensors" && ! -f "$BASE/pytorch_model.bin" ]]; then
  echo "[ERROR] existing EVM-BERT weights are missing from $BASE"
  exit 1
fi

export OMP_NUM_THREADS=2
torchrun --standalone --nproc_per_node=2 \
  src/pretrain_effect_flow_evm_bert.py \
  --config configs/sanity_pretrain_effect_flow_evm_bert_bjut_dive_train_balanced.yaml
