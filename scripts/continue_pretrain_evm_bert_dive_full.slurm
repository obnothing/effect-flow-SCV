#!/usr/bin/env bash
#SBATCH --job-name=evmbert_dive_full_mlm
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/slurm_evmbert_dive_full_mlm_%j.out
#SBATCH --error=logs/slurm_evmbert_dive_full_mlm_%j.err

set -eo pipefail

cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc
conda activate correlascan_a40
mkdir -p logs

BASE_MODEL="checkpoints/pretrain_evm_bert_base_full_bjut/hf_model"
for required in \
  "$BASE_MODEL/config.json" \
  "data/processed/BJUT_SC01/evm_vocab.json" \
  "data/processed/DIVE/train.jsonl" \
  "data/processed/DIVE/valid.jsonl" \
  "data/processed/DIVE/test.jsonl"; do
  if [[ ! -f "$required" ]]; then
    echo "[ERROR] missing required file: $required"
    echo "[ERROR] Run Stage 12A first. Continued MLM was not started."
    exit 1
  fi
done
if [[ ! -f "$BASE_MODEL/model.safetensors" && ! -f "$BASE_MODEL/pytorch_model.bin" ]]; then
  echo "[ERROR] BJUT EVM-BERT weights are missing from $BASE_MODEL"
  exit 1
fi

python scripts/check_dive_evm_tokenizer_compatibility.py \
  --vocab_path data/processed/BJUT_SC01/evm_vocab.json \
  --data_dir data/processed/DIVE \
  --max_chunks_per_contract 16

python scripts/build_dive_continued_mlm_corpus.py \
  --mode full_corpus \
  --max_len 512 \
  --chunk_stride 510 \
  --max_chunks_per_contract 16

export OMP_NUM_THREADS=2
torchrun --standalone --nproc_per_node=2 src/pretrain_evm_bert.py \
  --config configs/continue_pretrain_evm_bert_dive_full.yaml
