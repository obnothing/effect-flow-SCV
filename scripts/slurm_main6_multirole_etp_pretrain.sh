#!/usr/bin/env bash
#SBATCH --job-name=main6_multirole_etp_pretrain
#SBATCH --partition=gpu-a10
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/main6_multirole_etp_pretrain_%j.out
#SBATCH --error=logs/main6_multirole_etp_pretrain_%j.err

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

test -f data/processed/multirole_etp_pretrain/DIVE_main6_random_train/train_multirole_etp_chunks.jsonl
test -f data/processed/multirole_etp_pretrain/ethereum_19143/train_multirole_etp_chunks.jsonl

torchrun --standalone --nproc_per_node=2 src/pretrain_multirole_etp_evm_bert_main6.py \
  --config configs/pretrain_multirole_etp_main6_random.yaml

test -f checkpoints/pretrain_multirole_etp_main6_random/hf_model/config.json
test -f checkpoints/pretrain_multirole_etp_main6_random/etp_head.pt
test -f results/pretrain_multirole_etp_main6_random/pretraining_manifest.json
