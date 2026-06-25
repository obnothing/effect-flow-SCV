#!/usr/bin/env bash
#SBATCH --job-name=bjut_ef_first512
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --output=logs/slurm_bjut_ef_first512_%j.out
#SBATCH --error=logs/slurm_bjut_ef_first512_%j.err

set -eo pipefail

cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc
conda activate correlascan_a40
mkdir -p logs

CONFIG="configs/train_bjut_effect_flow_first512_weighted_server.yaml"

python scripts/check_evm_bert_pretrained_model.py --config "$CONFIG"

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=2 \
  src/train.py \
  --config "$CONFIG"

