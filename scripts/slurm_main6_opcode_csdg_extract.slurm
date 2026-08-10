#!/usr/bin/env bash
#SBATCH --job-name=main6_csdg_extract
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/main6_csdg_extract_%j.out
#SBATCH --error=logs/main6_csdg_extract_%j.err

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc
conda activate correlascan_a40
set -u
mkdir -p logs
python scripts/extract_evm_bert_chunk_features.py \
  --config configs/extract_main6_opcode_csdg_sequence.yaml \
  --splits train valid
python scripts/extract_main6_opcode_graph_cache.py \
  --config configs/train_main6_opcode_csdg.yaml \
  --splits train valid
python scripts/check_main6_opcode_graph_cache.py \
  --config configs/train_main6_opcode_csdg.yaml
