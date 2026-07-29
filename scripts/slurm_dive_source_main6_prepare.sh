#!/usr/bin/env bash
#SBATCH --job-name=source_main6_prepare
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/source_main6_prepare_%j.out
#SBATCH --error=logs/source_main6_prepare_%j.err
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
source ~/.bashrc
set -u
conda activate correlascan_a40
mkdir -p logs
python scripts/build_dive_source_main6.py --config configs/build_dive_source_main6.yaml
python scripts/cache_solidity_graph_units.py --config configs/train_dive_source_main6_graphcodebert.yaml --tokenizer-path models/graphcodebert-base --splits train valid
