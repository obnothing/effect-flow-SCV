#!/usr/bin/env bash
#SBATCH --job-name=source_main6_dapt
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/source_main6_dapt_%j.out
#SBATCH --error=logs/source_main6_dapt_%j.err
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
source ~/.bashrc
set -u
conda activate correlascan_a40
mkdir -p logs
test -f data/features/dive_source_main6_graphcodebert/train.pt
test ! -e data/features/dive_source_main6_graphcodebert/test.pt
torchrun --standalone --nproc_per_node=2 src/pretrain_solidity_graphcodebert.py --config configs/pretrain_dive_source_main6_graphcodebert.yaml
