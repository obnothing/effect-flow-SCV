#!/usr/bin/env bash
#SBATCH --job-name=source_main6_train
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/source_main6_train_%j.out
#SBATCH --error=logs/source_main6_train_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
source ~/.bashrc
conda activate correlascan_a40
mkdir -p logs
test -f checkpoints/dive_source_main6/dapt/hf_model/config.json
bash scripts/run_dive_source_main6_graphcodebert.sh train
