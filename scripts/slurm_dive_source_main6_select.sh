#!/usr/bin/env bash
#SBATCH --job-name=source_main6_select
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/source_main6_select_%j.out
#SBATCH --error=logs/source_main6_select_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
source ~/.bashrc
conda activate correlascan_a40
mkdir -p logs
test ! -e data/features/dive_source_main6_graphcodebert/test.pt
bash scripts/run_dive_source_main6_graphcodebert.sh select
