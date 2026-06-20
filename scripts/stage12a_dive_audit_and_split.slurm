#!/usr/bin/env bash
#SBATCH --job-name=stage12a_dive
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/slurm_stage12a_dive_%j.out
#SBATCH --error=logs/slurm_stage12a_dive_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc
conda activate correlascan_a40
mkdir -p logs

python scripts/find_dive_files.py
python scripts/audit_dive_dataset.py
python scripts/build_dive_splits.py
python scripts/check_dive_evm_tokenizer_compatibility.py
python scripts/write_dive_stage12_plan.py
