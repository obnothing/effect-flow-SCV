#!/usr/bin/env bash
#SBATCH --job-name=correlascan_unfrozen
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --output=logs/slurm_unfrozen_%j.out
#SBATCH --error=logs/slurm_unfrozen_%j.err

set -e

cd $SLURM_SUBMIT_DIR

source ~/.bashrc
conda activate correlascan

bash scripts/sanity_train_server.sh configs/sanity_mlsmote_unfrozen_server.yaml
