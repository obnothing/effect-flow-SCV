#!/usr/bin/env bash
#SBATCH --job-name=codebert_mlsmote_train
#SBATCH --partition=gpu-a10
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --output=logs/slurm_codebert_mlsmote_train_%j.out
#SBATCH --error=logs/slurm_codebert_mlsmote_train_%j.err

set -e

cd $SLURM_SUBMIT_DIR

source ~/.bashrc
conda activate correlascan_a40

mkdir -p logs

bash scripts/train_mlsmote_codebert_server.sh configs/train_mlsmote_codebert_server.yaml
