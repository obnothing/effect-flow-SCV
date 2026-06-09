#!/usr/bin/env bash
#SBATCH --job-name=evm_chunk_train
#SBATCH --partition=gpu-a30
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --output=logs/slurm_evm_chunk_train_%j.out
#SBATCH --error=logs/slurm_evm_chunk_train_%j.err

set -e

cd $SLURM_SUBMIT_DIR

source ~/.bashrc
conda activate correlascan_a40

mkdir -p logs

bash scripts/train_evm_chunk_server.sh configs/train_evm_chunk_server.yaml
