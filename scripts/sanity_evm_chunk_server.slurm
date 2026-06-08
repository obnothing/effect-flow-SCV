#!/usr/bin/env bash
#SBATCH --job-name=evm_chunk_sanity
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --output=logs/slurm_evm_chunk_sanity_%j.out
#SBATCH --error=logs/slurm_evm_chunk_sanity_%j.err

set -e

cd $SLURM_SUBMIT_DIR

source ~/.bashrc
conda activate correlascan

mkdir -p logs

bash scripts/sanity_evm_chunk_server.sh configs/sanity_evm_chunk_server.yaml
