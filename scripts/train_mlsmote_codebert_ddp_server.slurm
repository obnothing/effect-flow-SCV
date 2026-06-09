#!/usr/bin/env bash
#SBATCH --job-name=codebert_mlsmote_ddp
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --output=logs/slurm_codebert_mlsmote_ddp_%j.out
#SBATCH --error=logs/slurm_codebert_mlsmote_ddp_%j.err

set -e

cd $SLURM_SUBMIT_DIR

source ~/.bashrc
conda activate correlascan_a40

mkdir -p logs

export NPROC_PER_NODE=2

bash scripts/train_mlsmote_codebert_ddp_server.sh configs/train_mlsmote_codebert_server.yaml
