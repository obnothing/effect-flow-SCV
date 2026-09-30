#!/usr/bin/env bash
#SBATCH --job-name=p11-dive8
#SBATCH --partition=gpu-l20
#SBATCH --gres=gpu:l20:1
#SBATCH --cpus-per-task=2
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/oywater/workspace/smart_1/logs/p11_dive8_%j.out
#SBATCH --error=/home/oywater/workspace/smart_1/logs/p11_dive8_%j.err

set -eo pipefail
cd /home/oywater/workspace/smart_1
source "${HOME}/.bashrc"
conda activate py310
set -u
variant="${1:?Specify original or e3}"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1
python -m py_compile scripts/train_p11_dive8.py
python scripts/train_p11_dive8.py smoke --variant "$variant"
python scripts/train_p11_dive8.py train --variant "$variant"
