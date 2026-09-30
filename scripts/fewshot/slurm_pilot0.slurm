#!/usr/bin/env bash
#SBATCH --job-name=fewshot-pilot0
#SBATCH --partition=gpu-l20
#SBATCH --gres=gpu:l20:1
#SBATCH --cpus-per-task=2
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/oywater/workspace/smart_1/logs/fewshot_pilot0_%j.out
#SBATCH --error=/home/oywater/workspace/smart_1/logs/fewshot_pilot0_%j.err

set -eo pipefail
cd /home/oywater/workspace/smart_1
source "${HOME}/.bashrc"
conda activate py310
set -u

mkdir -p logs
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export PYTHONUNBUFFERED=1

python -m py_compile \
  scripts/fewshot/build_lovo_splits.py \
  scripts/fewshot/audit_architecture.py \
  scripts/fewshot/prepare_lovo_cache.py \
  scripts/fewshot/preflight_cuda.py \
  scripts/fewshot/pilot_core.py \
  scripts/fewshot/train_lovo_base.py \
  scripts/fewshot/run_pilot.py \
  scripts/fewshot/test_protocol.py

python - <<'PY'
import torch
print(f"torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}", flush=True)
if not torch.cuda.is_available():
    raise RuntimeError("Slurm job did not receive a CUDA GPU")
print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)
PY

python scripts/fewshot/run_pilot.py --smoke
python scripts/fewshot/run_pilot.py --full
