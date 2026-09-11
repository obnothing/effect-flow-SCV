#!/usr/bin/env bash
set -eo pipefail

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "${HOME}/.bashrc"
conda activate pytorch-2.1.1
set -u

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
mkdir -p logs

for config in configs/light_label/length_test/b2_len12288.yaml configs/light_label/length_test/b2_len16384.yaml
do
    echo "[length] preparing ${config}"
    python scripts/prepare_light_label_data.py --config "${config}"
    python scripts/resolve_light_label_runtime.py --config "${config}"
    echo "[length] training ${config}"
    python src/train_light_label_model.py --config "${config}" 2>&1 | tee -a logs/light_length_test.log
done

python scripts/analyze_length_test.py

