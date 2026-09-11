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

python scripts/audit_e5_tdvp.py

BASE_CHECKPOINT="checkpoints/light_label/b2_label_attention/full/best.pt"
for variant in d1_param_control d2_global_mean d3_shuffled_dictionary d4_tdvp
do
    echo "[e5-control] ${variant} seed=42"
    python scripts/train_e5_tdvp.py \
        --config "configs/e5_tdvp/${variant}.yaml" \
        --variant "${variant}" \
        --seed 42 \
        --init-checkpoint "${BASE_CHECKPOINT}" \
        2>&1 | tee -a logs/e5_tdvp_control_gate.log
done

python scripts/analyze_e5_tdvp.py

