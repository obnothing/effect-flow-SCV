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

python scripts/build_e5_grouped_dev_split.py
python scripts/prepare_light_label_data.py --config configs/e5_tdvp/grouped_base.yaml
python scripts/resolve_light_label_runtime.py --config configs/e5_tdvp/grouped_base.yaml

for seed in 42 43 44
do
    echo "[e5-grouped] d0_b2 seed=${seed}"
    python scripts/train_e5_tdvp.py --config configs/e5_tdvp/d0_b2_grouped.yaml --variant d0_b2 --seed "${seed}" 2>&1 | tee -a logs/e5_tdvp_grouped.log
    echo "[e5-grouped] d4_tdvp seed=${seed}"
    python scripts/train_e5_tdvp.py \
        --config configs/e5_tdvp/d4_tdvp_grouped.yaml \
        --variant d4_tdvp \
        --seed "${seed}" \
        --init-checkpoint "checkpoints/e5_tdvp/grouped/d0_b2/seed_${seed}/best.pt" \
        2>&1 | tee -a logs/e5_tdvp_grouped.log
done

python scripts/analyze_e5_grouped.py

