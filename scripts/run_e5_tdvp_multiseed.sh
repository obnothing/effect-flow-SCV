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

python -c 'import json, pathlib; p=pathlib.Path("results/e5_tdvp/main_results.json"); x=json.loads(p.read_text()); assert x["control_gate"]["proceed_multiseed"], "control gate is false; do not start multi-seed"'

for seed in 43 44
do
    echo "[e5-multiseed] d0_b2 seed=${seed}"
    python scripts/train_e5_tdvp.py --config configs/e5_tdvp/d0_b2.yaml --variant d0_b2 --seed "${seed}" 2>&1 | tee -a logs/e5_tdvp_multiseed.log
    for variant in d1_param_control d2_global_mean d3_shuffled_dictionary d4_tdvp
    do
        echo "[e5-multiseed] ${variant} seed=${seed}"
        python scripts/train_e5_tdvp.py \
            --config "configs/e5_tdvp/${variant}.yaml" \
            --variant "${variant}" \
            --seed "${seed}" \
            --init-checkpoint "checkpoints/e5_tdvp/random/d0_b2/seed_${seed}/best.pt" \
            2>&1 | tee -a logs/e5_tdvp_multiseed.log
    done
done

python scripts/analyze_e5_tdvp.py

