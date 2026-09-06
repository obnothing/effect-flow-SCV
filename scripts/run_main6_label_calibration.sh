#!/usr/bin/env bash
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source "${HOME}/.bashrc"
conda activate correlascan_a40
set -u
test "${ALLOW_TEST:-0}" != "1"

# Only loss weights change; architecture, split, cache, and thresholds stay fixed.
python src/train_chunk_mil.py \
  --config configs/train_main6_random_090.yaml \
  --variant mlm8_slot3 \
  --override experiment_name=main6_m0_label_calibration \
  --override checkpoint_dir=checkpoints/main6_random_090_mlm8/label_calibration \
  --override result_dir=results/main6_random_090_mlm8/label_calibration \
  --override report_dir=data/reports/main6_random_090_mlm8_label_calibration \
  --override epochs=30 \
  --override early_stopping_patience=5 \
  --override recognition_pos_weight_multipliers=[1.0,0.85,1.0,1.0,1.35,1.15]
