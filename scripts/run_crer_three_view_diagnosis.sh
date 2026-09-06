#!/usr/bin/env bash
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source "${HOME}/.bashrc"
conda activate correlascan_a40
set -u
test "${ALLOW_TEST:-0}" != "1"

for variant in A1_full A2_no_counterfactual; do
  checkpoint="results/crer_main6_three_view/${variant}/seed_42/best.pt"
  test -f "$checkpoint"
  python scripts/diagnose_crer_three_view.py \
    --config configs/train_crer_three_view.yaml \
    --checkpoint "$checkpoint" \
    --output "results/crer_main6_three_view/${variant}/seed_42/diagnosis.json"
done
