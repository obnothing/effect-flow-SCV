#!/usr/bin/env bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
CONFIG="${CONFIG:-configs/vulprobe/v1.yaml}"
STAGE="${1:-audit}"
test "${ALLOW_TEST:-0}" != "1"

case "$STAGE" in
  audit)
    python scripts/audit_vulprobe.py --config "$CONFIG"
    ;;
  smoke)
    python scripts/smoke_vulprobe.py
    for variant in b0_shared_representation b1_shared_probe b2_label_probe; do
      python scripts/train_vulprobe.py --config "$CONFIG" --variant "$variant" --stage frozen --smoke
    done
    ;;
  core)
    for variant in b0_shared_representation b1_shared_probe b2_label_probe; do
      python scripts/train_vulprobe.py --config "$CONFIG" --variant "$variant" --stage frozen
      python scripts/train_vulprobe.py --config "$CONFIG" --variant "$variant" --stage joint
    done
    python scripts/analyze_vulprobe.py core --config "$CONFIG"
    ;;
  diagnostics)
    python scripts/analyze_vulprobe.py diagnostics --config "$CONFIG"
    ;;
  conditional-b3)
    python scripts/analyze_vulprobe.py core --config "$CONFIG"
    gate="$(python -c "import json; print(str(json.load(open('results/vulprobe_v1/main_results.json'))['phase_d_gate']['passed']).lower())")"
    if [[ "$gate" != "true" ]]; then
      echo "Phase D gate did not pass; B3 is not trained."
      exit 0
    fi
    python scripts/train_vulprobe.py --config "$CONFIG" --variant b3_probe_interaction --stage frozen
    python scripts/train_vulprobe.py --config "$CONFIG" --variant b3_probe_interaction --stage joint
    python scripts/analyze_vulprobe.py core --config "$CONFIG"
    python scripts/analyze_vulprobe.py diagnostics --config "$CONFIG"
    ;;
  *)
    echo "Usage: $0 {audit|smoke|core|diagnostics|conditional-b3}" >&2
    exit 2
    ;;
esac
