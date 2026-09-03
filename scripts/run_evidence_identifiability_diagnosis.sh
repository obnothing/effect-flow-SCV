#!/usr/bin/env bash
set -eo pipefail

CONFIG="${CONFIG:-configs/evidence_identifiability_diagnosis.yaml}"
test "${ALLOW_TEST:-0}" != "1"
python scripts/diagnose_evidence_identifiability.py --config "$CONFIG"
