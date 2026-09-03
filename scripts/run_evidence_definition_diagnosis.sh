#!/usr/bin/env bash
set -eo pipefail

CONFIG="${CONFIG:-configs/evidence_definition_diagnosis.yaml}"
test "${ALLOW_TEST:-0}" != "1"
python scripts/diagnose_evidence_definition.py --config "$CONFIG"
