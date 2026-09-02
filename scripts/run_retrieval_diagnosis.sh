#!/usr/bin/env bash
set -eo pipefail

CONFIG="${CONFIG:-configs/retrieval_diagnosis.yaml}"
test "${ALLOW_TEST:-0}" != "1"
test ! -e results/retrieval_validation/diagnosis/test_queries.pt

python scripts/diagnose_retrieval.py --config "$CONFIG"
