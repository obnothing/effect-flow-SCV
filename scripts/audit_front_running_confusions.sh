#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python scripts/audit_front_running_confusions.py \
  --data data/processed/DIVE_random_split/test.jsonl \
  --predictions results/dive_side_scale_search/train_dive_side_scale_150_ep50/test_predictions_per_label.jsonl \
  --top_chunks results/dive_side_scale_search/train_dive_side_scale_150_ep50/top_chunks_test_calibrated.jsonl \
  --pattern_config configs/effect_flow_efpp_conservative_22.json \
  --semantic_cache data/features/effect_flow_semantics/dive_random_stride256_max64/test.pt \
  --chunk_size 512 \
  --chunk_stride 256 \
  --max_chunks 64 \
  --output_prefix data/reports/front_running_confusion_audit
