#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
MODE="${1:-help}"
case "$MODE" in
  train)
    for variant in view_cls view_mean view_max view_cls_mean view_mean_max view_cls_mean_max view_all8; do
      torchrun --standalone --nproc_per_node=2 scripts/train_main6_opcode_lse_mil.py \
        --config configs/train_main6_opcode_lse_mil.yaml --variant "$variant"
    done
    ;;
  select)
    python scripts/select_main6_opcode_lse_mil.py \
      --config configs/train_main6_opcode_lse_mil.yaml
    ;;
  *)
    echo "usage: $0 {train|select}" >&2
    exit 2
    ;;
esac
