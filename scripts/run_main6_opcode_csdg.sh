#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
MODE="${1:-help}"
case "$MODE" in
  audit)
    python scripts/audit_main6_opcode_graph.py --config configs/train_main6_opcode_csdg.yaml
    ;;
  extract)
    python scripts/extract_evm_bert_chunk_features.py --config configs/extract_main6_opcode_csdg_sequence.yaml --splits train valid
    python scripts/extract_main6_opcode_graph_cache.py --config configs/train_main6_opcode_csdg.yaml --splits train valid
    python scripts/check_main6_opcode_graph_cache.py --config configs/train_main6_opcode_csdg.yaml
    ;;
  train)
    for variant in cfg_residual cfg_stack_residual cfg_stack_storage_residual graph_mlp_control; do
      torchrun --standalone --nproc_per_node=2 scripts/train_main6_opcode_csdg.py --config configs/train_main6_opcode_csdg.yaml --variant "$variant"
      python scripts/analyze_main6_opcode_csdg.py --config configs/train_main6_opcode_csdg.yaml --variant "$variant"
    done
    P95=$(python -c "import json; print(json.load(open('data/reports/main6_opcode_csdg/graph_audit.json'))['splits']['train']['basic_blocks_p95'])")
    BASELINE=$(python -c "import json; p=json.load(open('results/main6_random_090_mlm8/mlm8_slot3/checkpoint_summary.json')); print(p.get('valid_macro_f1', p.get('best_macro_f1', -1)))")
    STACK=$(python -c "import json; print(json.load(open('results/main6_opcode_csdg/cfg_stack_residual/valid_summary.json'))['valid_macro_f1'])")
    if python - "$P95" "$BASELINE" "$STACK" <<'PY'
import sys
p95, baseline, stack = map(float, sys.argv[1:])
raise SystemExit(0 if p95 > 256 and stack >= baseline else 1)
PY
    then
      torchrun --standalone --nproc_per_node=2 scripts/train_main6_opcode_csdg.py --config configs/train_main6_opcode_csdg.yaml --variant cfg_stack_sparse_topk
      python scripts/analyze_main6_opcode_csdg.py --config configs/train_main6_opcode_csdg.yaml --variant cfg_stack_sparse_topk
    fi
    ;;
  dynamic)
    for variant in cfg_stack_dynamic_gate cfg_stack_dynamic_gate_shuffled graph_mlp_dynamic_gate; do
      torchrun --standalone --nproc_per_node=2 scripts/train_main6_opcode_csdg.py --config configs/train_main6_opcode_csdg.yaml --variant "$variant"
      python scripts/analyze_main6_opcode_csdg.py --config configs/train_main6_opcode_csdg.yaml --variant "$variant"
      python scripts/diagnose_main6_opcode_csdg.py --config configs/train_main6_opcode_csdg.yaml --variant "$variant" --num-workers 0
    done
    ;;
  extract-local)
    python scripts/extract_main6_opcode_graph_cache.py --config configs/train_main6_opcode_csdg.yaml --variant cfg_stack_local_attention --splits train valid
    python scripts/check_main6_opcode_graph_cache.py --config configs/train_main6_opcode_csdg.yaml --variant cfg_stack_local_attention
    ;;
  extract-instruction-value)
    python scripts/extract_main6_opcode_graph_cache.py --config configs/train_main6_opcode_csdg.yaml --variant instruction_value_dynamic_gate --splits train valid
    python scripts/check_main6_opcode_graph_cache.py --config configs/train_main6_opcode_csdg.yaml --variant instruction_value_dynamic_gate
    ;;
  train-local)
    for variant in cfg_stack_local_attention instruction_value_dynamic_gate; do
      torchrun --standalone --nproc_per_node=2 scripts/train_main6_opcode_csdg.py --config configs/train_main6_opcode_csdg.yaml --variant "$variant"
      python scripts/analyze_main6_opcode_csdg.py --config configs/train_main6_opcode_csdg.yaml --variant "$variant"
      python scripts/diagnose_main6_opcode_csdg.py --config configs/train_main6_opcode_csdg.yaml --variant "$variant" --num-workers 0
    done
    ;;
  select)
    python scripts/select_main6_opcode_csdg.py --config configs/train_main6_opcode_csdg.yaml --require-ablation
    ;;
  final)
    test "${ALLOW_TEST:-0}" = "1"
    python - <<'PY'
import json
from pathlib import Path
selection = json.loads(Path("results/main6_opcode_csdg/selection.json").read_text(encoding="utf-8"))
if not selection.get("approved_for_single_test") or not selection.get("selected"):
    raise SystemExit("Validation selection did not approve a single test")
PY
    if test -e data/features/main6_opcode_csdg/graph/test.pt || test -e data/features/main6_opcode_csdg/sequence/test.pt; then
      echo "test cache already exists; refusing a second test" >&2
      exit 1
    fi
    ALLOW_TEST=1 python scripts/extract_evm_bert_chunk_features.py --config configs/extract_main6_opcode_csdg_sequence.yaml --splits test --allow-test-cache
    ALLOW_TEST=1 python scripts/extract_main6_opcode_graph_cache.py --config configs/train_main6_opcode_csdg.yaml --splits test --allow-test-cache
    python scripts/evaluate_main6_opcode_csdg.py --config configs/train_main6_opcode_csdg.yaml --split test
    ;;
  *)
    echo "usage: $0 {audit|extract|train|dynamic|extract-local|extract-instruction-value|train-local|select|final}" >&2
    exit 2
    ;;
esac
