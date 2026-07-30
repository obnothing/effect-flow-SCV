#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-prepare}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
CONFIG="configs/train_dive_source_main6_v2.yaml"
PRETRAIN_CONFIG="configs/pretrain_dive_source_main6_v2.yaml"
CACHE_DIR="data/features/dive_source_main6_v2_windows"
RESULT_ROOT="results/dive_source_main6_v2"
CANDIDATES=(source_seq_coverage source_seq_multislot_mil source_seq_multislot_contrast source_seq_multislot_full)

case "$STAGE" in
  prepare)
    test ! -e "$CACHE_DIR/test.pt"
    python scripts/cache_solidity_graph_units.py --config "$CONFIG" --tokenizer-path models/graphcodebert-base --splits train valid
    python - <<'PY'
import json
from pathlib import Path
for split in ("train", "valid"):
    report = json.loads((Path("data/reports/dive_source_main6_v2") / f"{split}_window_coverage.json").read_text())
    assert report["mandatory_windows_preserved"], report
    if split == "train":
        assert report["within_frozen_budget"] >= 0.95, report
PY
    ;;
  pretrain)
    test -f "$CACHE_DIR/train.pt"
    test ! -e "$CACHE_DIR/test.pt"
    torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" src/pretrain_solidity_graphcodebert.py --config "$PRETRAIN_CONFIG"
    ;;
  train)
    test -f checkpoints/dive_source_main6_v2/dapt/hf_model/config.json
    test ! -e "$CACHE_DIR/test.pt"
    for variant in "${CANDIDATES[@]}"; do
      torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" src/train_solidity_graphcodebert.py --config "$CONFIG" --variant "$variant"
    done
    ;;
  select)
    python scripts/select_dive_source_main6_candidate.py --result-root "$RESULT_ROOT" --checkpoint-root checkpoints/dive_source_main6_v2 --output "$RESULT_ROOT/validation_selection.json" --candidates "${CANDIDATES[@]}"
    selection="$RESULT_ROOT/validation_selection.json"
    variant="$(python -c "import json; print(json.load(open('$selection'))['selected']['variant'])")"
    checkpoint="$(python -c "import json; print(json.load(open('$selection'))['selected']['checkpoint'])")"
    python src/evaluate_solidity_graphcodebert.py --config "$CONFIG" --variant "$variant" --checkpoint "$checkpoint" --split valid --threshold-search
    ;;
  final)
    echo "Final test remains intentionally unavailable during the coverage-baseline phase." >&2
    exit 2
    ;;
  *) echo "Usage: $0 {prepare|pretrain|train|select}" >&2; exit 2 ;;
esac
