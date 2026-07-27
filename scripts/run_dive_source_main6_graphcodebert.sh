#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-audit}"
CONFIG="configs/train_dive_source_main6_graphcodebert.yaml"
PRETRAIN_CONFIG="configs/pretrain_dive_source_main6_graphcodebert.yaml"
BUILD_CONFIG="configs/build_dive_source_main6.yaml"
BASE_MODEL="models/graphcodebert-base"
RESULT_ROOT="results/dive_source_main6"

case "$STAGE" in
  audit)
    python scripts/build_dive_source_main6.py --config "$BUILD_CONFIG"
    ;;
  cache)
    test ! -e data/features/dive_source_main6_graphcodebert/test.pt
    python scripts/cache_solidity_graph_units.py --config "$CONFIG" --tokenizer-path "$BASE_MODEL" --splits train valid
    ;;
  pretrain)
    test -f data/features/dive_source_main6_graphcodebert/train.pt
    test ! -e data/features/dive_source_main6_graphcodebert/test.pt
    torchrun --standalone --nproc_per_node=2 src/pretrain_solidity_graphcodebert.py --config "$PRETRAIN_CONFIG"
    ;;
  train)
    test -f checkpoints/dive_source_main6/dapt/hf_model/config.json
    for variant in source_seq_mean source_graph_mean source_graph_mil source_graph_mil_contrast source_graph_mil_full; do
      torchrun --standalone --nproc_per_node=2 src/train_solidity_graphcodebert.py --config "$CONFIG" --variant "$variant"
    done
    ;;
  select)
    python scripts/select_dive_source_main6_candidate.py
    selection="$RESULT_ROOT/validation_selection.json"
    variant="$(python -c "import json; print(json.load(open('$selection'))['selected']['variant'])")"
    checkpoint="$(python -c "import json; print(json.load(open('$selection'))['selected']['checkpoint'])")"
    python src/evaluate_solidity_graphcodebert.py --config "$CONFIG" --variant "$variant" --checkpoint "$checkpoint" --split valid --threshold-search
    ;;
  final)
    [[ "${ALLOW_TEST:-0}" == "1" ]] || { echo "Final test is locked; set ALLOW_TEST=1 after review." >&2; exit 2; }
    selection="$RESULT_ROOT/validation_selection.json"; test -f "$selection"
    [[ "$(python -c "import json; print(json.load(open('$selection'))['approved_for_single_test'])")" == "True" ]]
    variant="$(python -c "import json; print(json.load(open('$selection'))['selected']['variant'])")"
    checkpoint="$(python -c "import json; print(json.load(open('$selection'))['selected']['checkpoint'])")"
    result_dir="$RESULT_ROOT/$variant"
    if find "$result_dir" -maxdepth 1 -name 'test_*' -print -quit | grep -q .; then echo "Final test artifacts already exist." >&2; exit 2; fi
    python scripts/cache_solidity_graph_units.py --config "$CONFIG" --splits test --allow-test-cache
    python src/evaluate_solidity_graphcodebert.py --config "$CONFIG" --variant "$variant" --checkpoint "$checkpoint" --split test --threshold-file "$result_dir/per_label_thresholds_valid.json" --save-predictions
    ;;
  *) echo "Usage: $0 {audit|cache|pretrain|train|select|final}" >&2; exit 2 ;;
esac
