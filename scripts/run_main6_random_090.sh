#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-audit}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_CONFIG="configs/train_main6_random_090.yaml"
PRETRAIN_CONFIG="configs/pretrain_evm_bert_main6_random_train_mlm8.yaml"
EXTRACT_CONFIG="configs/extract_main6_random_mlm8_trainvalid.yaml"
FEATURE_DIR="data/features/main6_random_mlm8_trainvalid"
RESULT_ROOT="results/main6_random_090_mlm8"

run_python() { "$PYTHON_BIN" "$@"; }

case "$STAGE" in
  audit)
    run_python scripts/audit_main6_random_pretrain_protocol.py
    ;;
  pretrain)
    run_python src/pretrain_evm_bert.py --config "$PRETRAIN_CONFIG"
    ;;
  extract)
    run_python scripts/extract_evm_bert_chunk_features.py --config "$EXTRACT_CONFIG" --splits train valid
    ;;
  validate)
    run_python scripts/check_evm_bert_chunk_features.py \
      --feature_dir "$FEATURE_DIR" \
      --data_dir data/processed/DIVE_main6_random_split \
      --expected_max_chunks 64 --expected_num_views 8 \
      --expected_feature_dim 768 --expected_num_labels 6 \
      --output data/reports/check_main6_random_mlm8_trainvalid.txt \
      --splits train valid
    ;;
  train)
    for variant in mlm8_slot1 mlm8_slot2 mlm8_slot3 mlm8_slot4; do
      run_python src/train_chunk_mil.py --config "$TRAIN_CONFIG" --variant "$variant"
    done
    ;;
  select)
    run_python scripts/select_main6_random_090_candidate.py
    selection="$RESULT_ROOT/validation_selection.json"
    variant="$(run_python -c "import json; print(json.load(open('$selection'))['selected']['variant'])")"
    checkpoint="$(run_python -c "import json; print(json.load(open('$selection'))['selected']['checkpoint'])")"
    run_python src/evaluate_chunk_mil.py --config "$TRAIN_CONFIG" --variant "$variant" --checkpoint "$checkpoint" --split valid --threshold_search global_and_per_label
    ;;
  final)
    if [[ "${ALLOW_TEST:-0}" != "1" ]]; then
      echo "Final test is blocked. Review validation selection, then use ALLOW_TEST=1." >&2
      exit 2
    fi
    selection="$RESULT_ROOT/validation_selection.json"
    test -f "$selection"
    approved="$(run_python -c "import json; print(json.load(open('$selection'))['approved_for_single_test'])")"
    [[ "$approved" == "True" ]]
    variant="$(run_python -c "import json; print(json.load(open('$selection'))['selected']['variant'])")"
    checkpoint="$(run_python -c "import json; print(json.load(open('$selection'))['selected']['checkpoint'])")"
    result_dir="$RESULT_ROOT/$variant"
    if find "$result_dir" -maxdepth 1 -type f -name 'test_*' -print -quit | grep -q .; then
      echo "Final test is single-use; test artifacts already exist in $result_dir." >&2
      exit 2
    fi
    run_python scripts/extract_evm_bert_chunk_features.py --config "$EXTRACT_CONFIG" --splits test --allow-test-cache
    run_python scripts/check_evm_bert_chunk_features.py \
      --feature_dir "$FEATURE_DIR" \
      --data_dir data/processed/DIVE_main6_random_split \
      --expected_max_chunks 64 --expected_num_views 8 \
      --expected_feature_dim 768 --expected_num_labels 6 \
      --output data/reports/check_main6_random_mlm8_test.txt \
      --splits test
    test -f "$result_dir/per_label_thresholds_valid.json"
    test -f "$result_dir/valid_global_threshold_scan.json"
    run_python src/evaluate_chunk_mil.py --config "$TRAIN_CONFIG" --variant "$variant" --checkpoint "$checkpoint" --split test --threshold_file "$result_dir/per_label_thresholds_valid.json" --global_threshold_file "$result_dir/valid_global_threshold_scan.json" --save_predictions
    ;;
  *)
    echo "Usage: $0 {audit|pretrain|extract|validate|train|select|final}" >&2
    exit 2
    ;;
esac
