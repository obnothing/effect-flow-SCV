#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-audit}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_CONFIG="configs/train_main6_random_090.yaml"
PRETRAIN_CONFIG="configs/pretrain_multirole_etp_main6_random.yaml"
EXTRACT_CONFIG="configs/extract_main6_multirole_etp.yaml"

run_python() { "$PYTHON_BIN" "$@"; }

case "$STAGE" in
  audit)
    run_python scripts/audit_main6_random_pretrain_protocol.py
    ;;
  corpus)
    if [[ ! -f data/processed/multirole_etp_pretrain/DIVE_main6_random_train/train_multirole_etp_chunks.jsonl ]]; then
      run_python scripts/build_multirole_etp_pretraining_corpus.py \
        --dataset DIVE_main6_random_train \
        --train_path data/processed/DIVE_main6_random_split/train.jsonl \
        --output_dir data/processed/multirole_etp_pretrain/DIVE_main6_random_train \
        --vocab_path checkpoints/pretrain_evm_bert_19143_base_continue/evm_vocab.json
    fi
    if [[ ! -f data/processed/multirole_etp_pretrain/ethereum_19143/train_multirole_etp_chunks.jsonl ]]; then
      run_python scripts/build_multirole_etp_pretraining_corpus.py \
        --dataset ethereum_19143 \
        --train_path data/processed/ethereum_public_pretrain_19143_unique_runtime/runtime_opcode.jsonl \
        --output_dir data/processed/multirole_etp_pretrain/ethereum_19143 \
        --vocab_path checkpoints/pretrain_evm_bert_19143_base_continue/evm_vocab.json \
        --max_chunks_per_contract 32
    fi
    ;;
  pretrain)
    run_python src/pretrain_multirole_etp_evm_bert_main6.py --config "$PRETRAIN_CONFIG"
    ;;
  extract)
    run_python scripts/extract_evm_bert_chunk_features.py --config configs/extract_main6_random_mlm_control_trainvalid.yaml --splits train valid
    run_python scripts/extract_multirole_etp_token_features.py --config "$EXTRACT_CONFIG" --splits train valid
    ;;
  validate)
    run_python scripts/check_evm_bert_chunk_features.py --feature_dir data/features/main6_random_mlm_control_trainvalid --data_dir data/processed/DIVE_main6_random_split --expected_max_chunks 64 --expected_feature_dim 768 --expected_num_labels 6 --output data/reports/check_main6_random_mlm_control_trainvalid.txt --splits train valid
    run_python scripts/check_multirole_etp_token_features.py --feature_dir data/features/main6_random_multirole_etp --token_semantic_dir data/features/main6_random_multirole_etp_tokens --data_dir data/processed/DIVE_main6_random_split --splits train valid
    ;;
  train)
    for variant in mlm_label_mil etp_encoder_mil etp_concat_mil ld_etpca ld_etpca_sep; do
      run_python src/train_chunk_mil.py --config "$TRAIN_CONFIG" --variant "$variant"
    done
    ;;
  analyze)
    for variant in ld_etpca ld_etpca_sep; do
      checkpoint="checkpoints/main6_random_090/$variant/best_macro_f1.pt"
      run_python scripts/analyze_main6_label_interference.py --config "$TRAIN_CONFIG" --variant "$variant" --checkpoint "$checkpoint" --output "results/main6_random_090/$variant/valid_interference_analysis.json"
    done
    ;;
  select)
    run_python scripts/select_main6_random_090_candidate.py
    selection="results/main6_random_090/validation_selection.json"
    variant="$(run_python -c "import json; print(json.load(open('$selection'))['selected']['variant'])")"
    checkpoint="$(run_python -c "import json; print(json.load(open('$selection'))['selected']['checkpoint'])")"
    run_python src/evaluate_chunk_mil.py --config "$TRAIN_CONFIG" --variant "$variant" --checkpoint "$checkpoint" --split valid --threshold_search global_and_per_label
    ;;
  final)
    if [[ "${ALLOW_TEST:-0}" != "1" ]]; then
      echo "Final test is blocked. Review validation selection, then use ALLOW_TEST=1." >&2
      exit 2
    fi
    selection="results/main6_random_090/validation_selection.json"
    test -f "$selection"
    approved="$(run_python -c "import json; print(json.load(open('$selection'))['approved_for_single_test'])")"
    [[ "$approved" == "True" ]]
    variant="$(run_python -c "import json; print(json.load(open('$selection'))['selected']['variant'])")"
    checkpoint="$(run_python -c "import json; print(json.load(open('$selection'))['selected']['checkpoint'])")"
    result_dir="results/main6_random_090/$variant"
    if find "$result_dir" -maxdepth 1 -type f -name 'test_*' -print -quit | grep -q .; then
      echo "Final test is single-use; test artifacts already exist in $result_dir." >&2
      exit 2
    fi
    run_python scripts/extract_multirole_etp_token_features.py --config "$EXTRACT_CONFIG" --splits test --allow-test-cache
    run_python scripts/check_multirole_etp_token_features.py --feature_dir data/features/main6_random_multirole_etp --token_semantic_dir data/features/main6_random_multirole_etp_tokens --data_dir data/processed/DIVE_main6_random_split --splits test
    test -f "$result_dir/per_label_thresholds_valid.json"
    test -f "$result_dir/valid_global_threshold_scan.json"
    run_python src/evaluate_chunk_mil.py --config "$TRAIN_CONFIG" --variant "$variant" --checkpoint "$checkpoint" --split test --threshold_file "$result_dir/per_label_thresholds_valid.json" --global_threshold_file "$result_dir/valid_global_threshold_scan.json" --save_predictions
    ;;
  *)
    echo "Usage: $0 {audit|corpus|pretrain|extract|validate|train|analyze|select|final}" >&2
    exit 2
    ;;
esac
