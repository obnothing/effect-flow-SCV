#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-audit}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_CONFIG="configs/train_main6_random_090.yaml"
PRETRAIN_CONFIG="configs/pretrain_effect_flow_evm_bert_main6_random_train_new.yaml"

run_python() {
  "$PYTHON_BIN" "$@"
}

case "$STAGE" in
  audit)
    run_python scripts/audit_main6_random_pretrain_protocol.py
    ;;
  corpus)
    run_python scripts/build_effect_flow_pretraining_corpus.py \
      --dataset DIVE_main6 \
      --train_path data/processed/DIVE_main6_random_split/train.jsonl \
      --output_dir data/processed/effect_flow_pretrain/DIVE_main6_random_train_new \
      --vocab_path checkpoints/pretrain_evm_bert_19143_base_continue/evm_vocab.json \
      --template_path configs/vulnerability_templates_dive_main6_new.yaml \
      --disable_downstream_labels \
      --report_prefix effect_flow_corpus_build_DIVE_main6_random_train_new
    ;;
  pretrain)
    run_python src/pretrain_effect_flow_evm_bert_main6.py --config "$PRETRAIN_CONFIG"
    ;;
  extract)
    run_python scripts/extract_evm_bert_chunk_features.py --config configs/extract_main6_random_mlm_control_new.yaml
    run_python scripts/extract_evm_bert_chunk_features.py --config configs/extract_main6_random_effectflow_new.yaml
    run_python scripts/extract_effect_flow_semantic_features.py --config configs/extract_main6_random_semantics_new.yaml
    ;;
  validate)
    run_python scripts/check_evm_bert_chunk_features.py --feature_dir data/features/main6_random_mlm_control_new --data_dir data/processed/DIVE_main6_random_split --expected_max_chunks 64 --expected_feature_dim 768 --expected_num_labels 6 --output data/reports/check_main6_random_mlm_control_new.txt
    run_python scripts/check_evm_bert_chunk_features.py --feature_dir data/features/main6_random_effectflow_new --data_dir data/processed/DIVE_main6_random_split --expected_max_chunks 64 --expected_feature_dim 768 --expected_num_labels 6 --output data/reports/check_main6_random_effectflow_new.txt
    run_python scripts/check_effect_flow_semantic_features.py --semantic_dir data/features/main6_random_effectflow_semantics_new --feature_dir data/features/main6_random_effectflow_new --data_dir data/processed/DIVE_main6_random_split --expected_max_chunks 64 --expected_efpp_dim 22 --expected_etp_dim 16 --expected_relation_dim 6 --expected_global_template_dim 17 --expected_num_labels 6 --output data/reports/check_main6_random_effectflow_semantics_new.txt
    ;;
  train)
    for variant in mlm_control effectflow_control effectflow_multiscale; do
      run_python src/train_chunk_mil.py --config "$TRAIN_CONFIG" --variant "$variant"
    done
    ;;
  select)
    run_python scripts/select_main6_random_090_candidate.py
    ;;
  final)
    if [[ "${ALLOW_TEST:-0}" != "1" ]]; then
      echo "Final test is blocked. Review validation selection, then use ALLOW_TEST=1." >&2
      exit 2
    fi
    selection="results/main6_random_090/validation_selection.json"
    variant="$(run_python -c "import json; print(json.load(open('$selection'))['selected']['variant'])")"
    checkpoint="$(run_python -c "import json; print(json.load(open('$selection'))['selected']['checkpoint'])")"
    run_python src/evaluate_chunk_mil.py --config "$TRAIN_CONFIG" --variant "$variant" --checkpoint "$checkpoint" --split valid --threshold_search global_and_per_label
    run_python src/evaluate_chunk_mil.py --config "$TRAIN_CONFIG" --variant "$variant" --checkpoint "$checkpoint" --split test --threshold_file "results/main6_random_090/$variant/per_label_thresholds_valid.json" --global_threshold_file "results/main6_random_090/$variant/valid_global_threshold_scan.json" --save_predictions
    ;;
  *)
    echo "Usage: $0 {audit|corpus|pretrain|extract|validate|train|select|final}" >&2
    exit 2
    ;;
esac
