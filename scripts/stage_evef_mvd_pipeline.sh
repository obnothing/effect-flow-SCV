#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

MODE="all"
RESUME_FROM=""
FORCE_CORPUS=0
SKIP_PRETRAIN_SANITY=0
SKIP_PRETRAIN_FULL=0
DATASET_SCOPE="both"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
CONDA_ENV="${CONDA_ENV:-correlascan_a40}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/stage_evef_mvd_pipeline.sh [options]

Options:
  --mode <all|corpus|pretrain_sanity|pretrain_full|bjut|dive|downstream|status>
  --dataset <bjut|dive|both>
  --resume <checkpoint_path>
  --force-corpus
  --skip-pretrain-sanity
  --skip-pretrain-full
  --help

Examples:
  bash scripts/stage_evef_mvd_pipeline.sh --mode all
  bash scripts/stage_evef_mvd_pipeline.sh --mode pretrain_full --resume checkpoints/pretrain_effect_flow_evm_bert_bjut_dive_train_balanced/last.pt
  bash scripts/stage_evef_mvd_pipeline.sh --mode downstream --dataset bjut
  bash scripts/stage_evef_mvd_pipeline.sh --mode status
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --dataset)
      DATASET_SCOPE="$2"
      shift 2
      ;;
    --resume)
      RESUME_FROM="$2"
      shift 2
      ;;
    --force-corpus)
      FORCE_CORPUS=1
      shift
      ;;
    --skip-pretrain-sanity)
      SKIP_PRETRAIN_SANITY=1
      shift
      ;;
    --skip-pretrain-full)
      SKIP_PRETRAIN_FULL=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

mkdir -p logs data/reports

if [[ -f "$HOME/.bashrc" ]]; then
  # shellcheck disable=SC1090
  source "$HOME/.bashrc" || true
fi
if command -v conda >/dev/null 2>&1; then
  conda activate "$CONDA_ENV" || true
fi

if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "[ERROR] neither python nor python3 is available in PATH"
  exit 1
fi

run() {
  echo "[RUN] $*"
  "$@"
}

run_python() {
  echo "[PY] $*"
  "$PYTHON_BIN" "$@"
}

require_two_gpus() {
  local visible
  visible="$("$PYTHON_BIN" -c 'import torch; print(torch.cuda.device_count())')"
  if [[ "$visible" -lt 2 ]]; then
    echo "[ERROR] EVEF-MVD pipeline expects at least 2 visible CUDA devices, got $visible"
    exit 1
  fi
}

global_template_dim() {
  "$PYTHON_BIN" - <<'PY'
import sys
from pathlib import Path
root = Path.cwd() / "src"
sys.path.insert(0, str(root))
from effect_flow_utils import GLOBAL_VULNERABILITY_LABELS
print(len(GLOBAL_VULNERABILITY_LABELS))
PY
}

print_status() {
  "$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

root = Path.cwd()
targets = {
    "stage16a_bjut_corpus": root / "data/processed/effect_flow_pretrain/BJUT/train_effect_flow_chunks.jsonl",
    "stage16a_dive_corpus": root / "data/processed/effect_flow_pretrain/DIVE/train_effect_flow_chunks.jsonl",
    "stage16b_sanity_report": root / "data/reports/stage16b_effect_flow_pretraining_sanity_report.json",
    "stage16b_full_report": root / "data/reports/stage16b_effect_flow_pretraining_report.json",
    "bjut_semantic_cache": root / "data/features/effect_flow_semantics/bjut_random_stride256_max32/train.pt",
    "dive_semantic_cache": root / "data/features/effect_flow_semantics/dive_random_stride256_max64/train.pt",
    "bjut_downstream_result": root / "results/train_bjut_effect_flow_guided_mil/checkpoint_summary.json",
    "dive_downstream_result": root / "results/train_dive_effect_flow_guided_mil/checkpoint_summary.json",
}
print("EVEF-MVD pipeline status")
print("")
for name, path in targets.items():
    print(f"{name}: {'OK' if path.exists() else 'MISSING'} -> {path}")

stage = "Stage 16A/ontology-corpus build pending"
if targets["stage16a_bjut_corpus"].exists() and targets["stage16a_dive_corpus"].exists():
    stage = "Stage P1/P2 corpus v2 ready"
if targets["stage16b_sanity_report"].exists():
    stage = "Stage P1/P2 pretraining sanity completed"
if targets["stage16b_full_report"].exists():
    stage = "Stage P1/P2 full pretraining completed"
if targets["bjut_semantic_cache"].exists() and targets["dive_semantic_cache"].exists():
    stage = "Semantic cache v2 extracted for BJUT and DIVE"
if targets["bjut_downstream_result"].exists() or targets["dive_downstream_result"].exists():
    stage = "Template-aware guided MIL downstream experiments have started"
if targets["bjut_downstream_result"].exists() and targets["dive_downstream_result"].exists():
    stage = "EVEF-MVD end-to-end mainline has run on both BJUT and DIVE"

print("")
print(f"current_stage: {stage}")
print("research_direction: ontology/template-driven effect-flow pretraining + template-aware guided MIL")
PY
}

build_corpus() {
  echo "[STAGE] ontology + corpus v2"
  run_python scripts/find_effect_flow_inputs.py
  local force_args=()
  if [[ "$FORCE_CORPUS" -eq 1 ]]; then
    force_args+=(--force)
  fi
  run_python scripts/build_effect_flow_pretraining_corpus.py \
    --dataset BJUT \
    --max_chunks_per_contract 32 \
    --num_workers 4 \
    "${force_args[@]}"
  run_python scripts/build_effect_flow_pretraining_corpus.py \
    --dataset DIVE \
    --max_chunks_per_contract 64 \
    --num_workers 4 \
    "${force_args[@]}"
  run_python scripts/audit_effect_flow_annotations.py --dataset BJUT
  run_python scripts/audit_effect_flow_annotations.py --dataset DIVE
  run_python scripts/audit_label_pattern_relevance.py --dataset BJUT --split train
  run_python scripts/audit_label_pattern_relevance.py --dataset DIVE --split train
  run_python scripts/select_usable_efpp_patterns.py
  run_python scripts/summarize_stage16a_effect_flow.py
}

run_pretrain_sanity() {
  echo "[STAGE] pretraining sanity"
  require_two_gpus
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
  torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" \
    src/pretrain_effect_flow_evm_bert.py \
    --config configs/sanity_pretrain_effect_flow_evm_bert_bjut_dive_train_balanced.yaml
}

run_pretrain_full() {
  echo "[STAGE] pretraining full"
  require_two_gpus
  if [[ ! -f data/reports/stage16b_effect_flow_pretraining_sanity_report.json ]]; then
    echo "[ERROR] sanity pretraining report missing. Run --mode pretrain_sanity first."
    exit 1
  fi
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
  local args=(--config configs/pretrain_effect_flow_evm_bert_bjut_dive_train_balanced.yaml)
  if [[ -n "$RESUME_FROM" ]]; then
    args+=(--resume "$RESUME_FROM")
  fi
  torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" \
    src/pretrain_effect_flow_evm_bert.py \
    "${args[@]}"
}

ensure_base_chunk_cache() {
  local dataset="$1"
  local feature_dir data_dir config_path max_chunks num_labels output_path
  if [[ "$dataset" == "BJUT" ]]; then
    feature_dir="data/features/continued_dive_evm_bert_bjut_random_stride256"
    data_dir="data/processed/BJUT_SC01_random_split"
    config_path="configs/extract_continued_dive_evm_bert_bjut_random_stride256.yaml"
    max_chunks=32
    num_labels=10
    output_path="data/reports/check_bjut_effect_flow_guided_base_features.txt"
  else
    feature_dir="data/features/continued_dive_evm_bert_dive_random_stride256_max64"
    data_dir="data/processed/DIVE_random_split"
    config_path="configs/extract_continued_dive_evm_bert_dive_random_stride256_max64.yaml"
    max_chunks=64
    num_labels=8
    output_path="data/reports/check_dive_effect_flow_guided_base_features.txt"
  fi
  if [[ ! -f "$feature_dir/train.pt" ]]; then
    echo "[INFO] base chunk feature cache missing for $dataset, extracting now"
    run_python scripts/extract_evm_bert_chunk_features.py --config "$config_path"
  fi
  run_python scripts/check_evm_bert_chunk_features.py \
    --feature_dir "$feature_dir" \
    --data_dir "$data_dir" \
    --expected_max_chunks "$max_chunks" \
    --expected_feature_dim 768 \
    --expected_num_labels "$num_labels" \
    --output "$output_path"
}

extract_semantic_cache() {
  local dataset="$1"
  local config_path
  if [[ "$dataset" == "BJUT" ]]; then
    config_path="configs/extract_bjut_effect_flow_semantics_stride256.yaml"
  else
    config_path="configs/extract_dive_effect_flow_semantics_stride256_max64.yaml"
  fi
  run_python scripts/extract_effect_flow_semantic_features.py --config "$config_path"
}

check_semantic_cache_v2() {
  local dataset="$1"
  local semantic_dir feature_dir data_dir max_chunks num_labels output_path
  local template_dim
  template_dim="$(global_template_dim)"
  if [[ "$dataset" == "BJUT" ]]; then
    semantic_dir="data/features/effect_flow_semantics/bjut_random_stride256_max32"
    feature_dir="data/features/continued_dive_evm_bert_bjut_random_stride256"
    data_dir="data/processed/BJUT_SC01_random_split"
    max_chunks=32
    num_labels=10
    output_path="data/reports/check_bjut_effect_flow_guided_semantic_features.txt"
  else
    semantic_dir="data/features/effect_flow_semantics/dive_random_stride256_max64"
    feature_dir="data/features/continued_dive_evm_bert_dive_random_stride256_max64"
    data_dir="data/processed/DIVE_random_split"
    max_chunks=64
    num_labels=8
    output_path="data/reports/check_dive_effect_flow_guided_semantic_features.txt"
  fi
  run_python scripts/check_effect_flow_semantic_features.py \
    --semantic_dir "$semantic_dir" \
    --feature_dir "$feature_dir" \
    --data_dir "$data_dir" \
    --expected_max_chunks "$max_chunks" \
    --expected_efpp_dim 22 \
    --expected_etp_dim 16 \
    --expected_relation_dim 6 \
    --expected_global_template_dim "$template_dim" \
    --expected_num_labels "$num_labels" \
    --output "$output_path"
}

train_eval_dataset() {
  local dataset="$1"
  local sanity_config train_config checkpoint_dir result_dir
  if [[ "$dataset" == "BJUT" ]]; then
    sanity_config="configs/sanity_bjut_effect_flow_guided_mil.yaml"
    train_config="configs/train_bjut_effect_flow_guided_mil.yaml"
    checkpoint_dir="checkpoints/train_bjut_effect_flow_guided_mil"
    result_dir="results/train_bjut_effect_flow_guided_mil"
  else
    sanity_config="configs/sanity_dive_effect_flow_guided_mil.yaml"
    train_config="configs/train_dive_effect_flow_guided_mil.yaml"
    checkpoint_dir="checkpoints/train_dive_effect_flow_guided_mil"
    result_dir="results/train_dive_effect_flow_guided_mil"
  fi

  echo "[DATASET] $dataset base chunk cache"
  ensure_base_chunk_cache "$dataset"
  echo "[DATASET] $dataset semantic cache v2"
  extract_semantic_cache "$dataset"
  check_semantic_cache_v2 "$dataset"
  echo "[DATASET] $dataset sanity train"
  run_python src/train_chunk_mil.py --config "$sanity_config"
  echo "[DATASET] $dataset full train"
  run_python src/train_chunk_mil.py --config "$train_config"
  echo "[DATASET] $dataset threshold calibration"
  run_python src/evaluate_chunk_mil.py \
    --config "$train_config" \
    --checkpoint "$checkpoint_dir/best_macro_f1.pt" \
    --split valid \
    --threshold_search global_and_per_label \
    --output_prefix threshold_calibration
  run_python src/evaluate_chunk_mil.py \
    --config "$train_config" \
    --checkpoint "$checkpoint_dir/best_macro_f1.pt" \
    --split test \
    --threshold_file "$result_dir/per_label_thresholds_valid.json" \
    --global_threshold_file "$result_dir/valid_global_threshold_scan.json" \
    --output_prefix threshold_calibration \
    --save_predictions
  run_python scripts/summarize_effect_flow_guided_semantics.py --config "$train_config"
}

run_downstream_scope() {
  require_two_gpus
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
  case "$DATASET_SCOPE" in
    bjut|BJUT)
      train_eval_dataset "BJUT"
      ;;
    dive|DIVE)
      train_eval_dataset "DIVE"
      ;;
    both)
      train_eval_dataset "BJUT"
      train_eval_dataset "DIVE"
      ;;
    *)
      echo "[ERROR] unsupported dataset scope: $DATASET_SCOPE"
      exit 1
      ;;
  esac
}

case "$MODE" in
  all)
    build_corpus
    if [[ "$SKIP_PRETRAIN_SANITY" -eq 0 ]]; then
      run_pretrain_sanity
    fi
    if [[ "$SKIP_PRETRAIN_FULL" -eq 0 ]]; then
      run_pretrain_full
    fi
    run_downstream_scope
    ;;
  corpus)
    build_corpus
    ;;
  pretrain_sanity)
    run_pretrain_sanity
    ;;
  pretrain_full)
    run_pretrain_full
    ;;
  bjut)
    DATASET_SCOPE="bjut"
    run_downstream_scope
    ;;
  dive)
    DATASET_SCOPE="dive"
    run_downstream_scope
    ;;
  downstream)
    run_downstream_scope
    ;;
  status)
    print_status
    ;;
  *)
    echo "[ERROR] unsupported mode: $MODE"
    usage
    exit 1
    ;;
esac

echo "[OK] stage_evef_mvd_pipeline completed: mode=$MODE dataset=$DATASET_SCOPE"
