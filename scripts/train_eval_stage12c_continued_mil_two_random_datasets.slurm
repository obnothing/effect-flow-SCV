#!/usr/bin/env bash
#SBATCH --job-name=stage12c_two_random_mil
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --output=logs/slurm_stage12c_two_random_mil_%j.out
#SBATCH --error=logs/slurm_stage12c_two_random_mil_%j.err

set -eo pipefail

cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc
conda activate correlascan_a40
mkdir -p logs

CONTINUED_MODEL="checkpoints/continue_pretrain_evm_bert_dive_train_only/hf_model"
for required in \
  "$CONTINUED_MODEL/config.json" \
  "data/processed/BJUT_SC01/evm_vocab.json" \
  "data/processed/BJUT_SC01/train.jsonl" \
  "data/processed/BJUT_SC01/valid.jsonl" \
  "data/processed/BJUT_SC01/test.jsonl" \
  "data/processed/DIVE/train.jsonl" \
  "data/processed/DIVE/valid.jsonl" \
  "data/processed/DIVE/test.jsonl"; do
  if [[ ! -f "$required" ]]; then
    echo "[ERROR] missing required file: $required"
    exit 1
  fi
done
if [[ ! -f "$CONTINUED_MODEL/model.safetensors" && ! -f "$CONTINUED_MODEL/pytorch_model.bin" ]]; then
  echo "[ERROR] continued EVM-BERT weights missing from $CONTINUED_MODEL"
  exit 1
fi

echo "[PREP] BJUT random 80/10/10 split"
python scripts/create_bjut_random_split.py \
  --source_data_dir data/processed/BJUT_SC01 \
  --output_data_dir data/processed/BJUT_SC01_random_split \
  --seed 42 \
  --report_path data/reports/bjut_sc01_random_split_report.txt
python scripts/check_split_leakage.py \
  --data_dir data/processed/BJUT_SC01_random_split \
  --output data/reports/split_leakage_random_split_report.txt
python scripts/apply_mlsmote.py --config configs/mlsmote_random_split.yaml

echo "[PREP] DIVE random 80/10/10 split"
python scripts/create_dive_random_split.py \
  --source_data_dir data/processed/DIVE \
  --output_data_dir data/processed/DIVE_random_split \
  --seed 42 \
  --train_ratio 0.8 \
  --valid_ratio 0.1 \
  --test_ratio 0.1
python scripts/check_split_leakage.py \
  --data_dir data/processed/DIVE_random_split \
  --output data/reports/split_leakage_dive_random_split_report.txt
python scripts/apply_mlsmote.py --config configs/mlsmote_dive_random_split.yaml

python scripts/compare_bjut_dive_random_structure.py

run_bjut() {
  export CUDA_VISIBLE_DEVICES=0
  echo "[BJUT] extracting continued-EVM-BERT stride256 features"
  python scripts/extract_evm_bert_chunk_features.py \
    --config configs/extract_continued_dive_evm_bert_bjut_random_stride256.yaml
  python scripts/check_evm_bert_chunk_features.py \
    --feature_dir data/features/continued_dive_evm_bert_bjut_random_stride256 \
    --data_dir data/processed/BJUT_SC01_random_split \
    --expected_max_chunks 32 \
    --expected_feature_dim 768 \
    --expected_num_labels 10 \
    --output data/reports/check_continued_dive_evm_bert_bjut_random_stride256.txt
  python src/train_chunk_mil.py \
    --config configs/train_continued_dive_evm_bert_bjut_random_stride256_mil.yaml
  python src/evaluate_chunk_mil.py \
    --config configs/train_continued_dive_evm_bert_bjut_random_stride256_mil.yaml \
    --checkpoint checkpoints/train_continued_dive_evm_bert_bjut_random_stride256_mil/best_macro_f1.pt \
    --split valid \
    --threshold_search global_and_per_label \
    --output_prefix threshold_calibration
  python src/evaluate_chunk_mil.py \
    --config configs/train_continued_dive_evm_bert_bjut_random_stride256_mil.yaml \
    --checkpoint checkpoints/train_continued_dive_evm_bert_bjut_random_stride256_mil/best_macro_f1.pt \
    --split test \
    --threshold_file results/train_continued_dive_evm_bert_bjut_random_stride256_mil/per_label_thresholds_valid.json \
    --global_threshold_file results/train_continued_dive_evm_bert_bjut_random_stride256_mil/valid_global_threshold_scan.json \
    --output_prefix threshold_calibration \
    --save_predictions
}

run_dive() {
  export CUDA_VISIBLE_DEVICES=1
  echo "[DIVE] extracting continued-EVM-BERT stride256 features"
  python scripts/extract_evm_bert_chunk_features.py \
    --config configs/extract_continued_dive_evm_bert_dive_random_stride256.yaml
  python scripts/check_evm_bert_chunk_features.py \
    --feature_dir data/features/continued_dive_evm_bert_dive_random_stride256 \
    --data_dir data/processed/DIVE_random_split \
    --expected_max_chunks 32 \
    --expected_feature_dim 768 \
    --expected_num_labels 8 \
    --output data/reports/check_continued_dive_evm_bert_dive_random_stride256.txt
  python src/train_chunk_mil.py \
    --config configs/train_continued_dive_evm_bert_dive_random_stride256_mil.yaml
  python src/evaluate_chunk_mil.py \
    --config configs/train_continued_dive_evm_bert_dive_random_stride256_mil.yaml \
    --checkpoint checkpoints/train_continued_dive_evm_bert_dive_random_stride256_mil/best_macro_f1.pt \
    --split valid \
    --threshold_search global_and_per_label \
    --output_prefix threshold_calibration
  python src/evaluate_chunk_mil.py \
    --config configs/train_continued_dive_evm_bert_dive_random_stride256_mil.yaml \
    --checkpoint checkpoints/train_continued_dive_evm_bert_dive_random_stride256_mil/best_macro_f1.pt \
    --split test \
    --threshold_file results/train_continued_dive_evm_bert_dive_random_stride256_mil/per_label_thresholds_valid.json \
    --global_threshold_file results/train_continued_dive_evm_bert_dive_random_stride256_mil/valid_global_threshold_scan.json \
    --output_prefix threshold_calibration \
    --save_predictions
}

run_bjut >"logs/stage12c_bjut_${SLURM_JOB_ID}.log" 2>&1 &
BJUT_PID=$!
run_dive >"logs/stage12c_dive_${SLURM_JOB_ID}.log" 2>&1 &
DIVE_PID=$!

STATUS=0
wait "$BJUT_PID" || STATUS=1
wait "$DIVE_PID" || STATUS=1
if [[ "$STATUS" -ne 0 ]]; then
  echo "[ERROR] at least one dataset branch failed"
  echo "[INFO] inspect logs/stage12c_bjut_${SLURM_JOB_ID}.log"
  echo "[INFO] inspect logs/stage12c_dive_${SLURM_JOB_ID}.log"
  exit "$STATUS"
fi

python scripts/summarize_stage12c_two_random_results.py

echo "[OK] Stage 12C two-dataset random-split full MIL evaluation completed"
