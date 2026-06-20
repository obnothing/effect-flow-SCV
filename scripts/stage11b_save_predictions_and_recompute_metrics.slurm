#!/usr/bin/env bash
#SBATCH --job-name=stage11b_predictions
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/slurm_stage11b_predictions_%j.out
#SBATCH --error=logs/slurm_stage11b_predictions_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc
conda activate correlascan_a40
mkdir -p logs

run_eval() {
  local name="$1"
  local script="$2"
  local config="$3"
  local checkpoint="$4"
  local extra_args="$5"

  echo "[STAGE11B] ${name}"
  if [[ ! -f "$config" ]]; then
    echo "[WARNING] skip ${name}: missing config ${config}"
    return 0
  fi
  if [[ ! -f "$checkpoint" ]]; then
    echo "[WARNING] skip ${name}: missing checkpoint ${checkpoint}"
    return 0
  fi
  python "$script" \
    --config "$config" \
    --checkpoint "$checkpoint" \
    --split test \
    ${extra_args} \
    --save_predictions
}

run_eval \
  "CodeBERT weighted strict baseline" \
  "src/evaluate.py" \
  "configs/train_mlsmote_codebert_weighted_server.yaml" \
  "checkpoints/train_mlsmote_codebert_weighted/best_macro_f1.pt" \
  "--threshold auto"

run_eval \
  "EVM-BERT first512 weighted baseline" \
  "src/evaluate.py" \
  "configs/train_evm_bert_weighted_server.yaml" \
  "checkpoints/train_evm_bert_weighted/best_macro_f1.pt" \
  "--threshold auto"

run_eval \
  "strict grouped stride256 MIL threshold 0.5" \
  "src/evaluate_chunk_mil.py" \
  "configs/train_evm_bert_chunk_mil_mean_stride256_labelattn_weighted_server.yaml" \
  "checkpoints/train_evm_bert_chunk_mil_mean_stride256_labelattn_weighted/best_macro_f1.pt" \
  "--threshold 0.5"

run_eval \
  "random split stride256 MIL best global threshold 0.55" \
  "src/evaluate_chunk_mil.py" \
  "configs/train_evm_bert_chunk_mil_mean_stride256_labelattn_weighted_random_split_server.yaml" \
  "checkpoints/train_evm_bert_chunk_mil_mean_stride256_labelattn_weighted_random_split/best_macro_f1.pt" \
  "--threshold 0.55"

python scripts/recompute_metrics_all_variants.py \
  --output_stem metric_recomputation_summary_completed

python scripts/report_stage11b_prediction_artifacts.py
