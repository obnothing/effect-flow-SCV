#!/usr/bin/env bash
#SBATCH --job-name=stage16a_dive64
#SBATCH --partition=cpu56c
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=16:00:00
#SBATCH --output=logs/slurm_stage16a_dive64_%j.out
#SBATCH --error=logs/slurm_stage16a_dive64_%j.err

set -eo pipefail

cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc
conda activate correlascan_a40
mkdir -p logs data/reports

echo "[1/6] rebuild DIVE label-free corpus with max_chunks=64"
python scripts/build_effect_flow_pretraining_corpus.py \
  --dataset DIVE \
  --max_chunks_per_contract 64 \
  --num_workers 4 \
  --force

echo "[2/6] refresh DIVE annotation audit"
python scripts/audit_effect_flow_annotations.py --dataset DIVE

echo "[3/6] refresh train-only DIVE label-pattern relevance"
python scripts/audit_label_pattern_relevance.py \
  --dataset DIVE \
  --split train

echo "[4/6] refresh DIVE all-splits diagnostics"
python scripts/audit_label_pattern_relevance.py \
  --dataset DIVE \
  --all_splits_diagnostic

echo "[5/6] refresh joint EFPP recommendation using unchanged BJUT reports"
python scripts/select_usable_efpp_patterns.py

echo "[6/6] refresh Stage 16A summary using unchanged BJUT reports"
python scripts/summarize_stage16a_effect_flow.py

echo "[OK] DIVE max-64 refresh completed; BJUT corpus was not rebuilt."
