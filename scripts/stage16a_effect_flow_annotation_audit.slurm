#!/usr/bin/env bash
#SBATCH --job-name=stage16a_effect_flow
#SBATCH --partition=cpu56c
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_stage16a_effect_flow_%j.out
#SBATCH --error=logs/slurm_stage16a_effect_flow_%j.err

set -eo pipefail

cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc
conda activate correlascan_a40
mkdir -p logs data/reports

echo "[1/10] inventory strict inputs"
python scripts/find_effect_flow_inputs.py

echo "[2/10] build BJUT label-free effect-flow corpus"
python scripts/build_effect_flow_pretraining_corpus.py \
  --dataset BJUT \
  --num_workers 4 \
  --force

echo "[3/10] build DIVE label-free effect-flow corpus"
python scripts/build_effect_flow_pretraining_corpus.py \
  --dataset DIVE \
  --num_workers 4 \
  --force

echo "[4/10] audit BJUT annotations"
python scripts/audit_effect_flow_annotations.py --dataset BJUT

echo "[5/10] audit DIVE annotations"
python scripts/audit_effect_flow_annotations.py --dataset DIVE

echo "[6/10] train-only BJUT label-pattern relevance"
python scripts/audit_label_pattern_relevance.py --dataset BJUT --split train

echo "[7/10] train-only DIVE label-pattern relevance"
python scripts/audit_label_pattern_relevance.py --dataset DIVE --split train

echo "[8/10] all-splits diagnostics (never used as training priors)"
python scripts/audit_label_pattern_relevance.py --dataset BJUT --all_splits_diagnostic
python scripts/audit_label_pattern_relevance.py --dataset DIVE --all_splits_diagnostic

echo "[9/10] select candidate EFPP patterns"
python scripts/select_usable_efpp_patterns.py

echo "[10/10] summarize Stage 16A"
python scripts/summarize_stage16a_effect_flow.py

echo "[OK] Stage 16A completed without model training."
