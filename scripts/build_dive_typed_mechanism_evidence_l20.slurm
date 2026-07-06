#!/usr/bin/env bash
#SBATCH --job-name=dive_typed_ev
#SBATCH --partition=gpu-a10
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_dive_typed_mechanism_evidence_%j.out
#SBATCH --error=logs/slurm_dive_typed_mechanism_evidence_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs data/reports data/features/typed_mechanism_evidence

set +u
source ~/.bashrc || true
conda activate correlascan_a40
set -u

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

RUN_SANITY="${RUN_SANITY:-0}"
SPLITS="${SPLITS:-train valid test}"
NUM_WORKERS="${NUM_WORKERS:-4}"
FORCE="${FORCE:-0}"

if [[ "${RUN_SANITY}" == "1" ]]; then
  OUTPUT_DIR="data/features/typed_mechanism_evidence/dive_random_stride256_max64_sanity"
  DEBUG_ARGS=(--debug_num_contracts 512)
else
  OUTPUT_DIR="data/features/typed_mechanism_evidence/dive_random_stride256_max64"
  DEBUG_ARGS=()
fi

FORCE_ARGS=()
if [[ "${FORCE}" == "1" ]]; then
  FORCE_ARGS=(--force)
fi

echo "[INFO] current dir: $(pwd)"
echo "[INFO] python: $(which python)"
echo "[INFO] RUN_SANITY=${RUN_SANITY}"
echo "[INFO] SPLITS=${SPLITS}"
echo "[INFO] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[INFO] NUM_WORKERS=${NUM_WORKERS}"

echo "[DIVE 1/3] check strong base chunk cache"
python scripts/check_evm_bert_chunk_features.py \
  --feature_dir data/features/continued_dive_evm_bert_dive_random_stride256_max64 \
  --data_dir data/processed/DIVE_random_split \
  --expected_max_chunks 64 \
  --expected_feature_dim 768 \
  --expected_num_labels 8 \
  --output data/reports/check_dive_typed_mechanism_base_features.txt

echo "[DIVE 2/3] build typed mechanism evidence and audit"
python scripts/build_dive_typed_mechanism_evidence.py \
  --data_dir data/processed/DIVE_random_split \
  --feature_dir data/features/continued_dive_evm_bert_dive_random_stride256_max64 \
  --baseline_prediction_dir results/dive_side_scale_search/train_dive_side_scale_150_ep50 \
  --output_dir "${OUTPUT_DIR}" \
  --splits ${SPLITS} \
  --chunk_size 512 \
  --chunk_stride 256 \
  --max_chunks 64 \
  --num_labels 8 \
  --num_workers "${NUM_WORKERS}" \
  "${DEBUG_ARGS[@]}" \
  "${FORCE_ARGS[@]}"

echo "[DIVE 3/3] copy compact reports"
cp "${OUTPUT_DIR}/summary.txt" data/reports/check_dive_typed_mechanism_summary.txt
for split in ${SPLITS}; do
  if [[ -f "${OUTPUT_DIR}/${split}.audit.txt" ]]; then
    cp "${OUTPUT_DIR}/${split}.audit.txt" "data/reports/check_dive_typed_mechanism_${split}.txt"
  fi
done

echo "[OK] DIVE typed mechanism evidence audit finished."
