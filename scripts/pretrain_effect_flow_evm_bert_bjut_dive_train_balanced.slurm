#!/usr/bin/env bash
#SBATCH --job-name=eflow_pretrain
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/slurm_eflow_pretrain_%j.out
#SBATCH --error=logs/slurm_eflow_pretrain_%j.err

set -eo pipefail

cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc
conda activate correlascan_a40
mkdir -p logs

python - <<'PY'
import json
from pathlib import Path

path = Path("data/reports/stage16b_effect_flow_pretraining_sanity_report.json")
if not path.exists():
    raise SystemExit("[ERROR] Run Stage 16B sanity first; sanity report is missing.")
report = json.loads(path.read_text(encoding="utf-8"))
if not report.get("sanity_success"):
    raise SystemExit("[ERROR] Stage 16B sanity did not pass; full pretraining was not started.")
print("[OK] Stage 16B sanity gate passed.")
PY

ARGS=(
  --config configs/pretrain_effect_flow_evm_bert_bjut_dive_train_balanced.yaml
)
if [[ -n "${RESUME_FROM:-}" ]]; then
  ARGS+=(--resume "$RESUME_FROM")
fi

export OMP_NUM_THREADS=2
torchrun --standalone --nproc_per_node=2 \
  src/pretrain_effect_flow_evm_bert.py "${ARGS[@]}"
