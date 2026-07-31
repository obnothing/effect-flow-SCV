#!/usr/bin/env bash
#SBATCH --job-name=main6_csdg_train
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/main6_csdg_train_%j.out
#SBATCH --error=logs/main6_csdg_train_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source ~/.bashrc
conda activate correlascan_a40
mkdir -p logs
for variant in cfg_residual cfg_stack_residual cfg_stack_storage_residual graph_mlp_control; do
  torchrun --standalone --nproc_per_node=2 scripts/train_main6_opcode_csdg.py \
    --config configs/train_main6_opcode_csdg.yaml \
    --variant "$variant"
  python scripts/analyze_main6_opcode_csdg.py \
    --config configs/train_main6_opcode_csdg.yaml \
    --variant "$variant"
done

P95=$(python -c "import json; print(json.load(open('data/reports/main6_opcode_csdg/graph_audit.json'))['splits']['train']['basic_blocks_p95'])")
BASELINE=$(python -c "import json; p=json.load(open('results/main6_random_090_mlm8/mlm8_slot3/checkpoint_summary.json')); print(p.get('valid_macro_f1', p.get('best_macro_f1', -1)))")
STACK=$(python -c "import json; print(json.load(open('results/main6_opcode_csdg/cfg_stack_residual/valid_summary.json'))['valid_macro_f1'])")
if python - "$P95" "$BASELINE" "$STACK" <<'PY'
import sys
p95, baseline, stack = map(float, sys.argv[1:])
raise SystemExit(0 if p95 > 256 and stack >= baseline else 1)
PY
then
  torchrun --standalone --nproc_per_node=2 scripts/train_main6_opcode_csdg.py \
    --config configs/train_main6_opcode_csdg.yaml \
    --variant cfg_stack_sparse_topk
  python scripts/analyze_main6_opcode_csdg.py \
    --config configs/train_main6_opcode_csdg.yaml \
    --variant cfg_stack_sparse_topk
fi
