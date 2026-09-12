#!/usr/bin/env bash
set -eo pipefail

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "${HOME}/.bashrc"
conda activate pytorch-2.1.1
set -u

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
mkdir -p logs

python scripts/generate_baseline_sweep_configs.py

for config in configs/light_label/baseline_sweep/pw_*.yaml
do
    echo "[class_imbalance] ${config}"
    python scripts/resolve_light_label_runtime.py --config "${config}"
    python scripts/audit_light_label_config.py --config "${config}"
    python src/train_light_label_model.py --config "${config}" 2>&1 | tee -a logs/baseline_sweep.log
done

for name in hidden192 hidden256 hidden384 embed256 embed256_hidden256
do
    config="configs/light_label/baseline_sweep/${name}.yaml"
    echo "[capacity] ${config}"
    python scripts/resolve_light_label_runtime.py --config "${config}"
    python scripts/audit_light_label_config.py --config "${config}"
    python src/train_light_label_model.py --config "${config}" 2>&1 | tee -a logs/baseline_sweep.log
done

for name in lr3e4 lr5e4 wd3e4 wd1e3
do
    config="configs/light_label/baseline_sweep/${name}.yaml"
    echo "[optimizer] ${config}"
    python scripts/resolve_light_label_runtime.py --config "${config}"
    python scripts/audit_light_label_config.py --config "${config}"
    python src/train_light_label_model.py --config "${config}" 2>&1 | tee -a logs/baseline_sweep.log
done

python - <<'PY'
import csv
import glob
import json

rows = []
for path in glob.glob("results/light_label/baseline_sweep/*/full/metrics.json"):
    item = json.load(open(path, encoding="utf-8"))
    rows.append({
        "variant": path.split("/")[-3],
        "macro_f1": item["metrics"]["tuned"]["macro_f1"],
        "micro_f1": item["metrics"]["tuned"]["micro_f1"],
        "dos_f1": item["metrics"]["tuned"]["per_label_f1"][4],
        "best_epoch": item["best_epoch"],
    })
rows.sort(key=lambda row: row["macro_f1"], reverse=True)
with open("results/light_label/baseline_sweep/summary.csv", "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps({"completed": len(rows), "test_checked": False}, indent=2))
PY
