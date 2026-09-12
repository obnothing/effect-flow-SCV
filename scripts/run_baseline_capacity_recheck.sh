#!/usr/bin/env bash
set -eo pipefail

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
source "${HOME}/.bashrc"
conda activate pytorch-2.1.1
set -u

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
mkdir -p logs configs/light_label/baseline_recheck

python - <<'PY'
from pathlib import Path
import yaml

root = Path.cwd()
names = ("hidden192", "hidden256", "hidden384", "embed256_hidden256")
for name in names:
    source = root / "configs/light_label/baseline_sweep" / f"{name}.yaml"
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config.update({
        "runtime_path": f"results/light_label/baseline_sweep_corrected_runtime/{name}.json",
        "result_dir": f"results/light_label/baseline_sweep_corrected/{name}",
        "checkpoint_dir": f"checkpoints/light_label/baseline_sweep_corrected/{name}",
        "allow_test": False,
    })
    target = root / "configs/light_label/baseline_recheck" / f"{name}.yaml"
    target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
print(f"generated corrected capacity configs: {len(names)}")
PY

for name in hidden192 hidden256 hidden384 embed256_hidden256
do
    config="configs/light_label/baseline_recheck/${name}.yaml"
    echo "[capacity_recheck] ${config}"
    python scripts/resolve_light_label_runtime.py --config "${config}"
    python scripts/audit_light_label_config.py --config "${config}"
    python src/train_light_label_model.py --config "${config}" 2>&1 | tee -a logs/baseline_capacity_recheck.log
done

python - <<'PY'
import csv
import glob
import json
from pathlib import Path

rows = []
for path in glob.glob("results/light_label/baseline_sweep_corrected/*/full/metrics.json"):
    item = json.loads(Path(path).read_text(encoding="utf-8"))
    config = item["actual_config"]
    rows.append({
        "variant": Path(path).parents[1].name,
        "embedding_dim": config["embedding_dim"],
        "gru_hidden_size": config["gru_hidden_size"],
        "gru_layers": config["gru_layers"],
        "macro_f1": item["metrics"]["tuned"]["macro_f1"],
        "micro_f1": item["metrics"]["tuned"]["micro_f1"],
        "dos_f1": item["metrics"]["tuned"]["per_label_f1"][4],
        "total_params": item["total_params"],
        "best_epoch": item["best_epoch"],
        "test_checked": item["test_checked"],
    })
rows.sort(key=lambda row: row["macro_f1"], reverse=True)
output = Path("results/light_label/baseline_sweep_corrected/summary.csv")
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps({"completed": len(rows), "summary": str(output), "test_checked": False}, indent=2))
PY
