#!/usr/bin/env bash
#SBATCH --job-name=main6_downstream_valid
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/main6_downstream_valid_%j.out
#SBATCH --error=logs/main6_downstream_valid_%j.err

cd "${SLURM_SUBMIT_DIR:?Submit this job with sbatch from the project root}"
source ~/.bashrc
conda activate correlascan_a40
set -euo pipefail
mkdir -p logs

for path in \
  data/features/main6_random_mlm8_trainvalid/train.pt \
  data/features/main6_random_mlm8_trainvalid/valid.pt
do
  test -f "$path"
done

for path in \
  data/features/main6_random_mlm8_trainvalid/test.pt
do
  if [[ -e "$path" ]]; then
    echo "Refusing downstream validation training: locked test cache exists: $path" >&2
    exit 2
  fi
done

bash scripts/run_main6_random_090.sh train
