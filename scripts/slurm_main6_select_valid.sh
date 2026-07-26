#!/usr/bin/env bash
#SBATCH --job-name=main6_select_valid
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/main6_select_valid_%j.out
#SBATCH --error=logs/main6_select_valid_%j.err

cd "${SLURM_SUBMIT_DIR:?Submit this job with sbatch from the project root}"
source ~/.bashrc
conda activate correlascan_a40
set -euo pipefail
mkdir -p logs

test -f results/main6_random_090/mlm_label_mil/checkpoint_summary.json

for variant in mlm8_slot1 mlm8_slot2 mlm8_slot3 mlm8_slot4
do
  test -f "results/main6_random_090_mlm8/$variant/checkpoint_summary.json"
  test -f "checkpoints/main6_random_090_mlm8/$variant/best_macro_f1.pt"
done

bash scripts/run_main6_random_090.sh select
test -f results/main6_random_090_mlm8/validation_selection.json
