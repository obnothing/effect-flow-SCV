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

for variant in mlm_label_mil etp_encoder_mil etp_concat_mil ld_etpca ld_etpca_sep
do
  test -f "results/main6_random_090/$variant/checkpoint_summary.json"
  test -f "checkpoints/main6_random_090/$variant/best_macro_f1.pt"
done

for variant in ld_etpca ld_etpca_sep
do
  test -f "checkpoints/main6_random_090/$variant/best_macro_f1.pt"
done

bash scripts/run_main6_random_090.sh analyze
bash scripts/run_main6_random_090.sh select
test -f results/main6_random_090/validation_selection.json
