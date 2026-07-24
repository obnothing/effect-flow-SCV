#!/usr/bin/env bash
#SBATCH --job-name=main6_multirole_etp_cache
#SBATCH --partition=gpu-l20
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/main6_multirole_etp_cache_%j.out
#SBATCH --error=logs/main6_multirole_etp_cache_%j.err

cd "${SLURM_SUBMIT_DIR:?Submit this job with sbatch from the project root}"
source ~/.bashrc
conda activate correlascan_a40
set -euo pipefail
mkdir -p logs

test -f checkpoints/pretrain_evm_bert_19143_base_continue/hf_model/config.json
test -f checkpoints/pretrain_multirole_etp_main6_random/hf_model/config.json

for path in \
  data/features/main6_random_mlm_control_trainvalid/test.pt \
  data/features/main6_random_multirole_etp/test.pt \
  data/features/main6_random_multirole_etp_tokens/test.pt
do
  if [[ -e "$path" ]]; then
    echo "Refusing to run train/valid cache stage: locked test cache exists: $path" >&2
    exit 2
  fi
done

bash scripts/run_main6_random_090.sh extract
bash scripts/run_main6_random_090.sh validate
