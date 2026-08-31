#!/usr/bin/env bash
set -euo pipefail

# The extract stage is intentionally absent: the user has already completed
# and checked train/valid relation caches.
mkdir -p logs

pretrain_job="$(sbatch --parsable scripts/slurm_main6_stack_relational_pretrain.slurm)"
pretrain_job="${pretrain_job%%;*}"

cache_job="$(sbatch --parsable \
  --dependency="afterok:${pretrain_job}" \
  scripts/slurm_main6_stack_relational_cache.slurm)"
cache_job="${cache_job%%;*}"

train_job="$(sbatch --parsable \
  --dependency="afterok:${cache_job}" \
  scripts/slurm_main6_stack_relational_train.slurm)"
train_job="${train_job%%;*}"

select_job="$(sbatch --parsable \
  --dependency="afterok:${train_job}" \
  scripts/slurm_main6_stack_relational_select.slurm)"
select_job="${select_job%%;*}"

printf 'PRETRAIN=%s\nCACHE=%s\nTRAIN=%s\nSELECT=%s\n' \
  "${pretrain_job}" "${cache_job}" "${train_job}" "${select_job}"
printf '\nMonitor with:\n'
printf '  squeue -j %s,%s,%s,%s\n' "${pretrain_job}" "${cache_job}" "${train_job}" "${select_job}"
printf '  tail -f logs/main6_stack_relational_pretrain_%s.out\n' "${pretrain_job}"
