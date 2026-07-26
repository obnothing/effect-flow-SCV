# EVEF-MVD Main-6

Opcode-only smart-contract multi-label vulnerability detection for DIVE Main-6.

Current route:

```text
19,143 unique public runtime-opcode contracts
-> Main-6 train-only continued MLM
-> 64-chunk, eight-view pure-MLM caches
-> label-conditioned multi-slot MIL
```

Labels: Reentrancy, Access Control, Arithmetic, Unchecked Return Values, DoS,
and Time manipulation.

## Data

Datasets are not committed. Transfer these directories to the server before
running:

```text
data/processed/DIVE_main6_random_split
data/processed/ethereum_public_pretrain_19143_unique_runtime
```

## Server Run

Submit the full pretraining chain from the project root. Every dependency is
`afterok`. Continued MLM is initialized by the public 19,143 checkpoint and
then consumes only Main-6 train opcodes:

```bash
mkdir -p logs
BASE=$(sbatch --parsable scripts/slurm_main6_base_19143.sh)
CONT=$(sbatch --parsable --dependency=afterok:$BASE scripts/slurm_main6_19143_continue.sh)
MLM8=$(sbatch --parsable --dependency=afterok:$CONT scripts/slurm_main6_mlm8_pretrain.sh)
CACHE=$(sbatch --parsable --dependency=afterok:$MLM8 scripts/slurm_main6_extract_validate_trainvalid.sh)
TRAIN=$(sbatch --parsable --dependency=afterok:$CACHE scripts/slurm_main6_downstream_valid.sh)
SELECT=$(sbatch --parsable --dependency=afterok:$TRAIN scripts/slurm_main6_select_valid.sh)
echo "base=$BASE continue=$CONT mlm8=$MLM8 cache=$CACHE train=$TRAIN select=$SELECT"
```

The cache, training, and selection jobs are validation-only:

```bash
test ! -e data/features/main6_random_mlm8_trainvalid/test.pt
```

These jobs generate only train/valid caches and metrics. Review
`results/main6_random_090_mlm8/validation_selection.json`, frozen thresholds, and
per-label validation F1 after `SELECT` completes. The test cache and final
evaluation are intentionally blocked until that review. Then submit exactly
one final job:

```bash
sbatch scripts/slurm_main6_final_test.sh
```
