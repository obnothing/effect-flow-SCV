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

## DIVE Source-Main6 GraphCodeBERT

This independent route uses only source files that can be runtime-aligned to
the six-label Main-6 records. It is a new seed-42 source-grouped split and is
not comparable as a numeric continuation of the opcode-only result.

Transfer `DIVE_Raw_Data/Raw` to the project root, install `requirements.txt`,
then fetch the pinned base asset once on a login node:

```bash
python scripts/fetch_graphcodebert_asset.py \
  --revision 2b0488a7bb0eefc7041f1bb2cad1ab26b0da269d
```

Submit validation-only work in dependency order:

```bash
PREP=$(sbatch --parsable scripts/slurm_dive_source_main6_prepare.sh)
DAPT=$(sbatch --parsable --dependency=afterok:$PREP scripts/slurm_dive_source_main6_dapt.sh)
TRAIN=$(sbatch --parsable --dependency=afterok:$DAPT scripts/slurm_dive_source_main6_train.sh)
SELECT=$(sbatch --parsable --dependency=afterok:$TRAIN scripts/slurm_dive_source_main6_select.sh)
echo "prepare=$PREP dapt=$DAPT train=$TRAIN select=$SELECT"
```

Review `results/dive_source_main6/validation_selection.json` and the selected
candidate's valid thresholds before submitting `scripts/slurm_dive_source_main6_final.sh`.
That final job is the only path that creates Source-Main6 test artifacts.
