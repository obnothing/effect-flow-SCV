# EVEF-MVD Main-6

Opcode-only smart-contract multi-label vulnerability detection for DIVE Main-6.

Current route:

```text
19,143 unique public runtime-opcode contracts
+ DIVE_main6_random_split train only
-> EVM-BERT MLM and Effect-Flow pretraining
-> 64-chunk feature and semantic caches
-> six-label evidence-guided MIL / MultiScale MIL
```

Labels: Reentrancy, Access Control, Arithmetic, Unchecked Return Values, DoS,
and Time manipulation.

## Data

Datasets are not committed. Transfer these directories to the server before
running:

```text
data/processed/DIVE_main6_access4000_clean3952
data/processed/DIVE_main6_random_split
data/processed/ethereum_public_pretrain_19143_unique_runtime
```

## Server Run

After public MLM and Effect-Flow pretraining complete, submit the validation-only
stages from the project root:

```bash
mkdir -p logs
CACHE=$(sbatch --parsable scripts/slurm_main6_extract_validate_trainvalid.sh)
TRAIN=$(sbatch --parsable --dependency=afterok:$CACHE scripts/slurm_main6_downstream_valid.sh)
SELECT=$(sbatch --parsable --dependency=afterok:$TRAIN scripts/slurm_main6_select_valid.sh)
echo "cache=$CACHE train=$TRAIN select=$SELECT"
```

These jobs generate only train/valid caches and metrics. Review
`results/main6_random_090/validation_selection.json`, frozen thresholds, and
per-label validation F1 after `SELECT` completes. The test cache and final
evaluation are intentionally blocked until that review. Then submit exactly
one final job:

```bash
sbatch scripts/slurm_main6_final_test.sh
```
