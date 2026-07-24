# EVEF-MVD Main-6

Opcode-only smart-contract multi-label vulnerability detection for DIVE Main-6.

Current route:

```text
19,143 unique public runtime-opcode contracts
+ DIVE_main6_random_split train only
-> EVM-BERT MLM + 17-role multi-label ETP pretraining
-> 64-chunk feature and token-level Top-2 ETP caches
-> label-decoupled ETP cross-attention MIL
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
`afterok`, and all ETP pretraining inputs are public 19,143 plus Main-6 train only:

```bash
mkdir -p logs
BASE=$(sbatch --parsable scripts/slurm_main6_base_19143.sh)
CONT=$(sbatch --parsable --dependency=afterok:$BASE scripts/slurm_main6_19143_continue.sh)
ETP=$(sbatch --parsable --dependency=afterok:$CONT scripts/slurm_main6_multirole_etp_pretrain.sh)
CACHE=$(sbatch --parsable --dependency=afterok:$ETP scripts/slurm_main6_extract_validate_trainvalid.sh)
TRAIN=$(sbatch --parsable --dependency=afterok:$CACHE scripts/slurm_main6_downstream_valid.sh)
SELECT=$(sbatch --parsable --dependency=afterok:$TRAIN scripts/slurm_main6_select_valid.sh)
echo "base=$BASE continue=$CONT etp=$ETP cache=$CACHE train=$TRAIN select=$SELECT"
```

The cache, training, and selection jobs are validation-only:

```bash
test ! -e data/features/main6_random_multirole_etp/test.pt
test ! -e data/features/main6_random_multirole_etp_tokens/test.pt
```

These jobs generate only train/valid caches and metrics. Review
`results/main6_random_090/validation_selection.json`, frozen thresholds, and
per-label validation F1 after `SELECT` completes. The test cache and final
evaluation are intentionally blocked until that review. Then submit exactly
one final job:

```bash
sbatch scripts/slurm_main6_final_test.sh
```
