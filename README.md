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

The historical opcode-only route is immutable for comparison. A separate
`DIVE Main6 Opcode-CSDG` route adds opcode-native basic-block CFG and
conservative stack def-use edges as a residual graph branch over the frozen
MLM8 sequence baseline. Its caches, checkpoints, and metrics are stored under
`main6_opcode_csdg` and must not be mixed with the historical route or the
Source-Main6 GraphCodeBERT route.

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
then install the Python 3.8 Solidity parser and fetch the pinned base asset
once on a login node. The parser installer requires the fixed ABI-14 grammar
archive shown below:

```bash
python -m pip install -r requirements.txt
bash scripts/install_source_main6_dependencies.sh \
  third_party_wheels/tree-sitter-solidity-v1.2.2.tar.gz

python scripts/fetch_graphcodebert_asset.py \
  --revision 2b0488a7bb0eefc7041f1bb2cad1ab26b0da269d
```

The grammar archive is platform-independent and pinned by SHA256 in the
installer. On a compute server without GitHub/PyPI access, transfer this exact
archive from a networked machine:

```bash
curl -L https://github.com/JoranHonig/tree-sitter-solidity/archive/refs/tags/v1.2.2.tar.gz \
  -o third_party_wheels/tree-sitter-solidity-v1.2.2.tar.gz
bash scripts/install_source_main6_dependencies.sh \
  third_party_wheels/tree-sitter-solidity-v1.2.2.tar.gz
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

## DIVE Main6 Opcode-CSDG

This route uses the same six-label random split and the same train-only
continued MLM checkpoint as the opcode baseline. It builds generic EVM
control-flow and conservative stack def-use graphs, then adds a zero-initialized
label-conditioned graph residual to the frozen `mlm8_slot3` predictor. It does
not use source code, vulnerability templates, label retrieval, ETP, ASL,
MLSMOTE, or policy-gradient reinforcement learning.

## DIVE Main6 Opcode-Execution-Aware

This is an isolated route. It reuses the existing train/valid MLM8
sequence cache and derives label-agnostic execution summaries from EVM stack
semantics: producer-consumer lineage, pop/push effects, stack-height buckets,
opcode roles, and value provenance. A gated execution branch is fused into a
multi-slot MIL model. It does not use Solidity source, AST/CFG graphs, ETP,
vulnerability templates, label retrieval, ASL, MLSMOTE, or policy-gradient
reinforcement learning. Artifacts are stored under
`data/features/main6_opcode_execution_aware`,
`checkpoints/main6_opcode_execution_aware`, and
`results/main6_opcode_execution_aware`.

Run after the existing MLM8 sequence cache is present:

```bash
python scripts/extract_main6_execution_features.py \
  --config configs/train_main6_execution_aware_mil.yaml --splits train valid
python scripts/check_main6_execution_cache.py
python scripts/train_main6_execution_aware_mil.py \
  --config configs/train_main6_execution_aware_mil.yaml
python scripts/select_main6_execution_aware_valid.py
python scripts/diagnose_main6_execution_aware_valid.py
```

The default path never creates test execution features or test predictions.
The diagnostic writes validation-only PR-AUC, fixed-0.5 F1, threshold history,
probability distributions, Brier score, ECE, and TP/FP/FN to
`results/main6_opcode_execution_aware/valid_calibration_diagnostics.json`.

## DIVE Main6 Opcode-Stack-Relational

This independent route injects EVM stack execution relations directly into
the BERT input and the last four self-attention layers. It does not create a
post-BERT stack branch and does not use a GNN. The relation representation
keeps producer-consumer value lineage, operand slots, `DUP` aliases, `SWAP`
reordering, distance buckets, and conservative unknown states. The route uses
the same six-label seed-42 split and train-only continued MLM, with artifacts
under `data/features/main6_opcode_stack_relational`,
`checkpoints/main6_opcode_stack_relational`, and
`results/main6_opcode_stack_relational`.

Run the validation-only route in order:

```bash
bash scripts/run_main6_stack_relational.sh extract
bash scripts/run_main6_stack_relational.sh pretrain
bash scripts/run_main6_stack_relational.sh train
bash scripts/run_main6_stack_relational.sh select
```

If the train/valid relation cache has already been extracted and checked, the
remaining stages can be submitted as dependent Slurm jobs. The chain requests
one GPU for pretraining, one GPU for downstream training, and CPU resources for
validation-only selection:

```bash
bash scripts/submit_main6_stack_relational.sh
```

The relation cache is ragged and sparse; no dense 512x512 cache is written.
The default path does not read test labels, create a test cache, or generate
test predictions. Do not compare this route's eventual test result directly
with another isolated route without reporting the route name and protocol.

## Four Vulnerability Dataset

The repository also contains an independent four-label dataset with
Delegatecall, Integer Overflow/Underflow, Reentrancy, and Timestamp
Dependence. Its original files are converted into contract-ID grouped JSONL;
the route is isolated from every Main-6 experiment.

Run locally or on a server after the dataset directory is present:

```bash
bash scripts/run_four_vulnerability_opcode.sh audit
bash scripts/run_four_vulnerability_opcode.sh extract
bash scripts/run_four_vulnerability_opcode.sh validate
bash scripts/run_four_vulnerability_opcode.sh train
bash scripts/run_four_vulnerability_opcode.sh select
```

For Slurm:

```bash
AUDIT=$(sbatch --parsable scripts/slurm_four_vulnerability_audit.slurm)
EXTRACT=$(sbatch --parsable --dependency=afterok:$AUDIT scripts/slurm_four_vulnerability_extract.slurm)
TRAIN=$(sbatch --parsable --dependency=afterok:$EXTRACT scripts/slurm_four_vulnerability_train.slurm)
SELECT=$(sbatch --parsable --dependency=afterok:$TRAIN scripts/slurm_four_vulnerability_select_valid.slurm)
echo "AUDIT=$AUDIT EXTRACT=$EXTRACT TRAIN=$TRAIN SELECT=$SELECT"
```

The route uses only train/valid caches by default. Review
`data/processed/four_vulnerability_random_split/audit_report.json` and
`results/four_vulnerability_opcode_mlm8/validation_selection.json`; no test
artifact is created by these commands.

Run audit, extraction, validation training, and selection in order:

```bash
AUDIT=$(sbatch --parsable scripts/slurm_main6_opcode_csdg_audit.slurm)
EXTRACT=$(sbatch --parsable --dependency=afterok:$AUDIT scripts/slurm_main6_opcode_csdg_extract.slurm)
TRAIN=$(sbatch --parsable --dependency=afterok:$EXTRACT scripts/slurm_main6_opcode_csdg_train_valid.slurm)
SELECT=$(sbatch --parsable --dependency=afterok:$TRAIN scripts/slurm_main6_opcode_csdg_select_valid.slurm)
echo "audit=$AUDIT extract=$EXTRACT train=$TRAIN select=$SELECT"
```

Review `results/main6_opcode_csdg/selection.json`. Only the selected
validation candidate may create a test cache, and only one final test job may
be submitted after manual approval.
