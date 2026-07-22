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

Activate the server environment, then execute the stages in order:

```bash
conda activate correlascan_a40
bash scripts/run_main6_random_090.sh audit
bash scripts/run_main6_random_090.sh corpus
bash scripts/run_main6_random_090.sh pretrain
bash scripts/run_main6_random_090.sh extract
bash scripts/run_main6_random_090.sh validate
bash scripts/run_main6_random_090.sh train
bash scripts/run_main6_random_090.sh select
```

The final test is intentionally blocked. After reviewing the validation-only
selection, run `ALLOW_TEST=1 bash scripts/run_main6_random_090.sh final`.
