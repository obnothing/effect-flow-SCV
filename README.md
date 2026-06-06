# CorrelaScan Reproduction

This repository currently contains the stage-1 minimal runnable skeleton for a
CorrelaScan reproduction.

## Project Structure

```text
reproduc_corre_scan/
+-- configs/
|   +-- config.yaml
+-- src/
|   +-- model.py
|   +-- dataset.py
|   +-- train.py
|   +-- metrics.py
|   +-- utils.py
+-- scripts/
|   +-- train.sh
+-- requirements.txt
+-- README.md
```

## Data Format

Prepare the following JSONL files:

```text
data/processed/train.jsonl
data/processed/valid.jsonl
data/processed/test.jsonl
```

Each line should be one sample:

```json
{"opcode":"PUSH1 MSTORE CALLVALUE ...","binary_label":1,"multi_labels":[1,0,0,0,0,0,0,0,0,0]}
```

## Encoder Selection

The default shared encoder is:

```yaml
model_name: microsoft/codebert-base
```

You can switch `model_name` in `configs/config.yaml` to:

```yaml
model_name: microsoft/graphcodebert-base
```

or:

```yaml
model_name: huggingface/CodeBERTa-small-v1
```

## Run

```bash
cd reproduc_corre_scan
pip install -r requirements.txt
bash scripts/train.sh
```

The best checkpoint is saved to:

```text
checkpoints/best.pt
```

## Stage 2: BJUT SC01 Dataset Preparation

Download the dataset:

```bash
python scripts/download_bjut_sc01.py
```

Inspect the raw files:

```bash
python scripts/inspect_bjut_sc01.py
```

Preprocess the dataset:

```bash
python src/preprocess_bjut_sc01.py --config configs/config.yaml
```

Check the outputs:

```bash
ls data/processed/BJUT_SC01
cat data/reports/bjut_sc01_processed_report.txt
```
