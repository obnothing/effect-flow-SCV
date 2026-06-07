import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import yaml
from transformers import AutoTokenizer
from transformers.utils import logging

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
logging.set_verbosity_error()

from evm_tokenizer import EVMOpcodeTokenizer  # noqa: E402


EXAMPLE_TOKENS = [
    "PUSH1",
    "PUSH20",
    "PUSH32",
    "0x80",
    "0x40",
    "0x20",
    "0x00",
    "CALLVALUE",
    "CALLDATALOAD",
    "CALLDATASIZE",
    "DELEGATECALL",
    "SELFDESTRUCT",
    "TIMESTAMP",
    "SSTORE",
    "JUMPI",
    "0xffffffff",
    "0x1234567890abcdef1234567890abcdef12345678",
    "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Compare tokenizer fragmentation.")
    parser.add_argument(
        "--config",
        default="configs/evm_tokenizer.yaml",
        help="Path to EVM tokenizer YAML config.",
    )
    return parser.parse_args()


def resolve_project_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_config(path):
    with resolve_project_path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_samples(path, sample_size, seed=42):
    samples = []
    rng = random.Random(seed)
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            opcode = json.loads(line).get("opcode")
            if opcode is None:
                raise ValueError(f"{path}:{line_no} missing opcode")
            if len(samples) < sample_size:
                samples.append(opcode)
            else:
                idx = rng.randint(0, line_no - 1)
                if idx < sample_size:
                    samples[idx] = opcode
    return samples


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(values.max()),
    }


def analyze_samples(samples, code_tokenizer, evm_tokenizer):
    whitespace_counts = []
    code_counts = []
    evm_counts = []
    for opcode in samples:
        whitespace_count = max(len(opcode.split()), 1)
        code_count = len(
            code_tokenizer(
                opcode,
                add_special_tokens=True,
                truncation=False,
                padding=False,
                return_attention_mask=False,
            )["input_ids"]
        )
        evm_count = len(evm_tokenizer.tokenize(opcode, add_special_tokens=True))
        whitespace_counts.append(whitespace_count)
        code_counts.append(code_count)
        evm_counts.append(evm_count)

    whitespace = np.asarray(whitespace_counts, dtype=np.float64)
    code = np.asarray(code_counts, dtype=np.float64)
    evm = np.asarray(evm_counts, dtype=np.float64)
    return {
        "sample_count": len(samples),
        "whitespace_opcode_token_count": summarize(whitespace),
        "codeberta_token_count": summarize(code),
        "evm_token_count": summarize(evm),
        "codeberta_fragmentation_ratio": summarize(code / whitespace),
        "evm_tokenizer_ratio": summarize(evm / whitespace),
    }


def token_examples(code_tokenizer, evm_tokenizer):
    examples = {}
    for token in EXAMPLE_TOKENS:
        examples[token] = {
            "codeberta_tokens": code_tokenizer.tokenize(token),
            "evm_tokens": evm_tokenizer.tokenize(token, add_special_tokens=False),
        }
    return examples


def write_reports(report):
    txt_path = resolve_project_path("data/reports/tokenizer_fragmentation_report.txt")
    json_path = resolve_project_path("data/reports/tokenizer_fragmentation_report.json")
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["Tokenizer fragmentation report", ""]
    for key in [
        "sample_count",
        "whitespace_opcode_token_count",
        "codeberta_token_count",
        "evm_token_count",
        "codeberta_fragmentation_ratio",
        "evm_tokenizer_ratio",
    ]:
        lines.append(f"{key}: {report[key]}")
    lines.append("")
    lines.append("Token examples:")
    for token, examples in report["token_examples"].items():
        lines.append(f"- {token}")
        lines.append(f"  CodeBERTa: {examples['codeberta_tokens']}")
        lines.append(f"  EVM: {examples['evm_tokens']}")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    args = parse_args()
    config = load_config(args.config)
    train_path = resolve_project_path(config["train_path"])
    vocab_path = resolve_project_path(config["vocab_path"])
    if not vocab_path.exists():
        raise FileNotFoundError(
            f"EVM vocab not found: {vocab_path}. Run scripts/build_evm_vocab.py first."
        )
    samples = load_samples(
        train_path,
        int(config.get("sample_size_for_fragmentation", 1000)),
        seed=42,
    )
    code_tokenizer = AutoTokenizer.from_pretrained(
        config["model_name_for_comparison"],
        local_files_only=True,
    )
    evm_tokenizer = EVMOpcodeTokenizer.from_vocab_file(vocab_path)
    report = analyze_samples(samples, code_tokenizer, evm_tokenizer)
    report["token_examples"] = token_examples(code_tokenizer, evm_tokenizer)
    write_reports(report)
    print("[OK] wrote data/reports/tokenizer_fragmentation_report.txt")
    print("[OK] wrote data/reports/tokenizer_fragmentation_report.json")


if __name__ == "__main__":
    main()
