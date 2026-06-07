import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evm_tokenizer import EVMOpcodeTokenizer  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze EVM tokenizer lengths.")
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


def iter_opcodes(path):
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            opcode = json.loads(line).get("opcode")
            if opcode is None:
                raise ValueError(f"{path}:{line_no} missing opcode")
            yield opcode


def summarize(lengths):
    values = np.asarray(lengths, dtype=np.int64)
    if values.size == 0:
        return {
            "samples": 0,
            "min": 0,
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0,
            "over_128_ratio": 0.0,
            "over_256_ratio": 0.0,
            "over_512_ratio": 0.0,
            "over_1024_ratio": 0.0,
            "over_2048_ratio": 0.0,
        }
    return {
        "samples": int(values.size),
        "min": int(values.min()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": int(values.max()),
        "over_128_ratio": float((values > 128).mean()),
        "over_256_ratio": float((values > 256).mean()),
        "over_512_ratio": float((values > 512).mean()),
        "over_1024_ratio": float((values > 1024).mean()),
        "over_2048_ratio": float((values > 2048).mean()),
    }


def analyze_file(path, tokenizer):
    lengths = []
    for opcode in tqdm(iter_opcodes(path), desc=f"evm_lengths:{path.name}"):
        lengths.append(len(tokenizer.tokenize(opcode, add_special_tokens=True)))
    return summarize(lengths)


def write_reports(report):
    txt_path = resolve_project_path("data/reports/evm_opcode_length_report.txt")
    json_path = resolve_project_path("data/reports/evm_opcode_length_report.json")
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["EVM tokenizer opcode length report", ""]
    for split, values in report.items():
        lines.append(f"[{split}]")
        for key, value in values.items():
            lines.append(f"{key}: {value}")
        lines.append("")
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    args = parse_args()
    config = load_config(args.config)
    vocab_path = resolve_project_path(config["vocab_path"])
    if not vocab_path.exists():
        raise FileNotFoundError(
            f"EVM vocab not found: {vocab_path}. Run scripts/build_evm_vocab.py first."
        )
    tokenizer = EVMOpcodeTokenizer.from_vocab_file(vocab_path)
    split_paths = {
        "train": resolve_project_path(config["train_path"]),
        "valid": resolve_project_path(config["valid_path"]),
        "test": resolve_project_path(config["test_path"]),
    }
    report = {}
    for split, path in split_paths.items():
        print(f"[ANALYZE] {split}: {path}")
        report[split] = analyze_file(path, tokenizer)
    write_reports(report)
    print("[OK] wrote data/reports/evm_opcode_length_report.txt")
    print("[OK] wrote data/reports/evm_opcode_length_report.json")


if __name__ == "__main__":
    main()
