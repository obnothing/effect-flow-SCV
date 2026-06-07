import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze tokenized opcode lengths.")
    parser.add_argument(
        "--config",
        default="configs/train_mlsmote_codeberta_server.yaml",
        help="Training config used to load tokenizer and data paths.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid", "test"],
        choices=["train", "train_mlsmote", "valid", "test"],
        help="Dataset splits to analyze.",
    )
    parser.add_argument(
        "--report-path",
        default="data/reports/opcode_length_report.txt",
        help="Output text report path.",
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


def split_to_path(data_dir, split, use_mlsmote_train):
    if split == "train":
        file_name = "train_mlsmote.jsonl" if use_mlsmote_train else "train.jsonl"
    elif split == "train_mlsmote":
        file_name = "train_mlsmote.jsonl"
    else:
        file_name = f"{split}.jsonl"
    return data_dir / file_name


def iter_opcodes(path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            opcode = item.get("opcode")
            if opcode is None:
                raise ValueError(f"{path}:{line_no} missing opcode")
            yield opcode


def summarize_lengths(lengths):
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
    }


def analyze_split(path, tokenizer):
    lengths = []
    for opcode in tqdm(iter_opcodes(path), desc=f"lengths:{path.name}"):
        token_count = len(tokenizer.tokenize(opcode))
        lengths.append(token_count + tokenizer.num_special_tokens_to_add(pair=False))
    return summarize_lengths(lengths)


def write_report(path, config, split_reports):
    path = resolve_project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Opcode tokenized length report",
        "",
        f"model_name: {config['model_name']}",
        f"data_dir: {config['data_dir']}",
        "",
    ]
    for split, report in split_reports.items():
        lines.append(f"[{split}]")
        for key, value in report.items():
            lines.append(f"{key}: {value}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    config = load_config(args.config)
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"],
        local_files_only=config.get("local_files_only", True),
    )
    data_dir = resolve_project_path(config["data_dir"])
    use_mlsmote_train = bool(config.get("use_mlsmote_train", False))

    split_reports = {}
    for split in args.splits:
        split_path = split_to_path(data_dir, split, use_mlsmote_train)
        if not split_path.exists():
            raise FileNotFoundError(f"Dataset split not found: {split_path}")
        print(f"[ANALYZE] {split}: {split_path}")
        split_reports[split] = analyze_split(split_path, tokenizer)

    write_report(args.report_path, config, split_reports)
    print(f"[OK] wrote {args.report_path}")


if __name__ == "__main__":
    main()
