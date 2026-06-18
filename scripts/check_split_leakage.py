import argparse
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data/processed/BJUT_SC01"
REPORT_TXT = PROJECT_ROOT / "data/reports/split_leakage_report.txt"
REPORT_JSON = PROJECT_ROOT / "data/reports/split_leakage_report.json"
ADDRESS_KEYS = ["id", "address", "contract_address"]


def parse_args():
    parser = argparse.ArgumentParser(description="Check opcode/address overlap across splits.")
    parser.add_argument("--data_dir", default=str(DATA_DIR.relative_to(PROJECT_ROOT)))
    parser.add_argument("--output", default=str(REPORT_TXT.relative_to(PROJECT_ROOT)))
    return parser.parse_args()


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def opcode_hash(opcode):
    normalized = " ".join(str(opcode).split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_split(path):
    hashes = set()
    addresses = set()
    samples = 0
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            opcode = item.get("opcode")
            if opcode is None:
                raise ValueError(f"{path}:{line_no} missing opcode")
            hashes.add(opcode_hash(opcode))
            for key in ADDRESS_KEYS:
                value = item.get(key)
                if value is not None and str(value).strip():
                    addresses.add(str(value).strip().lower())
                    break
            samples += 1
    return {"samples": samples, "opcode_hashes": hashes, "addresses": addresses}


def overlap_count(left, right, key):
    return len(left[key] & right[key])


def write_reports(report, txt_path):
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = txt_path.with_suffix(".json")
    lines = ["BJUT SC01 split leakage report", ""]
    for key, value in report.items():
        lines.append(f"{key}: {value}")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return txt_path, json_path


def main():
    args = parse_args()
    data_dir = resolve(args.data_dir)
    output = resolve(args.output)
    split_paths = {
        "train": data_dir / "train.jsonl",
        "valid": data_dir / "valid.jsonl",
        "test": data_dir / "test.jsonl",
    }
    splits = {}
    for name, path in split_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Split file not found: {path}")
        print(f"[CHECK] {name}: {path.relative_to(PROJECT_ROOT)}")
        splits[name] = load_split(path)

    train_valid_opcode = overlap_count(splits["train"], splits["valid"], "opcode_hashes")
    train_test_opcode = overlap_count(splits["train"], splits["test"], "opcode_hashes")
    valid_test_opcode = overlap_count(splits["valid"], splits["test"], "opcode_hashes")
    train_valid_address = overlap_count(splits["train"], splits["valid"], "addresses")
    train_test_address = overlap_count(splits["train"], splits["test"], "addresses")
    valid_test_address = overlap_count(splits["valid"], splits["test"], "addresses")

    report = {
        "data_dir": data_dir.relative_to(PROJECT_ROOT).as_posix(),
        "train_samples": splits["train"]["samples"],
        "valid_samples": splits["valid"]["samples"],
        "test_samples": splits["test"]["samples"],
        "train_valid_opcode_hash_overlap": train_valid_opcode,
        "train_test_opcode_hash_overlap": train_test_opcode,
        "valid_test_opcode_hash_overlap": valid_test_opcode,
        "train_valid_address_overlap": train_valid_address,
        "train_test_address_overlap": train_test_address,
        "valid_test_address_overlap": valid_test_address,
        "leakage_status": "warning" if train_test_opcode > 0 else "ok",
    }
    txt_path, json_path = write_reports(report, output)
    print(f"[OK] wrote {txt_path.relative_to(PROJECT_ROOT)}")
    print(f"[OK] wrote {json_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
