import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data/processed/BJUT_SC01"
REPORT_TXT = PROJECT_ROOT / "data/reports/split_leakage_report.txt"
REPORT_JSON = PROJECT_ROOT / "data/reports/split_leakage_report.json"
ADDRESS_KEYS = ["id", "address", "contract_address"]


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


def write_reports(report):
    REPORT_TXT.parent.mkdir(parents=True, exist_ok=True)
    lines = ["BJUT SC01 split leakage report", ""]
    for key, value in report.items():
        lines.append(f"{key}: {value}")
    REPORT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main():
    split_paths = {
        "train": DATA_DIR / "train.jsonl",
        "valid": DATA_DIR / "valid.jsonl",
        "test": DATA_DIR / "test.jsonl",
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
    write_reports(report)
    print(f"[OK] wrote {REPORT_TXT.relative_to(PROJECT_ROOT)}")
    print(f"[OK] wrote {REPORT_JSON.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
