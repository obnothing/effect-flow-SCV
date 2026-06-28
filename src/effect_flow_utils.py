import hashlib
import json
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "data" / "reports"

DATASET_SPECS = {
    "BJUT": {
        "strict_dir": "data/processed/BJUT_SC01",
        "random_dir": "data/processed/BJUT_SC01_random_split",
        "output_dir": "data/processed/effect_flow_pretrain/BJUT",
        "label_names": [
            "Reentrancy",
            "Contract contains unknown address",
            "Integer overflow or underflow",
            "Timestamp dependence",
            "DoS with failed call",
            "Assert violation",
            "Unchecked call return value",
            "Unsafe send",
            "Multiplication after division",
            "Extra gas consumption",
        ],
    },
    "DIVE": {
        "strict_dir": "data/processed/DIVE",
        "random_dir": "data/processed/DIVE_random_split",
        "output_dir": "data/processed/effect_flow_pretrain/DIVE",
        "label_names": [
            "Reentrancy",
            "Access Control",
            "Arithmetic",
            "Unchecked Return Values",
            "DoS",
            "Bad Randomness",
            "Front Running",
            "Time manipulation",
        ],
    },
}

GLOBAL_VULNERABILITY_LABELS = []
for _dataset_name in ("BJUT", "DIVE"):
    for _label_name in DATASET_SPECS[_dataset_name]["label_names"]:
        if _label_name not in GLOBAL_VULNERABILITY_LABELS:
            GLOBAL_VULNERABILITY_LABELS.append(_label_name)


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def relative(path):
    path = Path(path)
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def iter_jsonl(path):
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if line:
                yield line_no, json.loads(line)


def count_jsonl(path):
    if not Path(path).exists():
        return 0
    with Path(path).open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def opcode_hash(opcode):
    return hashlib.sha256(str(opcode).encode("utf-8")).hexdigest()


def percentile(values, fraction):
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(ordered[low])
    return float(
        ordered[low] + (ordered[high] - ordered[low]) * (position - low)
    )


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(path, lines):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = lines if isinstance(lines, str) else "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")


def strict_split_paths(dataset):
    spec = DATASET_SPECS[dataset]
    root = resolve(spec["strict_dir"])
    return {split: root / f"{split}.jsonl" for split in ("train", "valid", "test")}


def corpus_split_paths(dataset):
    root = resolve(DATASET_SPECS[dataset]["output_dir"])
    return {
        split: root / f"{split}_effect_flow_chunks.jsonl"
        for split in ("train", "valid", "test")
    }


def infer_fields(record):
    fields = list(record)
    id_field = next((name for name in ("id", "address", "contract_id") if name in record), None)
    opcode_field = next(
        (name for name in ("opcode", "opcodes", "opcode_sequence") if name in record),
        None,
    )
    label_fields = [
        name
        for name in fields
        if name in {"binary_label", "multi_labels", "labels"}
        or "label" in name.lower()
    ]
    return fields, id_field, opcode_field, label_fields
