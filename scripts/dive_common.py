import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "data" / "reports"
DIVE_LABEL_NAMES = [
    "Reentrancy",
    "Access Control",
    "Arithmetic",
    "Unchecked Return Values",
    "DoS",
    "Bad Randomness",
    "Front Running",
    "Time manipulation",
]


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_inventory():
    path = REPORT_DIR / "dive_file_inventory.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def choose_inventory_file(role):
    inventory = load_inventory()
    candidates = [
        item for item in inventory.get("files", []) if item.get("inferred_role") == role
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda item: item["size_bytes"], reverse=True)
    return resolve(candidates[0]["path"])


def default_opcode_path():
    return choose_inventory_file("runtime_opcode_jsonl") or (
        PROJECT_ROOT / "DIVE/DIVE_Raw_Data/Raw/POST/Runtime_Opcode.jsonl"
    )


def default_label_path():
    return choose_inventory_file("label_csv") or (
        PROJECT_ROOT / "DIVE/DIVE_Labels/Labels/DIVE_Labels.csv"
    )


def parse_binary(value):
    text = str(value).strip().lower()
    return 1 if text in {"1", "1.0", "true", "yes"} else 0


def load_labels(path):
    path = resolve(path)
    labels = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = [name for name in ["contractID", *DIVE_LABEL_NAMES] if name not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path} missing columns: {missing}")
        for row in reader:
            contract_id = str(row["contractID"]).strip()
            labels[contract_id] = [parse_binary(row[name]) for name in DIVE_LABEL_NAMES]
    return labels


def iter_opcodes(path):
    path = resolve(path)
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            contract_id = str(item.get("contractID", item.get("id", ""))).strip()
            opcode = item.get("Opcodes", item.get("opcode", item.get("opcodes")))
            if not contract_id or opcode is None:
                raise ValueError(f"{path}:{line_no} missing contractID/Opcodes")
            yield contract_id, str(opcode)


def opcode_hash(opcode):
    normalized = " ".join(str(opcode).split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def percentile(sorted_values, p):
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * (p / 100.0)
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return float(sorted_values[lower])
    frac = pos - lower
    return float(sorted_values[lower] * (1 - frac) + sorted_values[upper] * frac)


def summarize_values(values):
    values = sorted(int(v) for v in values)
    if not values:
        return {
            "count": 0,
            "min": 0,
            "mean": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0,
        }
    return {
        "count": len(values),
        "min": int(values[0]),
        "mean": float(sum(values) / len(values)),
        "median": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": int(values[-1]),
    }


def chunk_coverage(length, chunk_content_size, stride, max_chunks):
    length = int(length)
    if length <= 0:
        return {
            "chunks_needed": 1,
            "chunks_kept": 1,
            "coverage_ratio": 1.0,
            "truncated": False,
        }
    chunks_needed = max(1, math.ceil(max(0, length - chunk_content_size) / stride) + 1)
    chunks_kept = min(chunks_needed, max_chunks)
    covered = min(length, (chunks_kept - 1) * stride + chunk_content_size)
    return {
        "chunks_needed": int(chunks_needed),
        "chunks_kept": int(chunks_kept),
        "coverage_ratio": float(covered / length),
        "truncated": chunks_needed > max_chunks,
    }


def write_json_txt(report, json_path, txt_path, title):
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [title, ""]
    for key, value in report.items():
        if isinstance(value, (dict, list)):
            lines.append(f"{key}:")
            lines.append(json.dumps(value, indent=2))
        else:
            lines.append(f"{key}: {value}")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote {txt_path.relative_to(PROJECT_ROOT)}")
    print(f"[OK] wrote {json_path.relative_to(PROJECT_ROOT)}")
