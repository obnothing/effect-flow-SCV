import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "BJUT_SC01" / "extracted"
REPORT_DIR = PROJECT_ROOT / "data" / "reports"
TXT_REPORT = REPORT_DIR / "bjut_sc01_raw_report.txt"
JSON_REPORT = REPORT_DIR / "bjut_sc01_raw_report.json"
LABEL_VALUE_TXT_REPORT = REPORT_DIR / "bjut_sc01_label_value_report.txt"
LABEL_VALUE_JSON_REPORT = REPORT_DIR / "bjut_sc01_label_value_report.json"

SUPPORTED_EXTENSIONS = {".csv", ".json", ".jsonl", ".txt", ".xlsx"}
BYTECODE_CANDIDATES = {"contractCode", "bytecode", "runtime_bytecode", "creationCode"}
OPCODE_CANDIDATES = {"opcode", "opcodes", "opcode_sequence"}
ADDRESS_CANDIDATES = {"address", "contract_address"}
LABEL_KEYWORDS = [
    "vulnerability",
    "label",
    "bug",
    "defect",
    "reentrancy",
    "overflow",
    "underflow",
    "timestamp",
    "dos",
    "assert",
    "unchecked",
    "unsafe",
    "send",
    "gas",
    "unknown",
]
PAPER_LABEL_NAMES = [
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
]


def parse_label_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        if float(value) == 1.0:
            return 1
        if float(value) == 0.0:
            return 0
        return None

    text = str(value).strip()
    if text == "":
        return None
    lower = text.lower()
    if lower in {"1", "1.0", "true"}:
        return 1
    if lower in {"0", "0.0", "false"}:
        return 0
    return None


def file_size(path):
    size = path.stat().st_size
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def read_table(path):
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        try:
            return pd.read_json(path)
        except ValueError:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return pd.DataFrame(data)
            if isinstance(data, dict):
                for value in data.values():
                    if isinstance(value, list):
                        return pd.DataFrame(value)
                return pd.DataFrame([data])
            raise
    if suffix == ".xlsx":
        return pd.read_excel(path)
    if suffix == ".txt":
        try:
            return pd.read_csv(path)
        except Exception:
            try:
                return pd.read_csv(path, sep="\t")
            except Exception:
                rows = path.read_text(encoding="utf-8", errors="ignore").splitlines()
                return pd.DataFrame({"text": rows})
    raise ValueError(f"Unsupported file type: {path}")


def is_binary_label_series(series):
    values = series.dropna().unique()
    if len(values) == 0:
        return False
    normalized = {str(value).strip().lower() for value in values}
    return normalized.issubset({"0", "1", "0.0", "1.0", "false", "true"})


def inspect_file(path):
    info = {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "size": file_size(path),
        "rows": None,
        "columns": [],
        "bytecode_fields": [],
        "opcode_fields": [],
        "address_fields": [],
        "label_fields": [],
        "preview": [],
        "error": None,
    }
    try:
        df = read_table(path)
        info["rows"] = int(len(df))
        info["columns"] = [str(col) for col in df.columns]
        lower_columns = {str(col).lower(): str(col) for col in df.columns}

        info["bytecode_fields"] = [
            original
            for lower, original in lower_columns.items()
            if lower in {field.lower() for field in BYTECODE_CANDIDATES}
        ]
        info["opcode_fields"] = [
            original
            for lower, original in lower_columns.items()
            if lower in {field.lower() for field in OPCODE_CANDIDATES}
        ]
        info["address_fields"] = [
            original
            for lower, original in lower_columns.items()
            if lower in {field.lower() for field in ADDRESS_CANDIDATES}
        ]

        label_fields = []
        for column in df.columns:
            name = str(column)
            lower = name.lower()
            if is_binary_label_series(df[column]) or any(
                keyword in lower for keyword in LABEL_KEYWORDS
            ):
                label_fields.append(name)
        info["label_fields"] = label_fields
        info["preview"] = df.head(3).fillna("").astype(str).to_dict(orient="records")
    except Exception as exc:
        info["error"] = str(exc)
    return info


def write_reports(reports):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_REPORT.write_text(
        json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = ["BJUT SC01 raw data inspection report", ""]
    for item in reports["files"]:
        lines.append(f"File: {item['path']}")
        lines.append(f"Size: {item['size']}")
        if item["error"]:
            lines.append(f"Error: {item['error']}")
            lines.append("")
            continue
        lines.append(f"Rows/Samples: {item['rows']}")
        lines.append(f"Columns: {', '.join(item['columns'])}")
        lines.append(f"Bytecode candidates: {', '.join(item['bytecode_fields']) or '-'}")
        lines.append(f"Opcode candidates: {', '.join(item['opcode_fields']) or '-'}")
        lines.append(f"Address candidates: {', '.join(item['address_fields']) or '-'}")
        lines.append(f"Label candidates: {', '.join(item['label_fields']) or '-'}")
        lines.append("Preview:")
        for sample in item["preview"]:
            lines.append(json.dumps(sample, ensure_ascii=False))
        lines.append("")
    TXT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def find_column(columns, label_name):
    normalized = {
        "".join(ch for ch in str(column).lower() if ch.isalnum()): column
        for column in columns
    }
    key = "".join(ch for ch in label_name.lower() if ch.isalnum())
    return normalized.get(key)


def inspect_label_values(path):
    df = read_table(path)
    report = {
        "source_file": str(path.relative_to(PROJECT_ROOT)),
        "rows": int(len(df)),
        "labels": {},
    }
    for label_name in PAPER_LABEL_NAMES:
        column = find_column(df.columns, label_name)
        if column is None:
            report["labels"][label_name] = {
                "column": None,
                "error": "column not found",
            }
            continue

        series = df[column]
        parsed = series.map(parse_label_value)
        value_counts = {
            str(key): int(value)
            for key, value in series.value_counts(dropna=False).items()
        }
        unique_values = [str(value) for value in series.drop_duplicates().head(200)]

        report["labels"][label_name] = {
            "column": str(column),
            "dtype": str(series.dtype),
            "unique_values": unique_values,
            "value_counts": value_counts,
            "empty_string_count": int(series.astype(str).str.strip().eq("").sum()),
            "nan_count": int(series.isna().sum()),
            "parsed_one_count": int((parsed == 1).sum()),
            "parsed_zero_count": int((parsed == 0).sum()),
            "unparsed_count": int(parsed.isna().sum()),
        }
    return report


def write_label_value_report(reports):
    LABEL_VALUE_JSON_REPORT.write_text(
        json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = ["BJUT SC01 label value report", ""]
    for file_report in reports["files"]:
        lines.append(f"File: {file_report['source_file']}")
        lines.append(f"Rows: {file_report['rows']}")
        lines.append("")
        for label_name, item in file_report["labels"].items():
            lines.append(f"Label: {label_name}")
            if item.get("error"):
                lines.append(f"Error: {item['error']}")
                lines.append("")
                continue
            lines.append(f"Column: {item['column']}")
            lines.append(f"Dtype: {item['dtype']}")
            lines.append(f"Unique values: {item['unique_values']}")
            lines.append(f"Value counts: {item['value_counts']}")
            lines.append(f"Empty strings: {item['empty_string_count']}")
            lines.append(f"NaN: {item['nan_count']}")
            lines.append(f"Parsed as 1: {item['parsed_one_count']}")
            lines.append(f"Parsed as 0: {item['parsed_zero_count']}")
            lines.append(f"Unable to parse: {item['unparsed_count']}")
            lines.append("")
    LABEL_VALUE_TXT_REPORT.write_text("\n".join(lines), encoding="utf-8")


def main():
    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Raw extracted directory not found: {RAW_DIR.relative_to(PROJECT_ROOT)}"
        )

    paths = sorted(
        path
        for path in RAW_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    reports = {
        "raw_dir": str(RAW_DIR.relative_to(PROJECT_ROOT)),
        "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
        "files": [inspect_file(path) for path in paths],
    }
    write_reports(reports)
    label_value_reports = {
        "raw_dir": str(RAW_DIR.relative_to(PROJECT_ROOT)),
        "files": [inspect_label_values(path) for path in paths],
    }
    write_label_value_report(label_value_reports)
    print(f"[OK] wrote {TXT_REPORT.relative_to(PROJECT_ROOT)}")
    print(f"[OK] wrote {JSON_REPORT.relative_to(PROJECT_ROOT)}")
    print(f"[OK] wrote {LABEL_VALUE_TXT_REPORT.relative_to(PROJECT_ROOT)}")
    print(f"[OK] wrote {LABEL_VALUE_JSON_REPORT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
