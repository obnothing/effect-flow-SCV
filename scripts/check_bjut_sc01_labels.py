import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from preprocess_bjut_sc01 import (  # noqa: E402
    LABEL_NAMES,
    PAPER_TABLE_II_COUNTS,
    load_config,
    load_raw_frames,
    parse_label_value,
)


METADATA_COLUMNS = {
    "Unnamed: 0",
    "address",
    "contractCode",
    "timestamp",
    "createValue",
    "createdBlockNumber",
    "createdTransactionHash",
    "creationCode",
    "creator",
    "flag",
    "_source_file",
}
REPORT_TXT = PROJECT_ROOT / "data" / "reports" / "bjut_sc01_all_label_counts.txt"
REPORT_JSON = PROJECT_ROOT / "data" / "reports" / "bjut_sc01_all_label_counts.json"


def normalize_column_name(value):
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def summarize_value_types(series):
    counts = {}
    for value in series:
        if pd.isna(value):
            key = "NaN"
        elif isinstance(value, bool):
            key = "bool"
        elif isinstance(value, int):
            key = "int"
        elif isinstance(value, float):
            key = "float"
        elif isinstance(value, str):
            text = value.strip()
            if text == "":
                key = "empty_string"
            elif text.startswith("[") and text.endswith("]"):
                key = "list_string"
            else:
                key = "string"
        else:
            key = type(value).__name__
        counts[key] = counts.get(key, 0) + 1
    return counts


def parse_label_column(series):
    parsed = series.map(parse_label_value)
    parsed_known = parsed.notna()
    parsed_unique = set(parsed[parsed_known].astype(int).unique())

    if parsed_known.all() and parsed_unique.issubset({0, 1}):
        values = parsed.astype(int)
        strategy = "standard_binary"
    else:
        nonempty = series.notna() & ~series.astype(str).str.strip().eq("")
        if int(nonempty.sum()) > 0:
            values = nonempty.astype(int)
            strategy = "nonempty_positive"
        else:
            values = pd.Series([0] * len(series), index=series.index)
            strategy = "all_empty_or_unparsed"

    return values, strategy, parsed


def is_vulnerability_column(column, series):
    if column in METADATA_COLUMNS:
        return False
    values, strategy, parsed = parse_label_column(series)
    if strategy == "standard_binary":
        return set(parsed.astype(int).unique()).issubset({0, 1})
    return int(values.sum()) > 0


def inspect_label_column(column, series):
    values, strategy, parsed = parse_label_column(series)
    positive_mask = values == 1
    positive_values = series[positive_mask].dropna().astype(str).drop_duplicates().head(10)
    value_counts = {
        str(key): int(value)
        for key, value in series.value_counts(dropna=False).head(50).items()
    }

    return {
        "column_name": str(column),
        "positive_count": int(positive_mask.sum()),
        "zero_count": int((values == 0).sum()),
        "empty_count": int(series.astype(str).str.strip().eq("").sum()),
        "nan_count": int(series.isna().sum()),
        "value_type_summary": summarize_value_types(series),
        "example_positive_values": positive_values.tolist(),
        "parse_strategy": strategy,
        "raw_value_counts_top50": value_counts,
        "standard_parser_one_count": int((parsed == 1).sum()),
        "standard_parser_zero_count": int((parsed == 0).sum()),
        "standard_parser_unparsed_count": int(parsed.isna().sum()),
    }


def reverse_match_expected(label_reports):
    matches = {}
    for paper_label, expected in PAPER_TABLE_II_COUNTS.items():
        exact = []
        close = []
        tolerance = max(10, int(round(expected * 0.05)))
        for item in label_reports:
            actual = item["positive_count"]
            diff = actual - expected
            row = {
                "column_name": item["column_name"],
                "positive_count": actual,
                "diff": diff,
                "abs_diff": abs(diff),
                "parse_strategy": item["parse_strategy"],
            }
            if diff == 0:
                exact.append(row)
            elif abs(diff) <= tolerance:
                close.append(row)
        matches[paper_label] = {
            "expected_count": expected,
            "exact_matches": sorted(exact, key=lambda row: row["column_name"]),
            "near_matches_within_5_percent_or_10": sorted(
                close, key=lambda row: row["abs_diff"]
            ),
        }
    return matches


def build_unchecked_audit(df, label_reports, reverse_matches):
    by_name = {item["column_name"]: item for item in label_reports}
    exact_7214 = reverse_matches["Unchecked call return value"]["exact_matches"]
    near_7214 = reverse_matches["Unchecked call return value"][
        "near_matches_within_5_percent_or_10"
    ]
    original = by_name.get("Unchecked call return value")
    overpowered = by_name.get("Overpowered role")

    return {
        "paper_label": "Unchecked call return value",
        "paper_expected_count": PAPER_TABLE_II_COUNTS["Unchecked call return value"],
        "original_same_name_column": original,
        "overpowered_role_column": overpowered,
        "is_overpowered_role_unique_exact_7214_match": (
            len(exact_7214) == 1 and exact_7214[0]["column_name"] == "Overpowered role"
        ),
        "exact_7214_matches": exact_7214,
        "near_7214_matches": near_7214,
        "assessment": {
            "why_original_same_name_is_suspicious": (
                "The raw same-name column has positive_count=666, while paper Table II "
                "expects 7214 for Unchecked call return value."
            ),
            "why_overpowered_role_is_used_by_current_preprocess": (
                "Overpowered role is the only raw label column with positive_count=7214."
            ),
            "risk": (
                "This is a semantic name mismatch. Count matching supports reproduction "
                "compatibility, but true semantic correctness cannot be guaranteed "
                "without author code or dataset documentation."
            ),
            "possible_causes": [
                "CSV column name mismatch",
                "paper label mapping error",
                "dataset export column shift or relabeling",
                "different BJUT SC01 version used by the paper",
            ],
        },
    }


def write_reports(report):
    REPORT_TXT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = ["BJUT SC01 all label count audit", ""]
    lines.append(f"rows: {report['rows']}")
    lines.append(f"label_column_count: {report['label_column_count']}")
    lines.append("")
    lines.append("Top 20 labels by positive_count:")
    for item in report["top20_by_positive_count"]:
        lines.append(
            f"- {item['column_name']}: positive={item['positive_count']}, "
            f"zero={item['zero_count']}, nan={item['nan_count']}, "
            f"strategy={item['parse_strategy']}"
        )

    lines.append("")
    lines.append("Paper Table II reverse matches:")
    for label, item in report["paper_table_ii_reverse_matches"].items():
        lines.append(f"- {label} expected={item['expected_count']}")
        lines.append(f"  exact: {item['exact_matches']}")
        lines.append(f"  near: {item['near_matches_within_5_percent_or_10']}")

    lines.append("")
    lines.append("Unchecked call return value audit:")
    unchecked = report["unchecked_call_return_value_audit"]
    lines.append(
        "original_same_name_positive_count: "
        f"{unchecked['original_same_name_column']['positive_count']}"
    )
    lines.append(
        "overpowered_role_positive_count: "
        f"{unchecked['overpowered_role_column']['positive_count']}"
    )
    lines.append(
        "is_overpowered_role_unique_exact_7214_match: "
        f"{unchecked['is_overpowered_role_unique_exact_7214_match']}"
    )
    lines.append(f"exact_7214_matches: {unchecked['exact_7214_matches']}")
    lines.append(f"near_7214_matches: {unchecked['near_7214_matches']}")
    lines.append(f"risk: {unchecked['assessment']['risk']}")

    lines.append("")
    lines.append("All label columns:")
    for item in report["label_columns"]:
        lines.append(
            f"- {item['column_name']} | positive={item['positive_count']} | "
            f"zero={item['zero_count']} | empty={item['empty_count']} | "
            f"nan={item['nan_count']} | strategy={item['parse_strategy']} | "
            f"types={item['value_type_summary']} | "
            f"examples={item['example_positive_values']}"
        )

    REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")


def main():
    config = load_config(PROJECT_ROOT / "configs" / "config.yaml")
    raw_data_dir = PROJECT_ROOT / config.get("raw_data_dir", "data/raw/BJUT_SC01/extracted")
    frames, read_errors = load_raw_frames(raw_data_dir)
    if not frames:
        raise RuntimeError(f"No readable raw files under {raw_data_dir}")

    df = pd.concat(frames, ignore_index=True, sort=False)
    label_reports = []
    for column in df.columns:
        if is_vulnerability_column(column, df[column]):
            label_reports.append(inspect_label_column(column, df[column]))

    reverse_matches = reverse_match_expected(label_reports)
    top20 = sorted(
        label_reports, key=lambda item: item["positive_count"], reverse=True
    )[:20]

    report = {
        "rows": int(len(df)),
        "read_errors": read_errors,
        "paper_target_labels": LABEL_NAMES,
        "paper_expected_counts": PAPER_TABLE_II_COUNTS,
        "label_column_count": len(label_reports),
        "label_columns": label_reports,
        "top20_by_positive_count": top20,
        "paper_table_ii_reverse_matches": reverse_matches,
        "unchecked_call_return_value_audit": build_unchecked_audit(
            df, label_reports, reverse_matches
        ),
    }
    write_reports(report)
    print(f"[OK] wrote {REPORT_TXT.relative_to(PROJECT_ROOT)}")
    print(f"[OK] wrote {REPORT_JSON.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
