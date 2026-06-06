import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from preprocess_bjut_sc01 import (  # noqa: E402
    LABEL_NAMES,
    build_label_matrix,
    load_config,
    load_raw_frames,
    map_label_columns,
    parse_label_value,
    resolve_label_mapping,
)


NON_VULNERABILITY_COLUMNS = {
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

REPORT_TXT = PROJECT_ROOT / "data" / "reports" / "bjut_sc01_label_overlap_report.txt"
REPORT_JSON = PROJECT_ROOT / "data" / "reports" / "bjut_sc01_label_overlap_report.json"


def parse_general_vulnerability_series(series):
    parsed = series.map(parse_label_value)
    if parsed.notna().all():
        return parsed.astype(int).to_numpy(), "standard_binary"

    nonempty = series.notna() & ~series.astype(str).str.strip().eq("")
    if int(nonempty.sum()) > 0:
        return nonempty.astype(int).to_numpy(), "nonempty_positive"

    return np.zeros(len(series), dtype=int), "all_empty_or_unparsed"


def is_vulnerability_column(column, series):
    if column in NON_VULNERABILITY_COLUMNS:
        return False
    parsed = series.map(parse_label_value)
    if parsed.notna().all() and set(parsed.unique()).issubset({0, 1}):
        return True
    nonempty = series.notna() & ~series.astype(str).str.strip().eq("")
    return int(nonempty.sum()) > 0


def write_reports(report):
    REPORT_TXT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = ["BJUT SC01 label overlap report", ""]
    lines.append(f"rows: {report['rows']}")
    lines.append(f"raw_detected_vulnerability_label_count: {report['raw_detected_vulnerability_label_count']}")
    lines.append(f"paper_scope_vulnerability_label_count: {report['paper_scope_vulnerability_label_count']}")
    lines.append(f"target_10_label_count: {report['target_10_label_count']}")
    lines.append(f"remaining_label_count: {report['remaining_label_count']}")
    lines.append(f"excluded_duplicate_target_columns: {report['excluded_duplicate_target_columns']}")
    lines.append("")
    lines.append("Summary:")
    for key, value in report["summary"].items():
        lines.append(f"- {key}: {value}")

    lines.append("")
    lines.append("Current binary_label implication:")
    for key, value in report["binary_label_implication"].items():
        lines.append(f"- {key}: {value}")

    lines.append("")
    lines.append("Target 10 labels:")
    for item in report["target_10_labels"]:
        lines.append(
            f"- {item['label']} <- {item['source_column']} "
            f"(positive={item['positive_count']}, mode={item['parse_mode']})"
        )

    lines.append("")
    lines.append("Remaining vulnerability labels:")
    for item in report["remaining_labels"]:
        lines.append(
            f"- {item['column']} (positive={item['positive_count']}, "
            f"mode={item['parse_mode']})"
        )

    REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")


def main():
    config = load_config(PROJECT_ROOT / "configs" / "config.yaml")
    raw_data_dir = PROJECT_ROOT / config.get("raw_data_dir", "data/raw/BJUT_SC01/extracted")

    frames, read_errors = load_raw_frames(raw_data_dir)
    if not frames:
        raise RuntimeError(f"No readable raw data files found under {raw_data_dir}")
    df = pd.concat(frames, ignore_index=True, sort=False)

    warnings = []
    label_mapping, missing_labels, mapping_warnings = map_label_columns(df.columns)
    warnings.extend(mapping_warnings)
    label_mapping, parse_modes = resolve_label_mapping(df, label_mapping, warnings)
    target_matrix, _ = build_label_matrix(df, label_mapping, parse_modes, warnings)

    target_source_columns = set(label_mapping.values())
    target_semantic_column_names = {
        column
        for column in df.columns
        if any(
            "".join(ch for ch in str(column).lower() if ch.isalnum())
            == "".join(ch for ch in label.lower() if ch.isalnum())
            for label in LABEL_NAMES
        )
    }
    all_label_columns = []
    all_label_arrays = {}
    all_label_reports = {}
    for column in df.columns:
        if not is_vulnerability_column(column, df[column]):
            continue
        values, mode = parse_general_vulnerability_series(df[column])
        all_label_columns.append(column)
        all_label_arrays[column] = values
        all_label_reports[column] = {
            "column": str(column),
            "positive_count": int(values.sum()),
            "parse_mode": mode,
        }

    paper_scope_columns = []
    excluded_duplicate_target_columns = []
    for column in all_label_columns:
        if column in target_source_columns:
            paper_scope_columns.append(column)
        elif column in target_semantic_column_names:
            excluded_duplicate_target_columns.append(column)
        else:
            paper_scope_columns.append(column)

    remaining_columns = [
        column for column in paper_scope_columns if column not in target_source_columns
    ]
    paper_scope_matrix = np.array(
        [all_label_arrays[column] for column in paper_scope_columns]
    ).T
    remaining_matrix = np.array(
        [all_label_arrays[column] for column in remaining_columns]
    ).T

    target_positive = target_matrix.sum(axis=1) > 0
    remaining_positive = remaining_matrix.sum(axis=1) > 0
    all_positive = paper_scope_matrix.sum(axis=1) > 0
    only_target_positive = target_positive & ~remaining_positive
    only_remaining_positive = remaining_positive & ~target_positive
    both_target_and_remaining_positive = target_positive & remaining_positive
    no_vulnerability_positive = ~all_positive

    report = {
        "rows": int(len(df)),
        "read_errors": read_errors,
        "warnings": list(dict.fromkeys(warnings)),
        "raw_detected_vulnerability_label_count": len(all_label_columns),
        "paper_scope_vulnerability_label_count": len(paper_scope_columns),
        "target_10_label_count": len(LABEL_NAMES),
        "remaining_label_count": len(remaining_columns),
        "raw_detected_vulnerability_columns": [str(column) for column in all_label_columns],
        "paper_scope_vulnerability_columns": [str(column) for column in paper_scope_columns],
        "remaining_vulnerability_columns": [str(column) for column in remaining_columns],
        "excluded_duplicate_target_columns": [
            str(column) for column in excluded_duplicate_target_columns
        ],
        "summary": {
            "target_10_positive_count": int(target_positive.sum()),
            "remaining_positive_count": int(remaining_positive.sum()),
            "all_vulnerability_positive_count": int(all_positive.sum()),
            "only_target_10_positive_count": int(only_target_positive.sum()),
            "only_remaining_positive_count": int(only_remaining_positive.sum()),
            "both_target_10_and_remaining_positive_count": int(
                both_target_and_remaining_positive.sum()
            ),
            "no_vulnerability_positive_count": int(no_vulnerability_positive.sum()),
        },
        "binary_label_implication": {
            "current_binary_label_definition": "binary_label = any(target 10 labels)",
            "current_binary_positive_count": int(target_positive.sum()),
            "current_binary_negative_count": int((~target_positive).sum()),
            "would_be_positive_if_using_all_labels": int(all_positive.sum()),
            "only_remaining_positive_currently_binary_negative": int(
                only_remaining_positive.sum()
            ),
        },
        "target_10_labels": [
            {
                "label": label,
                "source_column": str(label_mapping[label]),
                "positive_count": int(target_matrix[:, idx].sum()),
                "parse_mode": parse_modes.get(label),
            }
            for idx, label in enumerate(LABEL_NAMES)
        ],
        "remaining_labels": [
            all_label_reports[column] for column in remaining_columns
        ],
    }

    write_reports(report)
    print(f"[OK] wrote {REPORT_TXT.relative_to(PROJECT_ROOT)}")
    print(f"[OK] wrote {REPORT_JSON.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
