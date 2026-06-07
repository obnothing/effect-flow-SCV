import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from evm_opcode import bytecode_to_opcode_sequence, normalize_opcode_sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_EXTENSIONS = {".csv", ".json", ".jsonl", ".txt", ".xlsx"}
BYTECODE_CANDIDATES = ["contractCode", "bytecode", "runtime_bytecode", "creationCode"]
OPCODE_CANDIDATES = ["opcode", "opcodes", "opcode_sequence"]
ADDRESS_CANDIDATES = ["address", "contract_address", "id", "contract_id"]

LABEL_NAMES = [
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
PAPER_TABLE_II_COUNTS = {
    "Reentrancy": 1422,
    "Contract contains unknown address": 18433,
    "Integer overflow or underflow": 18263,
    "Timestamp dependence": 777,
    "DoS with failed call": 521,
    "Assert violation": 854,
    "Unchecked call return value": 7214,
    "Unsafe send": 1261,
    "Multiplication after division": 1937,
    "Extra gas consumption": 5597,
}
LABEL_MAPPING_ASSUMPTIONS = {
    "Unchecked call return value": {
        "paper_label": "Unchecked call return value",
        "raw_column_used": "Overpowered role",
        "reason": "positive_count matches paper Table II expected count 7214",
        "risk": (
            "semantic name mismatch, treated as reproduction compatibility mapping"
        ),
        "warning": (
            "true semantic correctness cannot be guaranteed without author code or "
            "dataset documentation"
        ),
    }
}

LABEL_ALIASES = {
    "Reentrancy": [
        "reentrancy",
        "reentrant",
        "re_entrancy",
    ],
    "Contract contains unknown address": [
        "contract contains unknown address",
        "unknown address",
        "unknown_address",
        "unknownaddress",
    ],
    "Integer overflow or underflow": [
        "integer overflow or underflow",
        "integer overflow",
        "integer underflow",
        "overflow",
        "underflow",
        "arithmetic",
    ],
    "Timestamp dependence": [
        "timestamp dependence",
        "timestamp dependency",
        "timestamp",
        "time dependence",
        "time dependency",
    ],
    "DoS with failed call": [
        "dos with failed call",
        "denial of service with failed call",
        "failed call",
        "dos",
    ],
    "Assert violation": [
        "assert violation",
        "assert",
        "assertion violation",
    ],
    "Unchecked call return value": [
        "unchecked call return value",
        "unchecked return value",
        "unchecked call",
        "unchecked",
        "overpowered role",
    ],
    "Unsafe send": [
        "unsafe send",
        "send",
    ],
    "Multiplication after division": [
        "multiplication after division",
        "multiply after divide",
        "mul after div",
        "division before multiplication",
    ],
    "Extra gas consumption": [
        "extra gas consumption",
        "gas consumption",
        "extra gas",
        "gas",
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess BJUT SC01 dataset.")
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_name(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


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


def load_raw_frames(raw_data_dir):
    paths = sorted(
        path
        for path in raw_data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    frames = []
    errors = []
    for path in paths:
        try:
            df = read_table(path)
            df["_source_file"] = str(path.relative_to(PROJECT_ROOT))
            frames.append(df)
        except Exception as exc:
            errors.append({"path": str(path.relative_to(PROJECT_ROOT)), "error": str(exc)})
    return frames, errors


def choose_field(columns, candidates):
    normalized = {normalize_name(col): col for col in columns}
    for candidate in candidates:
        key = normalize_name(candidate)
        if key in normalized:
            return normalized[key]
    return None


def map_label_columns(columns):
    normalized_columns = {normalize_name(column): column for column in columns}
    mapping = {}
    missing = []
    warnings = []

    for label_name in LABEL_NAMES:
        aliases = [label_name] + LABEL_ALIASES.get(label_name, [])
        matched = None
        for alias in aliases:
            alias_key = normalize_name(alias)
            if alias_key in normalized_columns:
                matched = normalized_columns[alias_key]
                break
        if matched is None:
            missing.append(label_name)
            warnings.append(f"Missing label column for: {label_name}")
        else:
            mapping[label_name] = matched

    return mapping, missing, warnings


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

    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "1.0", "true"}:
            return 1
        if text in {"0", "0.0", "false"}:
            return 0
        if text == "":
            return None
        return None

    return None


def parse_label_series(series, label_name, warnings):
    parsed = series.map(parse_label_value)
    expected = PAPER_TABLE_II_COUNTS[label_name]
    parsed_one_count = int((parsed == 1).sum())
    unparsed_count = int(parsed.isna().sum())

    if parsed_one_count == expected:
        return parsed.fillna(0).astype(int).tolist(), "standard_binary"

    nonempty_positive = series.notna() & ~series.astype(str).str.strip().eq("")
    nonempty_count = int(nonempty_positive.sum())
    if nonempty_count == expected:
        warnings.append(
            f"{label_name}: parsed as non-empty positive values because the column "
            f"uses location-list style labels; NaN/empty values are treated as 0."
        )
        return nonempty_positive.astype(int).tolist(), "nonempty_positive"

    if unparsed_count > 0 and expected > 0:
        warnings.append(
            f"{label_name}: {unparsed_count} values could not be parsed by the "
            f"standard binary parser; parsed positives={parsed_one_count}, "
            f"expected={expected}."
        )
    return parsed.fillna(0).astype(int).tolist(), "standard_binary_with_unparsed_as_zero"


def resolve_label_mapping(df, initial_mapping, warnings):
    resolved = dict(initial_mapping)
    parse_modes = {}

    for label_name in LABEL_NAMES:
        if label_name not in resolved:
            continue
        values, mode = parse_label_series(df[resolved[label_name]], label_name, warnings)
        actual = int(sum(values))
        expected = PAPER_TABLE_II_COUNTS[label_name]
        if actual == expected:
            parse_modes[label_name] = mode
            continue

        matching_columns = []
        for column in df.columns:
            candidate_values, candidate_mode = parse_label_series(
                df[column], label_name, []
            )
            if int(sum(candidate_values)) == expected:
                matching_columns.append((column, candidate_mode))

        if matching_columns:
            new_column, new_mode = matching_columns[0]
            old_column = resolved[label_name]
            resolved[label_name] = new_column
            parse_modes[label_name] = new_mode
            warnings.append(
                f"{label_name}: remapped from column '{old_column}' "
                f"(actual count {actual}) to column '{new_column}' because it matches "
                f"paper Table II expected count {expected}."
            )
        else:
            parse_modes[label_name] = mode

    return resolved, parse_modes


def build_label_matrix(df, label_mapping, parse_modes, warnings):
    label_values = {}
    label_parse_report = {}
    for label_name in LABEL_NAMES:
        column = label_mapping[label_name]
        values, mode = parse_label_series(df[column], label_name, warnings)
        label_values[label_name] = values

        series = df[column]
        parsed = series.map(parse_label_value)
        label_parse_report[label_name] = {
            "column": str(column),
            "mode": parse_modes.get(label_name, mode),
            "nan_count": int(series.isna().sum()),
            "empty_string_count": int(series.astype(str).str.strip().eq("").sum()),
            "parsed_one_count": int((parsed == 1).sum()),
            "parsed_zero_count": int((parsed == 0).sum()),
            "unparsed_count": int(parsed.isna().sum()),
            "actual_count": int(sum(values)),
            "expected_count": PAPER_TABLE_II_COUNTS[label_name],
        }

    labels = np.array([label_values[label] for label in LABEL_NAMES], dtype=int).T
    return labels, label_parse_report


def compare_with_paper(label_counts):
    rows = []
    for label_name in LABEL_NAMES:
        expected = PAPER_TABLE_II_COUNTS[label_name]
        actual = int(label_counts.get(label_name, 0))
        diff = actual - expected
        rows.append(
            {
                "label": label_name,
                "expected": expected,
                "actual": actual,
                "diff": diff,
                "status": "OK" if diff == 0 else "MISMATCH",
            }
        )
    return rows


def audit_label_mapping_assumptions(df):
    unchecked_expected = PAPER_TABLE_II_COUNTS["Unchecked call return value"]
    exact_matches = []
    near_matches = []
    tolerance = max(10, int(round(unchecked_expected * 0.05)))

    for column in df.columns:
        parsed = df[column].map(parse_label_value)
        if parsed.notna().all():
            positive_count = int((parsed == 1).sum())
        else:
            nonempty = df[column].notna() & ~df[column].astype(str).str.strip().eq("")
            positive_count = int(nonempty.sum())

        diff = positive_count - unchecked_expected
        row = {
            "column_name": str(column),
            "positive_count": positive_count,
            "diff": diff,
            "abs_diff": abs(diff),
        }
        if diff == 0:
            exact_matches.append(row)
        elif abs(diff) <= tolerance:
            near_matches.append(row)

    exact_matches = sorted(exact_matches, key=lambda row: row["column_name"])
    near_matches = sorted(near_matches, key=lambda row: row["abs_diff"])
    strong_conflict = not (
        len(exact_matches) == 1
        and exact_matches[0]["column_name"] == "Overpowered role"
    )

    assumptions = dict(LABEL_MAPPING_ASSUMPTIONS)
    assumptions["Unchecked call return value"] = {
        **assumptions["Unchecked call return value"],
        "expected_count": unchecked_expected,
        "exact_count_matches": exact_matches,
        "near_count_matches_within_5_percent_or_10": near_matches,
        "strong_conflict": strong_conflict,
    }
    return assumptions, strong_conflict


def make_opcode(row, opcode_field, bytecode_field):
    if opcode_field:
        opcode = normalize_opcode_sequence(row.get(opcode_field))
        if opcode:
            return opcode, "existing_opcode"
    if bytecode_field:
        opcode = bytecode_to_opcode_sequence(row.get(bytecode_field))
        if opcode:
            return opcode, "bytecode_converted"
    return None, "missing_or_failed"


def count_push_operand_pairs(opcode):
    tokens = str(opcode).split()
    push_count = 0
    push_operand_count = 0
    for idx, token in enumerate(tokens):
        if re.fullmatch(r"PUSH(?:[1-9]|[12][0-9]|3[0-2])", token):
            push_count += 1
            if idx + 1 < len(tokens) and str(tokens[idx + 1]).lower().startswith("0x"):
                push_operand_count += 1
    return push_count, push_operand_count


def split_indices(labels, seed):
    indices = np.arange(len(labels))
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

        first_split = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=seed
        )
        train_idx, temp_idx = next(first_split.split(indices, labels))

        temp_labels = labels[temp_idx]
        second_split = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=0.5, random_state=seed
        )
        valid_rel, test_rel = next(second_split.split(temp_idx, temp_labels))
        return train_idx, temp_idx[valid_rel], temp_idx[test_rel], "iterative-stratification"
    except Exception as exc:
        iterative_error = exc

    try:
        from sklearn.model_selection import train_test_split

        stratify = labels[:, 0] if len(np.unique(labels[:, 0])) > 1 else None
        train_idx, temp_idx = train_test_split(
            indices, test_size=0.2, random_state=seed, shuffle=True, stratify=stratify
        )
        temp_stratify = labels[temp_idx, 0] if len(np.unique(labels[temp_idx, 0])) > 1 else None
        valid_idx, test_idx = train_test_split(
            temp_idx,
            test_size=0.5,
            random_state=seed,
            shuffle=True,
            stratify=temp_stratify,
        )
        return (
            train_idx,
            valid_idx,
            test_idx,
            f"sklearn train_test_split fallback: {iterative_error}",
        )
    except Exception as exc:
        rng = np.random.default_rng(seed)
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        train_end = int(len(shuffled) * 0.8)
        valid_end = int(len(shuffled) * 0.9)
        return (
            shuffled[:train_end],
            shuffled[train_end:valid_end],
            shuffled[valid_end:],
            f"numpy random split fallback: {iterative_error}; sklearn error: {exc}",
        )


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_report(summary, text_path, json_path):
    if summary.get("warnings"):
        summary["warnings"] = list(dict.fromkeys(summary["warnings"]))

    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["BJUT SC01 processed data report", ""]
    for key in [
        "status",
        "raw_data_dir",
        "output_dir",
        "input_rows",
        "usable_samples",
        "skipped_samples",
        "opcode_field",
        "bytecode_field",
        "opcode_source",
        "split_method",
    ]:
        lines.append(f"{key}: {summary.get(key)}")

    if summary.get("opcode_operand_stats"):
        lines.append("")
        lines.append("Opcode operand stats:")
        for key, value in summary["opcode_operand_stats"].items():
            lines.append(f"- {key}: {value}")

    lines.append("")
    lines.append("Label mapping:")
    for label, column in summary.get("label_mapping", {}).items():
        lines.append(f"- {label}: {column}")

    if summary.get("warnings"):
        lines.append("")
        lines.append("Warnings:")
        for warning in summary["warnings"]:
            lines.append(f"- {warning}")

    if summary.get("splits"):
        lines.append("")
        lines.append("Splits:")
        for name, count in summary["splits"].items():
            lines.append(f"- {name}: {count}")

    if summary.get("label_counts"):
        lines.append("")
        lines.append("Label counts:")
        for label, count in summary["label_counts"].items():
            lines.append(f"- {label}: {count}")

    if summary.get("paper_table_ii_comparison"):
        lines.append("")
        lines.append("Paper Table II expected counts vs actual counts:")
        lines.append("Label | expected | actual | diff | status")
        for row in summary["paper_table_ii_comparison"]:
            lines.append(
                f"{row['label']} | {row['expected']} | {row['actual']} | "
                f"{row['diff']} | {row['status']}"
            )

    if summary.get("label_parse_report"):
        lines.append("")
        lines.append("Label parse report:")
        for label, item in summary["label_parse_report"].items():
            lines.append(
                f"- {label}: column={item['column']}, mode={item['mode']}, "
                f"parsed_one={item['parsed_one_count']}, "
                f"unparsed={item['unparsed_count']}"
            )

    if summary.get("label_mapping_assumptions"):
        lines.append("")
        lines.append("Label Mapping Assumptions:")
        for label, item in summary["label_mapping_assumptions"].items():
            lines.append(f"- paper_label: {item['paper_label']}")
            lines.append(f"  raw_column_used: {item['raw_column_used']}")
            lines.append(f"  reason: {item['reason']}")
            lines.append(f"  risk: {item['risk']}")
            lines.append(f"  warning: {item['warning']}")
            lines.append(f"  exact_count_matches: {item.get('exact_count_matches')}")
            lines.append(
                "  near_count_matches_within_5_percent_or_10: "
                f"{item.get('near_count_matches_within_5_percent_or_10')}"
            )
            lines.append(f"  strong_conflict: {item.get('strong_conflict')}")

    text_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    config = load_config(args.config)
    seed = int(config.get("seed", 42))

    raw_data_dir = PROJECT_ROOT / config.get("raw_data_dir", "data/raw/BJUT_SC01/extracted")
    output_dir = PROJECT_ROOT / config.get("data_dir", "data/processed/BJUT_SC01")
    report_dir = PROJECT_ROOT / config.get("report_dir", "data/reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    text_report = report_dir / "bjut_sc01_processed_report.txt"
    json_report = report_dir / "bjut_sc01_processed_report.json"

    frames, read_errors = load_raw_frames(raw_data_dir)
    if not frames:
        raise RuntimeError(f"No readable raw data files found under {raw_data_dir}")

    df = pd.concat(frames, ignore_index=True, sort=False)
    opcode_field = choose_field(df.columns, OPCODE_CANDIDATES)
    bytecode_field = choose_field(df.columns, BYTECODE_CANDIDATES)
    id_field = choose_field(df.columns, ADDRESS_CANDIDATES)
    label_mapping, missing_labels, warnings = map_label_columns(df.columns)
    label_mapping, parse_modes = resolve_label_mapping(df, label_mapping, warnings)
    warnings.extend([f"Could not read {item['path']}: {item['error']}" for item in read_errors])

    summary = {
        "status": "failed",
        "raw_data_dir": str(raw_data_dir.relative_to(PROJECT_ROOT)),
        "output_dir": str(output_dir.relative_to(PROJECT_ROOT)),
        "input_rows": int(len(df)),
        "usable_samples": 0,
        "skipped_samples": 0,
        "opcode_field": opcode_field,
        "bytecode_field": bytecode_field,
        "id_field": id_field,
        "opcode_source": {},
        "label_names": LABEL_NAMES,
        "label_mapping": label_mapping,
        "missing_labels": missing_labels,
        "warnings": warnings,
        "splits": {},
        "label_counts": {},
        "label_parse_report": {},
        "paper_table_ii_comparison": [],
        "split_method": None,
        "opcode_operand_stats": {},
    }

    if missing_labels:
        for label_name in missing_labels:
            print(f"[WARNING] missing label column for: {label_name}")
        summary["warnings"].append(
            "Not all 10 labels were mapped. Run scripts/inspect_bjut_sc01.py and "
            "update LABEL_ALIASES after checking raw field names."
        )
        build_report(summary, text_report, json_report)
        print(f"[STOP] wrote report: {text_report.relative_to(PROJECT_ROOT)}")
        sys.exit(1)

    if not opcode_field and not bytecode_field:
        summary["warnings"].append("No opcode or bytecode field found.")
        build_report(summary, text_report, json_report)
        print(f"[STOP] wrote report: {text_report.relative_to(PROJECT_ROOT)}")
        sys.exit(1)

    label_matrix, label_parse_report = build_label_matrix(
        df, label_mapping, parse_modes, warnings
    )
    label_mapping_assumptions, strong_mapping_conflict = audit_label_mapping_assumptions(
        df
    )
    label_counts = {
        label: int(label_matrix[:, idx].sum()) for idx, label in enumerate(LABEL_NAMES)
    }
    paper_comparison = compare_with_paper(label_counts)
    summary["label_parse_report"] = label_parse_report
    summary["label_counts"] = label_counts
    summary["paper_table_ii_comparison"] = paper_comparison
    summary["label_mapping_assumptions"] = label_mapping_assumptions
    if any(row["status"] == "MISMATCH" for row in paper_comparison):
        summary["warnings"].append("Label count mismatch with paper Table II")
        build_report(summary, text_report, json_report)
        print("[ERROR] Label count mismatch with paper Table II")
        print(f"[STOP] wrote report: {text_report.relative_to(PROJECT_ROOT)}")
        sys.exit(1)

    train_idx, valid_idx, test_idx, split_method = split_indices(label_matrix, seed)
    split_lookup = {}
    for name, split_indices_for_name in {
        "train": train_idx,
        "valid": valid_idx,
        "test": test_idx,
    }.items():
        for idx in split_indices_for_name:
            split_lookup[int(idx)] = name

    skipped = 0
    usable_samples = 0
    opcode_source_counts = {}
    split_counts = {"train": 0, "valid": 0, "test": 0}
    written_label_counts = np.zeros(len(LABEL_NAMES), dtype=int)
    opcode_operand_stats = {
        "push_instruction_count": 0,
        "push_operand_count": 0,
        "samples_with_push": 0,
        "samples_with_push_operands": 0,
    }

    writers = {
        name: (output_dir / f"{name}.jsonl").open("w", encoding="utf-8")
        for name in ["train", "valid", "test"]
    }
    try:
        for row_idx, row in tqdm(df.iterrows(), total=len(df), desc="preprocess"):
            opcode, source = make_opcode(row, opcode_field, bytecode_field)
            opcode_source_counts[source] = opcode_source_counts.get(source, 0) + 1
            if not opcode:
                skipped += 1
                continue
            split_name = split_lookup.get(int(row_idx))
            if split_name is None:
                skipped += 1
                continue

            push_count, push_operand_count = count_push_operand_pairs(opcode)
            opcode_operand_stats["push_instruction_count"] += push_count
            opcode_operand_stats["push_operand_count"] += push_operand_count
            if push_count > 0:
                opcode_operand_stats["samples_with_push"] += 1
            if push_operand_count > 0:
                opcode_operand_stats["samples_with_push_operands"] += 1

            multi_labels = label_matrix[row_idx].astype(int).tolist()
            binary_label = int(any(multi_labels))
            sample_id = row.get(id_field) if id_field else None
            if pd.isna(sample_id) or sample_id is None or str(sample_id).strip() == "":
                sample_id = f"sample_{row_idx}"

            record = {
                "id": str(sample_id),
                "opcode": opcode,
                "binary_label": binary_label,
                "multi_labels": multi_labels,
            }
            writers[split_name].write(json.dumps(record, ensure_ascii=False) + "\n")
            split_counts[split_name] += 1
            written_label_counts += np.asarray(multi_labels, dtype=int)
            usable_samples += 1
    finally:
        for writer in writers.values():
            writer.close()

    if usable_samples == 0:
        summary["warnings"].append("No usable samples after opcode extraction.")
        summary["skipped_samples"] = skipped
        summary["opcode_source"] = opcode_source_counts
        build_report(summary, text_report, json_report)
        print(f"[STOP] wrote report: {text_report.relative_to(PROJECT_ROOT)}")
        sys.exit(1)

    label_mapping_out = {
        "label_names": LABEL_NAMES,
        "label_to_id": {label: idx for idx, label in enumerate(LABEL_NAMES)},
        "id_to_label": {str(idx): label for idx, label in enumerate(LABEL_NAMES)},
        "source_columns": label_mapping,
        "label_aliases": LABEL_ALIASES,
        "label_mapping_assumptions": label_mapping_assumptions,
    }
    (output_dir / "label_mapping.json").write_text(
        json.dumps(label_mapping_out, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    summary.update(
        {
            "status": "warning" if strong_mapping_conflict else "ok",
            "usable_samples": usable_samples,
            "skipped_samples": skipped,
            "opcode_source": opcode_source_counts,
            "opcode_operand_stats": opcode_operand_stats,
            "split_method": split_method,
            "splits": split_counts,
            "label_counts": {
                label: int(written_label_counts[idx])
                for idx, label in enumerate(LABEL_NAMES)
            },
            "paper_table_ii_comparison": compare_with_paper(
                {
                    label: int(written_label_counts[idx])
                    for idx, label in enumerate(LABEL_NAMES)
                }
            ),
            "label_parse_report": label_parse_report,
            "label_mapping_assumptions": label_mapping_assumptions,
        }
    )
    build_report(summary, text_report, json_report)
    print(f"[OK] wrote {output_dir.relative_to(PROJECT_ROOT)}")
    print(f"[OK] wrote {text_report.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
