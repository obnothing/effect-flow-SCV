import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from effect_flow_utils import (  # noqa: E402
    DATASET_SPECS,
    REPORT_DIR,
    count_jsonl,
    infer_fields,
    iter_jsonl,
    relative,
    resolve,
    write_json,
    write_text,
)


def inspect_file(dataset, split, path, split_type, selected):
    exists = path.exists()
    first = None
    if exists:
        first = next((item for _, item in iter_jsonl(path)), None)
    fields, id_field, opcode_field, label_fields = (
        infer_fields(first) if first else ([], None, None, [])
    )
    warnings = []
    if not exists:
        warnings.append("missing file")
    if exists and not id_field:
        warnings.append("id field not found")
    if exists and not opcode_field:
        warnings.append("opcode field not found")
    if "mlsmote" in path.name.lower():
        warnings.append("oversampled file: excluded from semantic pretraining audit")
    return {
        "dataset_name": dataset,
        "split_name": split,
        "file_path": relative(path),
        "exists": exists,
        "sample_count": count_jsonl(path),
        "fields": fields,
        "id_field": id_field,
        "opcode_field": opcode_field,
        "label_fields": label_fields,
        "split_type_inferred": split_type,
        "selected_for_stage16a": selected,
        "warnings": warnings,
    }


def main():
    inventory = []
    selections = {}
    fatal_warnings = []
    for dataset, spec in DATASET_SPECS.items():
        strict_root = resolve(spec["strict_dir"])
        random_root = resolve(spec["random_dir"])
        selected_paths = {}
        for split in ("train", "valid", "test"):
            strict_path = strict_root / f"{split}.jsonl"
            random_path = random_root / f"{split}.jsonl"
            inventory.append(
                inspect_file(dataset, split, strict_path, "opcode_hash_grouped_strict", True)
            )
            if random_path.exists():
                inventory.append(
                    inspect_file(dataset, split, random_path, "random_sample_split", False)
                )
            oversampled = strict_root / "train_mlsmote.jsonl"
            if split == "train" and oversampled.exists():
                inventory.append(
                    inspect_file(
                        dataset,
                        "train_mlsmote",
                        oversampled,
                        "oversampled_from_strict_train",
                        False,
                    )
                )
            if strict_path.exists():
                selected_paths[split] = relative(strict_path)
        if len(selected_paths) != 3:
            fatal_warnings.append(
                f"{dataset}: strict split is incomplete; candidate paths require manual confirmation"
            )
        selections[dataset] = {
            "selection_reason": (
                "project provenance identifies this directory as opcode-hash grouped; "
                "random and MLSMOTE variants are excluded"
            ),
            "paths": selected_paths,
        }

    status = "ok" if not fatal_warnings else "blocked"
    report = {
        "status": status,
        "selected_inputs": selections,
        "inventory": inventory,
        "fatal_warnings": fatal_warnings,
        "policy": {
            "prefer_strict_grouped_split": True,
            "exclude_train_mlsmote": True,
            "exclude_random_split": True,
        },
    }
    json_path = REPORT_DIR / "stage16a_input_inventory.json"
    txt_path = REPORT_DIR / "stage16a_input_inventory.txt"
    write_json(json_path, report)
    lines = ["Stage 16A effect-flow input inventory", "", f"status: {status}"]
    for row in inventory:
        lines.extend(
            [
                "",
                f"dataset_name: {row['dataset_name']}",
                f"split_name: {row['split_name']}",
                f"file_path: {row['file_path']}",
                f"exists: {row['exists']}",
                f"sample_count: {row['sample_count']}",
                f"fields: {row['fields']}",
                f"id_field: {row['id_field']}",
                f"opcode_field: {row['opcode_field']}",
                f"label_fields: {row['label_fields']}",
                f"split_type_inferred: {row['split_type_inferred']}",
                f"selected_for_stage16a: {row['selected_for_stage16a']}",
                f"warnings: {row['warnings']}",
            ]
        )
    if fatal_warnings:
        lines.extend(["", "FATAL WARNINGS:", *fatal_warnings])
    write_text(txt_path, lines)
    print(f"[OK] wrote {relative(txt_path)}")
    print(f"[OK] wrote {relative(json_path)}")
    if fatal_warnings:
        raise SystemExit("Input paths are ambiguous or incomplete; inspect the inventory report.")


if __name__ == "__main__":
    main()
