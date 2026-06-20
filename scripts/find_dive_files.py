import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "data" / "reports"
SEARCH_TERMS = ["DIVE", "dive", "Runtime_Opcode", "labels", "Labels", "opcode"]
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "checkpoints", "models", "logs"}


def infer_role(path):
    name = path.name.lower()
    parts = [part.lower() for part in path.parts]
    if name == "runtime_opcode.jsonl":
        return "runtime_opcode_jsonl"
    if name == "creation_opcode.jsonl":
        return "creation_opcode_jsonl"
    if name == "dive_labels.csv":
        return "label_csv"
    if name == "tool_results.csv":
        return "tool_results_csv"
    if "source codes" in parts:
        return "solidity_source"
    if "label" in name:
        return "label_related"
    if "opcode" in name:
        return "opcode_related"
    if "dive" in name:
        return "dive_related"
    return "candidate"


def should_skip(path):
    return any(part in SKIP_DIRS for part in path.parts)


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    files = []
    for path in PROJECT_ROOT.rglob("*"):
        if should_skip(path.relative_to(PROJECT_ROOT)):
            continue
        if not path.is_file():
            continue
        if not any(term.lower() in path.name.lower() for term in SEARCH_TERMS):
            continue
        stat = path.stat()
        files.append(
            {
                "path": path.relative_to(PROJECT_ROOT).as_posix(),
                "size_bytes": stat.st_size,
                "modified_time": stat.st_mtime,
                "inferred_role": infer_role(path),
                "exists": True,
            }
        )
    files.sort(key=lambda item: (item["inferred_role"], item["path"]))
    report = {
        "project_root": str(PROJECT_ROOT),
        "search_terms": SEARCH_TERMS,
        "file_count": len(files),
        "files": files,
    }
    json_path = REPORT_DIR / "dive_file_inventory.json"
    txt_path = REPORT_DIR / "dive_file_inventory.txt"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["DIVE file inventory", ""]
    lines.append("path | size_bytes | inferred_role | exists")
    lines.append("-" * 120)
    for item in files:
        lines.append(
            f"{item['path']} | {item['size_bytes']} | "
            f"{item['inferred_role']} | {item['exists']}"
        )
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote {txt_path.relative_to(PROJECT_ROOT)}")
    print(f"[OK] wrote {json_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
