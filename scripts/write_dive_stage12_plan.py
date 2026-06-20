import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "data" / "reports"


def load_report(name):
    path = REPORT_DIR / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    audit = load_report("dive_dataset_audit.json")
    split = load_report("dive_split_report.json")
    compat = load_report("dive_evm_tokenizer_compatibility.json")
    can_reuse_tokenizer = bool(compat and compat.get("can_reuse_bjut_evm_bert"))
    split_ok = bool(split and split.get("leakage_status") == "ok")
    aligned_samples = audit.get("aligned_samples") if audit else None
    report = {
        "status": "ok" if can_reuse_tokenizer and split_ok else "warning",
        "dive_suitable_for_bjut_evm_tokenizer": can_reuse_tokenizer,
        "can_reuse_bjut_full_corpus_evm_bert_as_external_domain_pretraining": True,
        "dive_uses_bjut_labels": False,
        "dive_num_labels": 8,
        "dive_label_head_required": "reinitialize downstream classification head to 8 labels",
        "aligned_samples": aligned_samples,
        "strict_split_ready": split_ok,
        "tokenizer_compatibility_warning": compat.get("warnings", []) if compat else ["compatibility report missing"],
        "recommended_stage12b_models": [
            "EVM-BERT first-512 weighted baseline with num_labels=8",
            "EVM-BERT stride=256 masked_mean label-attention MIL with num_labels=8",
        ],
        "important_caveats": [
            "Do not merge BJUT 10 labels with DIVE 8 labels.",
            "BJUT-pretrained EVM-BERT is external-domain pretraining for DIVE if DIVE was not used in MLM.",
            "If later using DIVE full corpus for MLM, mark that experiment as DIVE transductive pretraining.",
            "Do not use DIVE test for threshold selection.",
        ],
        "next_stage_training_recommendation": (
            "Proceed to Stage 12B DIVE first-512 baseline"
            if can_reuse_tokenizer and split_ok
            else "Fix warnings before Stage 12B"
        ),
        "source_reports": [
            "data/reports/dive_file_inventory.txt",
            "data/reports/dive_dataset_audit.txt",
            "data/reports/dive_split_report.txt",
            "data/reports/dive_evm_tokenizer_compatibility.txt",
        ],
    }
    json_path = REPORT_DIR / "dive_stage12_plan.json"
    txt_path = REPORT_DIR / "dive_stage12_plan.txt"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["DIVE Stage 12 plan", ""]
    for key, value in report.items():
        if isinstance(value, (list, dict)):
            lines.append(f"{key}:")
            lines.append(json.dumps(value, indent=2))
        else:
            lines.append(f"{key}: {value}")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote {txt_path.relative_to(PROJECT_ROOT)}")
    print(f"[OK] wrote {json_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
