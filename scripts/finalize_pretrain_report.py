import argparse
import json
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRANSDUCTIVE_WARNING = (
    "This pretraining corpus uses opcode inputs from train/valid/test without "
    "labels. It is a transductive self-supervised setting."
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finalize EVM-BERT pretraining report from existing artifacts."
    )
    parser.add_argument(
        "--config",
        default="configs/pretrain_evm_bert_base_full_bjut.yaml",
        help="Path to pretraining config.",
    )
    return parser.parse_args()


def resolve_project_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def project_relative(path):
    path = Path(path)
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def load_json(path):
    path = resolve_project_path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_yaml(path):
    with resolve_project_path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def safe_checkpoint(path):
    path = resolve_project_path(path)
    if not path.exists():
        return None
    try:
        return torch.load(path, map_location="cpu")
    except Exception:
        return None


def read_history(config):
    result_dir = resolve_project_path(config["result_dir"])
    history_path = result_dir / "pretrain_epoch_history.json"
    if not history_path.exists():
        return []
    return json.loads(history_path.read_text(encoding="utf-8"))


def build_report(config):
    report_dir = resolve_project_path(config.get("report_dir", "data/reports"))
    corpus_report = (
        load_json(report_dir / "pretrain_corpus_full_bjut_report.json")
        or load_json(report_dir / "pretrain_evm_bert_base_full_bjut_corpus_report.json")
        or {}
    )
    checkpoint_dir = resolve_project_path(config["checkpoint_dir"])
    hf_model_dir = checkpoint_dir / "hf_model"
    best_path = checkpoint_dir / "best_mlm_loss.pt"
    last_path = checkpoint_dir / "last.pt"
    best_ckpt = safe_checkpoint(best_path)
    last_ckpt = safe_checkpoint(last_path)
    history = read_history(config)

    hf_config = load_json(hf_model_dir / "config.json") or {}
    completed_epochs = len(history)
    final_mlm_loss = history[-1]["train_mlm_loss"] if history else "unknown"
    best_from_history = min(
        history,
        key=lambda row: row.get("train_mlm_loss", float("inf")),
    ) if history else {}

    report = {
        "experiment_name": config.get("experiment_name"),
        "is_transductive_pretraining": True,
        "use_labels": False,
        "transductive_warning": TRANSDUCTIVE_WARNING,
        "original_contracts": corpus_report.get(
            "original_contract_samples",
            corpus_report.get("total_samples", "unknown"),
        ),
        "generated_mlm_chunks": corpus_report.get("generated_mlm_chunks", "unknown"),
        "average_chunks_per_contract": corpus_report.get(
            "average_chunks_per_contract", "unknown"
        ),
        "max_chunks_per_contract": corpus_report.get(
            "max_chunks_per_contract", "unknown"
        ),
        "truncated_contracts": corpus_report.get(
            "truncated_contract_count", "unknown"
        ),
        "mean_token_coverage": corpus_report.get(
            "covered_token_ratio_mean", "unknown"
        ),
        "covered_token_ratio_p50": corpus_report.get(
            "covered_token_ratio_p50", "unknown"
        ),
        "covered_token_ratio_p90": corpus_report.get(
            "covered_token_ratio_p90", "unknown"
        ),
        "completed_epochs": completed_epochs,
        "epoch_mlm_losses": [
            {
                "epoch": row.get("epoch"),
                "mlm_loss": row.get("train_mlm_loss"),
                "global_step": row.get("global_step"),
                "learning_rate": row.get("learning_rate"),
            }
            for row in history
        ],
        "final_mlm_loss": final_mlm_loss,
        "best_mlm_loss": (
            best_ckpt.get("mlm_loss")
            if isinstance(best_ckpt, dict) and best_ckpt.get("mlm_loss") is not None
            else best_from_history.get("train_mlm_loss", "unknown")
        ),
        "best_epoch": (
            best_ckpt.get("epoch")
            if isinstance(best_ckpt, dict) and best_ckpt.get("epoch") is not None
            else best_from_history.get("epoch", "unknown")
        ),
        "last_epoch": (
            last_ckpt.get("epoch")
            if isinstance(last_ckpt, dict) and last_ckpt.get("epoch") is not None
            else (history[-1].get("epoch") if history else "unknown")
        ),
        "hf_model_saved": (hf_model_dir / "config.json").exists()
        and (
            (hf_model_dir / "model.safetensors").exists()
            or (hf_model_dir / "pytorch_model.bin").exists()
        ),
        "hf_model_path": project_relative(hf_model_dir),
        "hf_config_path": project_relative(hf_model_dir / "config.json"),
        "hf_vocab_copy_saved": (hf_model_dir / "evm_vocab.json").exists(),
        "best_mlm_checkpoint_saved": best_path.exists(),
        "best_mlm_checkpoint_path": project_relative(best_path),
        "last_checkpoint_saved": last_path.exists(),
        "last_checkpoint_path": project_relative(last_path),
        "vocab_path": config.get("vocab_path"),
        "vocab_size": hf_config.get("vocab_size", "unknown"),
        "hidden_size": hf_config.get("hidden_size", "unknown"),
        "num_hidden_layers": hf_config.get("num_hidden_layers", "unknown"),
        "num_attention_heads": hf_config.get("num_attention_heads", "unknown"),
        "intermediate_size": hf_config.get("intermediate_size", "unknown"),
        "max_position_embeddings": hf_config.get("max_position_embeddings", "unknown"),
    }
    return report


def write_report(config, report):
    report_dir = resolve_project_path(config.get("report_dir", "data/reports"))
    report_dir.mkdir(parents=True, exist_ok=True)
    txt_path = report_dir / "pretrain_evm_bert_base_full_bjut_report.txt"
    json_path = report_dir / "pretrain_evm_bert_base_full_bjut_report.json"
    lines = ["EVM-BERT-base full BJUT pretraining final report", ""]
    for key, value in report.items():
        if key == "epoch_mlm_losses":
            lines.append("epoch_mlm_losses:")
            for row in value:
                lines.append(
                    f"- epoch {row['epoch']}: mlm_loss={row['mlm_loss']} "
                    f"global_step={row['global_step']} lr={row['learning_rate']}"
                )
        else:
            lines.append(f"{key}: {value}")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[OK] wrote {project_relative(txt_path)}")
    print(f"[OK] wrote {project_relative(json_path)}")


def main():
    args = parse_args()
    config = load_yaml(args.config)
    report = build_report(config)
    write_report(config, report)


if __name__ == "__main__":
    main()
