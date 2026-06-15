import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch
import yaml
from transformers import BertForMaskedLM


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check pretrained EVM-BERT artifacts before downstream training."
    )
    parser.add_argument(
        "--config",
        default="configs/train_evm_bert_weighted_server.yaml",
        help="Downstream config containing hf_model_path and vocab_path.",
    )
    return parser.parse_args()


def resolve_path(path):
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


def load_yaml(path):
    with resolve_path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_evm_vocab(path):
    path = resolve_path(path)
    if not path.exists():
        raise FileNotFoundError(f"EVM vocab not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    token_to_id = data.get("token_to_id")
    if not isinstance(token_to_id, dict):
        raise ValueError(f"{path} does not contain token_to_id.")
    return data, token_to_id


def tensor_checksum(tensor):
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def parameter_count(module):
    return sum(param.numel() for param in module.parameters())


def ensure_hf_vocab_copy(hf_model_path, vocab_path):
    hf_vocab_path = hf_model_path / "evm_vocab.json"
    if hf_vocab_path.exists():
        return hf_vocab_path, False
    shutil.copy2(vocab_path, hf_vocab_path)
    return hf_vocab_path, True


def build_report(config):
    hf_model_path = resolve_path(config["hf_model_path"])
    vocab_path = resolve_path(config["vocab_path"])
    report_dir = resolve_path(config.get("report_dir", "data/reports"))

    if not hf_model_path.exists():
        raise FileNotFoundError(f"hf_model_path does not exist: {hf_model_path}")
    config_path = hf_model_path / "config.json"
    safetensors_path = hf_model_path / "model.safetensors"
    bin_path = hf_model_path / "pytorch_model.bin"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json: {config_path}")
    if not safetensors_path.exists() and not bin_path.exists():
        raise FileNotFoundError(
            f"Missing model weights: expected {safetensors_path} or {bin_path}"
        )

    vocab_data, token_to_id = load_evm_vocab(vocab_path)
    hf_vocab_path, copied_vocab = ensure_hf_vocab_copy(hf_model_path, vocab_path)

    mlm_model = BertForMaskedLM.from_pretrained(
        hf_model_path,
        local_files_only=True,
    )
    encoder = mlm_model.bert
    model_vocab_size = int(mlm_model.config.vocab_size)
    evm_vocab_size = len(token_to_id)
    vocab_size_matches = model_vocab_size == evm_vocab_size
    if not vocab_size_matches:
        raise ValueError(
            "EVM vocab size mismatch: "
            f"model.config.vocab_size={model_vocab_size}, "
            f"len(evm_vocab)={evm_vocab_size}. Stop before downstream training."
        )

    embedding_checksum = tensor_checksum(encoder.embeddings.word_embeddings.weight)
    first_attention = encoder.encoder.layer[0].attention.self
    first_attention_checksums = {
        "query_weight": tensor_checksum(first_attention.query.weight),
        "key_weight": tensor_checksum(first_attention.key.weight),
        "value_weight": tensor_checksum(first_attention.value.weight),
    }
    first_attention_combined = hashlib.sha256(
        "".join(first_attention_checksums.values()).encode("utf-8")
    ).hexdigest()

    report = {
        "status": "ok",
        "hf_model_path": project_relative(hf_model_path),
        "config_json_exists": config_path.exists(),
        "model_safetensors_exists": safetensors_path.exists(),
        "pytorch_model_bin_exists": bin_path.exists(),
        "vocab_path": project_relative(vocab_path),
        "hf_vocab_path": project_relative(hf_vocab_path),
        "hf_vocab_copied_from_project_vocab": copied_vocab,
        "vocab_size": evm_vocab_size,
        "model_config_vocab_size": model_vocab_size,
        "vocab_size_matches_model": vocab_size_matches,
        "hidden_size": int(mlm_model.config.hidden_size),
        "num_hidden_layers": int(mlm_model.config.num_hidden_layers),
        "num_attention_heads": int(mlm_model.config.num_attention_heads),
        "intermediate_size": int(mlm_model.config.intermediate_size),
        "max_position_embeddings": int(mlm_model.config.max_position_embeddings),
        "total_parameters": parameter_count(mlm_model),
        "encoder_parameters": parameter_count(encoder),
        "embedding_weight_sha256": embedding_checksum,
        "first_attention_sha256": first_attention_combined,
        "first_attention_components_sha256": first_attention_checksums,
        "encoder_weight_source": project_relative(hf_model_path),
        "loaded_with": "BertForMaskedLM.from_pretrained(...).bert",
        "local_files_only": True,
        "is_transductive_pretraining": bool(
            config.get("is_transductive_pretraining", True)
        ),
        "transductive_warning": (
            "EVM-BERT was pretrained on unlabeled BJUT train/valid/test opcodes; "
            "use this result as transductive domain pretraining, not a strict "
            "inductive baseline."
        ),
        "preserved_operand_count": len(vocab_data.get("preserved_operands", [])),
    }

    report_dir.mkdir(parents=True, exist_ok=True)
    txt_path = report_dir / "check_evm_bert_pretrained_model.txt"
    json_path = report_dir / "check_evm_bert_pretrained_model.json"
    lines = ["EVM-BERT pretrained model check", ""]
    for key, value in report.items():
        if key == "first_attention_components_sha256":
            lines.append("first_attention_components_sha256:")
            for name, checksum in value.items():
                lines.append(f"- {name}: {checksum}")
        else:
            lines.append(f"{key}: {value}")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[OK] wrote {project_relative(txt_path)}")
    print(f"[OK] wrote {project_relative(json_path)}")
    return report


def main():
    args = parse_args()
    try:
        config = load_yaml(args.config)
        report = build_report(config)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
    print(
        "[OK] pretrained EVM-BERT loaded; "
        f"vocab_size={report['vocab_size']} hidden_size={report['hidden_size']}"
    )


if __name__ == "__main__":
    main()
