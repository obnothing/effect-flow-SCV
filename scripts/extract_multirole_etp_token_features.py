"""Extract aligned 64x768 features and Top-2 multi-role ETP token caches."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm
from transformers import BertForMaskedLM


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from effect_flow_schema import EFFECT_TYPES, build_token_units
from evm_tokenizer import EVMOpcodeTokenizer


SENTINEL = 255


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--splits", nargs="+", required=True, choices=["train", "valid", "test"])
    parser.add_argument("--allow-test-cache", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def load_model(config, device):
    checkpoint_dir = resolve(config["checkpoint_dir"])
    base = BertForMaskedLM.from_pretrained(checkpoint_dir / "hf_model", local_files_only=True)
    head = torch.load(checkpoint_dir / "etp_head.pt", map_location="cpu")
    etp = torch.nn.Linear(int(base.config.hidden_size), int(head["num_effect_types"]))
    etp.load_state_dict(head["etp_head_state_dict"])
    if int(etp.out_features) != len(EFFECT_TYPES):
        raise ValueError("ETP checkpoint does not use the 17-role ontology")
    return base.bert.to(device).eval(), etp.to(device).eval(), head


def chunks_for_row(row, tokenizer, config):
    raw = str(row.get("opcode", "") or "")
    units = build_token_units(raw, tokenizer)
    starts = list(range(0, len(units), int(config["chunk_stride"])))[: int(config["max_chunks"])]
    content_size = int(config["max_len"]) - 2
    return [units[start : start + content_size] for start in starts]


def prepare_chunk(units, tokenizer, max_len):
    tokens = [tokenizer.cls_token] + [unit.token for unit in units] + [tokenizer.sep_token]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    attention = [1] * len(input_ids)
    input_ids += [tokenizer.pad_token_id] * (max_len - len(input_ids))
    attention += [0] * (max_len - len(attention))
    return input_ids, attention, len(units)


def encode_batch(prepared, encoder, etp_head, thresholds, device):
    ids = torch.tensor([item[0] for item in prepared], dtype=torch.long, device=device)
    mask = torch.tensor([item[1] for item in prepared], dtype=torch.long, device=device)
    with torch.no_grad():
        hidden = encoder(input_ids=ids, attention_mask=mask, return_dict=True).last_hidden_state
        probabilities = torch.sigmoid(etp_head(hidden))
    content = mask.bool()
    content[:, 0] = False
    for batch_index, (_, _, unit_count) in enumerate(prepared):
        content[batch_index, unit_count + 1 :] = False
    weights = content.unsqueeze(-1).to(hidden.dtype)
    features = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
    top_probs, top_ids = probabilities.topk(k=2, dim=-1)
    threshold_tensor = torch.tensor(thresholds, dtype=top_probs.dtype, device=device)
    ids_out = torch.full(top_ids.shape, SENTINEL, dtype=torch.uint8, device=device)
    conf_out = torch.zeros(top_ids.shape, dtype=torch.uint8, device=device)
    first_valid = content
    second_valid = content & (top_probs[..., 1] >= threshold_tensor[top_ids[..., 1]])
    ids_out[..., 0][first_valid] = top_ids[..., 0][first_valid].to(torch.uint8)
    conf_out[..., 0][first_valid] = torch.round(top_probs[..., 0][first_valid] * 255).clamp(0, 255).to(torch.uint8)
    ids_out[..., 1][second_valid] = top_ids[..., 1][second_valid].to(torch.uint8)
    conf_out[..., 1][second_valid] = torch.round(top_probs[..., 1][second_valid] * 255).clamp(0, 255).to(torch.uint8)
    return features.cpu().half(), ids_out.cpu(), conf_out.cpu()


def main():
    args = parse_args()
    config = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = EVMOpcodeTokenizer.from_vocab_file(resolve(config["vocab_path"]))
    encoder, etp_head, checkpoint = load_model(config, device)
    batch_size = int(args.batch_size or config.get("batch_size", 32))
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    thresholds = checkpoint.get("etp_thresholds")
    if not thresholds or len(thresholds) != len(EFFECT_TYPES):
        raise ValueError("Missing 17 validation-selected ETP thresholds")
    feature_dir = resolve(config["feature_dir"])
    semantic_dir = resolve(config["token_semantic_dir"])
    feature_dir.mkdir(parents=True, exist_ok=True)
    semantic_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = resolve(config["checkpoint_dir"])
    checkpoint_hash = sha256_file(checkpoint_dir / "etp_head.pt")
    encoder_config_hash = sha256_file(checkpoint_dir / "hf_model" / "config.json")
    encoder_weights = next(
        (
            path for path in (
                checkpoint_dir / "hf_model" / "model.safetensors",
                checkpoint_dir / "hf_model" / "pytorch_model.bin",
            )
            if path.exists()
        ),
        None,
    )
    if encoder_weights is None:
        raise FileNotFoundError("ETP checkpoint is missing model.safetensors/pytorch_model.bin")
    encoder_weights_hash = sha256_file(encoder_weights)
    for split in args.splits:
        if split == "test" and not (bool(config.get("allow_test_cache", False)) or args.allow_test_cache):
            raise PermissionError("Test token cache is locked until final selection")
        rows = [json.loads(line) for line in resolve(config["data_dir"]).joinpath(f"{split}.jsonl").open("r", encoding="utf-8") if line.strip()]
        n = len(rows)
        max_chunks = int(config["max_chunks"])
        max_len = int(config["max_len"])
        features = torch.zeros((n, max_chunks, int(encoder.config.hidden_size)), dtype=torch.float16)
        chunk_mask = torch.zeros((n, max_chunks), dtype=torch.bool)
        top2_ids = torch.full((n, max_chunks, max_len, 2), SENTINEL, dtype=torch.uint8)
        top2_confidence = torch.zeros((n, max_chunks, max_len, 2), dtype=torch.uint8)
        ids = []
        binary = []
        labels = []
        pending = []

        def flush_pending():
            if not pending:
                return
            batch_features, batch_ids, batch_confidence = encode_batch(
                [item[2] for item in pending], encoder, etp_head, thresholds, device
            )
            for batch_index, (row_index, chunk_index, _) in enumerate(pending):
                features[row_index, chunk_index] = batch_features[batch_index]
                chunk_mask[row_index, chunk_index] = True
                top2_ids[row_index, chunk_index] = batch_ids[batch_index]
                top2_confidence[row_index, chunk_index] = batch_confidence[batch_index]
            pending.clear()

        for row_index, row in enumerate(tqdm(rows, desc=f"extract:{split}")):
            ids.append(str(row["id"]))
            current_labels = [int(value) for value in row["multi_labels"]]
            labels.append(current_labels)
            binary.append(int(any(current_labels)))
            for chunk_index, units in enumerate(chunks_for_row(row, tokenizer, config)):
                pending.append((row_index, chunk_index, prepare_chunk(units, tokenizer, max_len)))
                if len(pending) >= batch_size:
                    flush_pending()
        flush_pending()
        if not torch.isfinite(features.float()).all():
            raise ValueError(f"{split}: extracted features contain NaN/Inf")
        invalid_ids = (top2_ids < len(EFFECT_TYPES)) | (top2_ids == SENTINEL)
        if not bool(invalid_ids.all()):
            raise ValueError(f"{split}: extracted Top-2 cache contains an invalid role ID")
        if torch.any(top2_confidence[top2_ids == SENTINEL] != 0):
            raise ValueError(f"{split}: sentinel Top-2 slots must have zero confidence")
        inactive = ~chunk_mask
        if torch.any(top2_ids[inactive] != SENTINEL) or torch.any(top2_confidence[inactive] != 0):
            raise ValueError(f"{split}: padded chunks must use only sentinel Top-2 slots")
        common = {"ids": ids, "chunk_mask": chunk_mask, "binary_labels": torch.tensor(binary, dtype=torch.int8), "multi_labels": torch.tensor(labels, dtype=torch.int8)}
        feature_path = feature_dir / f"{split}.pt"
        semantic_path = semantic_dir / f"{split}.pt"
        torch.save({**common, "features": features, "report": {"feature_dim": int(encoder.config.hidden_size), "checkpoint_sha256": checkpoint_hash, "split": split}}, feature_path)
        torch.save({**common, "etp_top2_ids": top2_ids, "etp_top2_confidence": top2_confidence, "report": {"effect_type_names": EFFECT_TYPES, "etp_thresholds": thresholds, "checkpoint_sha256": checkpoint_hash, "split": split, "sentinel": SENTINEL}}, semantic_path)
        manifest = {
            "split": split,
            "test_cache_explicitly_unlocked": split == "test" and args.allow_test_cache,
            "checkpoint_dir": str(checkpoint_dir),
            "etp_head_sha256": checkpoint_hash,
            "encoder_config_sha256": encoder_config_hash,
            "encoder_weights_path": str(encoder_weights),
            "encoder_weights_sha256": encoder_weights_hash,
            "effect_type_names": EFFECT_TYPES,
            "effect_type_ids": {name: index for index, name in enumerate(EFFECT_TYPES)},
            "etp_thresholds": thresholds,
            "max_chunks": max_chunks,
            "max_len": max_len,
            "chunk_stride": int(config["chunk_stride"]),
            "sample_ids_in_order": ids,
            "feature": {"path": str(feature_path), "shape": list(features.shape), "dtype": str(features.dtype), "sha256": sha256_file(feature_path)},
            "token_semantic": {"path": str(semantic_path), "ids_shape": list(top2_ids.shape), "ids_dtype": str(top2_ids.dtype), "confidence_dtype": str(top2_confidence.dtype), "sha256": sha256_file(semantic_path)},
            "finite_features": True,
        }
        manifest_path = semantic_dir / f"{split}_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[OK] wrote {feature_path}")
        print(f"[OK] wrote {semantic_path}")
        print(f"[OK] wrote {manifest_path}")


if __name__ == "__main__":
    main()
