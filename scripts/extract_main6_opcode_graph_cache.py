"""Extract full-coverage EVM basic-block features and CSDG caches."""

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm
from transformers import BertForMaskedLM

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evm_control_stack_graph import build_evm_graph, build_token_spans  # noqa: E402
from evm_tokenizer import EVMOpcodeTokenizer  # noqa: E402


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_jsonl(path):
    with resolve(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            if line.strip():
                yield line_number, json.loads(line)


def encode_window(tokenizer, tokens, max_len):
    content = tokens[: max_len - 2]
    values = [tokenizer.cls_token] + content + [tokenizer.sep_token]
    ids = tokenizer.convert_tokens_to_ids(values)
    mask = [1] * len(ids)
    pad = max_len - len(ids)
    ids.extend([tokenizer.pad_token_id] * pad)
    mask.extend([0] * pad)
    return ids, mask


def extract_nodes(encoder, tokenizer, opcode, graph, config, device):
    tokens, _ = build_token_spans(opcode, tokenizer)
    node_count = len(graph["nodes"])
    hidden_size = int(encoder.config.hidden_size)
    token_sums = torch.zeros(len(tokens), hidden_size, dtype=torch.float32)
    token_counts = torch.zeros(len(tokens), dtype=torch.float32)
    content_size = int(config.get("graph_chunk_content_size", 510))
    stride = int(config.get("graph_chunk_stride", 256))
    max_len = content_size + 2
    windows = []
    starts = list(range(0, len(tokens), stride)) if tokens else [0]
    for start in starts:
        windows.append((start, encode_window(tokenizer, tokens[start : start + content_size], max_len)))
        if start + content_size >= len(tokens):
            break
    for offset in range(0, len(windows), int(config.get("graph_batch_size", 32))):
        selected = windows[offset : offset + int(config.get("graph_batch_size", 32))]
        input_ids = torch.tensor([item[1][0] for item in selected], dtype=torch.long, device=device)
        attention_mask = torch.tensor([item[1][1] for item in selected], dtype=torch.long, device=device)
        with torch.no_grad():
            if bool(config.get("fp16", True)) and device.type == "cuda":
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    hidden = encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            else:
                hidden = encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        hidden = hidden.float().cpu()
        for local_index, (start, _) in enumerate(selected):
            left, right = start, min(len(tokens), start + content_size)
            if right > left:
                token_sums[left:right] += hidden[local_index, 1 : 1 + right - left]
                token_counts[left:right] += 1.0
    token_features = token_sums / token_counts.clamp_min(1.0).unsqueeze(1)
    features, local_features, offsets = [], [], [0]
    for node in graph["nodes"]:
        left, right = int(node["token_start"]), int(node["token_end"])
        current = token_features[left:right]
        features.append(current.mean(dim=0) if current.numel() else torch.zeros(hidden_size))
        local_features.append(current)
        offsets.append(offsets[-1] + current.shape[0])
    feature_tensor = torch.stack(features) if features else torch.empty((0, hidden_size))
    local_tensor = torch.cat(local_features, dim=0) if local_features else torch.empty((0, hidden_size))
    valid = torch.tensor([right > left and bool(token_counts[left:right].all()) for left, right in [(int(n["token_start"]), int(n["token_end"])) for n in graph["nodes"]]], dtype=torch.bool)
    return feature_tensor.to(torch.float16), valid, local_tensor.to(torch.float16), torch.tensor(offsets, dtype=torch.long), len(windows), len(tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_main6_opcode_csdg.yaml")
    parser.add_argument("--variant", default=None, help="Optional cache-specific CSDG variant.")
    parser.add_argument("--splits", nargs="+", choices=["train", "valid", "test"], default=["train", "valid"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-test-cache", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
    if "common" in config:
        config = {**config["common"], **config.get("graph_cache", {}), **(config.get("variants", {}).get(args.variant, {}) if args.variant else {})}
    tokenizer = EVMOpcodeTokenizer.from_vocab_file(resolve(config["vocab_path"]))
    model_path = resolve(config["hf_model_path"])
    model = BertForMaskedLM.from_pretrained(model_path, local_files_only=True)
    if model.config.vocab_size != len(tokenizer):
        raise ValueError("EVM-BERT vocabulary and tokenizer size differ")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = model.bert.to(device).eval()
    model_hashes = {}
    for name in ("config.json", "pytorch_model.bin"):
        path = model_path / name
        if path.exists():
            model_hashes[name] = sha256(path)
    output_dir = resolve(config["graph_cache_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": config["train_path"],
        "valid": config["valid_path"],
        "test": config["test_path"],
    }
    label_names = list(config["label_names"])
    for split in args.splits:
        if split == "test":
            if not args.allow_test_cache or os.environ.get("ALLOW_TEST") != "1":
                raise PermissionError("Test graph cache is locked until ALLOW_TEST=1 final selection")
        output_path = output_dir / f"{split}.pt"
        if output_path.exists() and not args.overwrite:
            print(f"[OK] {output_path} exists; skipping")
            continue
        ids = []
        labels = []
        records = []
        reports = []
        for line_number, item in tqdm(iter_jsonl(paths[split]), desc=f"graph:{split}"):
            sample_id = str(item.get("id") or f"{split}:{line_number}")
            opcode = str(item.get("opcode") or "")
            graph = build_evm_graph(
                opcode,
                tokenizer=tokenizer,
                include_storage_edges=bool(config.get("include_storage_edges", False)),
                max_producers=int(config.get("max_stack_producers", 4)),
                max_worklist_steps=int(config.get("max_stack_worklist_steps", 20000)),
                max_instruction_visits=int(config.get("max_stack_instruction_visits", 250000)),
            )
            graph_view = graph["instruction_value"] if str(config.get("graph_granularity", "basic_block")) == "instruction_value" else graph
            features, feature_mask, local_features, local_offsets, windows, token_count = extract_nodes(
                encoder, tokenizer, opcode, graph_view, config, device
            )
            ids.append(sample_id)
            labels.append([float(value) for value in item["multi_labels"]])
            edge_index = torch.tensor(
                [[edge["src"] for edge in graph_view["edges"]], [edge["dst"] for edge in graph_view["edges"]]],
                dtype=torch.long,
            ) if graph_view["edges"] else torch.empty((2, 0), dtype=torch.long)
            edge_type = torch.tensor([edge["type"] for edge in graph_view["edges"]], dtype=torch.long)
            records.append({
                "node_features": features,
                "node_mask": feature_mask.bool(),
                "node_local_features": local_features,
                "node_local_offsets": local_offsets,
                "node_type": torch.tensor([int(node.get("node_type", 0)) for node in graph_view["nodes"]], dtype=torch.long),
                "edge_index": edge_index,
                "edge_type": edge_type,
                "block_ranges": [
                    [int(node["pc_start"]), int(node["pc_end"]), int(node["token_start"]), int(node["token_end"])]
                    for node in graph_view["nodes"]
                ],
            })
            report = dict(graph["report"])
            report.update({
                "id": sample_id,
                "windows": windows,
                "feature_node_coverage": float(feature_mask.float().mean().item()) if len(feature_mask) else 1.0,
                "label_width": len(item["multi_labels"]),
            })
            reports.append(report)
        if len(label_names) != len(labels[0]) if labels else False:
            raise ValueError("Graph cache label width differs from Main-6 labels")
        payload = {
            "schema": "main6_opcode_csdg_v2",
            "ids": ids,
            "records": records,
            "multi_labels": torch.tensor(labels, dtype=torch.float32),
            "manifest": {
                "route": "DIVE Main6 Opcode-CSDG",
                "split": split,
                "label_names": label_names,
                "seed": 42,
                "train_only_pretraining": True,
                "test_labels_read": split == "test",
                "no_label_retrieval": True,
                "hf_model_path": str(model_path),
                "model_hashes": model_hashes,
                "tokenizer_path": str(resolve(config["vocab_path"])),
                "tokenizer_sha256": sha256(resolve(config["vocab_path"])),
                "edge_types": graph["edge_types"] if records else {},
                "graph_pooling_cache": "token_features_available",
                "graph_granularity": str(config.get("graph_granularity", "basic_block")),
                "max_stack_producers": int(config.get("max_stack_producers", 4)),
                "reports": reports,
            },
        }
        torch.save(payload, output_path)
        sidecar = {
            "schema": payload["schema"],
            "split": split,
            "cache_path": str(output_path),
            "cache_file_sha256": sha256(output_path),
            "samples": len(ids),
            "node_feature_dtype": "float16",
            "node_feature_width": 768,
            "label_width": len(label_names),
            "edge_type_count": len(payload["manifest"]["edge_types"]),
            "model_hashes": model_hashes,
        }
        sidecar_path = output_dir / f"{split}.manifest.json"
        sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
        print(f"[OK] wrote {output_path}")


if __name__ == "__main__":
    main()
