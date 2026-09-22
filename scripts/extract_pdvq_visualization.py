"""Extract validation-only P11 evidence features and attention on GPU."""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_polarity_queries as base
from evm_tokenizer import EVMOpcodeTokenizer
from light_label_data import collate_light_label
from polarity_query_model import build_model


CONFIG_PATH = ROOT / "configs/light_label/polarity_queries_followup_p11.yaml"
CHECKPOINT_PATH = ROOT / "checkpoints/light_label/polarity_queries_followup_p11/P11/best.pt"
OUTPUT = ROOT / "results/visualization"
LABELS = ["Reentrancy", "Access Control", "Arithmetic", "Unchecked Return Values", "DoS", "Time manipulation"]
ATTENTION_LABELS = [0, 4, 1]


def forward_with_hidden(model, input_ids, lengths, mask, attention=False):
    hidden = model.encode_tokens(input_ids, lengths, mask)
    queries = model.queries.flatten(0, 1).unsqueeze(0).expand(input_ids.shape[0], -1, -1)
    evidence, weights = model.cross_attention(
        queries,
        hidden,
        hidden,
        key_padding_mask=~mask,
        need_weights=attention,
        average_attn_weights=False,
    )
    representation = evidence.reshape(input_ids.shape[0], model.num_labels, model.polarities, model.query_dim)
    logits, energies = model.score(representation)
    result = {"hidden": hidden, "representations": representation, "logits": logits, "energies": energies}
    if attention:
        result["attention"] = weights.mean(1).reshape(input_ids.shape[0], model.num_labels, model.polarities, -1)
    return result


def choose_attention_contracts(valid, raw):
    raw_by_id = {str(row["id"]): row for row in raw}
    rows = []
    for index, contract_id in enumerate(valid.ids):
        raw_row = raw_by_id[str(contract_id)]
        rows.append({
            "dataset_index": index,
            "id": str(contract_id),
            "labels": valid.labels[valid.indices[index]].int().tolist(),
            "original_length": int(valid.original_lengths[valid.indices[index]]),
            "opcode": raw_row.get("opcode", ""),
        })
    selected, used = [], set()
    # Two readable positive examples for each focal vulnerability. Prefer
    # contracts with fewer co-occurring labels and a moderate opcode length.
    for target in (0, 4, 1, 0, 4, 1):
        candidates = [row for row in rows if row["labels"][target] == 1 and row["id"] not in used]
        candidates.sort(key=lambda row: (sum(row["labels"]), abs(row["original_length"] - 512), row["id"]))
        if not candidates:
            raise RuntimeError(f"No validation contract for {LABELS[target]}")
        item = candidates[0]
        item["focus_label"] = LABELS[target]
        selected.append(item); used.add(item["id"])
    return {item["id"]: item for item in selected}


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for feature extraction")
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(CHECKPOINT_PATH)
    config = base.load_config(CONFIG_PATH)
    if config.get("allow_test"):
        raise ValueError("Visualization route requires a test-locked config")
    train, valid = base.datasets(config)
    raw = [json.loads(line) for line in (ROOT / config["data_dir"] / "valid.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if len(raw) != len(valid):
        raise ValueError("Validation raw/cache length mismatch")
    tokenizer = EVMOpcodeTokenizer.from_vocab_file(ROOT / config["vocab_path"])
    model = build_model("P11", config, len(tokenizer), tokenizer.pad_token_id).cuda()
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    selected = choose_attention_contracts(valid, raw)
    loader = DataLoader(valid, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=lambda items: collate_light_label(items, tokenizer.pad_token_id))
    z_pos, z_neg, z_diff, means, labels, ids, lengths = [], [], [], [], [], [], []
    selected_attention, selected_top = {}, {}
    label_cosine_sum = torch.zeros(6, 6, dtype=torch.float64)
    polarity_cosine_sum = torch.zeros(6, dtype=torch.float64)
    attention_count = 0
    with torch.no_grad():
        for batch in loader:
            contract_id = str(batch["ids"][0])
            inputs = batch["input_ids"].cuda(non_blocking=True)
            mask = batch["mask"].cuda(non_blocking=True)
            # Attention weights are needed for validation-wide similarity
            # statistics; only the six selected contracts are persisted.
            need_attention = True
            with torch.autocast("cuda", dtype=torch.float16, enabled=config["amp"]):
                out = forward_with_hidden(model, inputs, batch["lengths"], mask, attention=need_attention)
            representations = out["representations"].float().cpu()[0]
            positive, negative = representations[:, 0], representations[:, 1]
            z_pos.append(positive.numpy()); z_neg.append(negative.numpy()); z_diff.append((positive - negative).numpy())
            hidden = out["hidden"].float()
            mean = (hidden * mask.unsqueeze(-1).to(hidden.dtype)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
            means.append(mean.cpu()[0].numpy())
            labels.append(batch["labels"][0].numpy()); ids.append(contract_id); lengths.append(int(batch["original_lengths"][0]))
            length = int(mask[0].sum())
            attention = out["attention"].float().cpu()[0, :, :, :length]
            averaged = attention.mean(1)
            normalized = torch.nn.functional.normalize(averaged, dim=1)
            label_cosine_sum += normalized @ normalized.T
            polarity_cosine_sum += torch.nn.functional.cosine_similarity(attention[:, 0], attention[:, 1], dim=1).double()
            attention_count += 1
            if contract_id in selected:
                selected_attention[contract_id] = attention.numpy()
                opcodes = selected[contract_id]["opcode"].split()[:length]
                top = {}
                for label_index in ATTENTION_LABELS:
                    for polarity_index, polarity_name in enumerate(("positive", "negative")):
                        top_indices = torch.topk(attention[label_index, polarity_index], k=min(10, length)).indices.tolist()
                        top[f"{LABELS[label_index]}_{polarity_name}"] = [
                            {"position": int(position), "opcode": opcodes[position] if position < len(opcodes) else "<missing>",
                             "attention": float(attention[label_index, polarity_index, position])}
                            for position in top_indices
                        ]
                selected_top[contract_id] = top
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pos = np.asarray(z_pos, dtype=np.float32); neg = np.asarray(z_neg, dtype=np.float32); diff = np.asarray(z_diff, dtype=np.float32)
    np.save(OUTPUT / "pdvq_tsne_features.npy", diff.reshape(len(diff), -1))
    np.savez_compressed(OUTPUT / "pdvq_evidence_features.npz", z_pos=pos, z_neg=neg, z_label=diff,
                        mean_pool=np.asarray(means, dtype=np.float32), labels=np.asarray(labels, dtype=np.int8),
                        ids=np.asarray(ids), original_lengths=np.asarray(lengths, dtype=np.int32))
    attention_ids = list(selected_attention)
    np.savez_compressed(OUTPUT / "attention_selected_contracts.npz", ids=np.asarray(attention_ids),
                        attentions=np.asarray([selected_attention[item] for item in attention_ids], dtype=object))
    (OUTPUT / "attention_top_opcodes.json").write_text(json.dumps({"contracts": selected, "top_opcodes": selected_top}, indent=2), encoding="utf-8")
    statistics = {
        "attention_contract_count": attention_count,
        "label_attention_cosine": (label_cosine_sum / max(attention_count, 1)).tolist(),
        "positive_negative_attention_cosine": (polarity_cosine_sum / max(attention_count, 1)).tolist(),
        "labels": LABELS,
        "attention_labels": [LABELS[index] for index in ATTENTION_LABELS],
        "test_checked": False,
        "interpretation": "Attention similarities summarize model distributions and do not establish ground-truth opcode localization.",
    }
    (OUTPUT / "attention_statistics.json").write_text(json.dumps(statistics, indent=2), encoding="utf-8")
    manifest = {
        "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)), "dataset": config["data_dir"],
        "split": "validation", "seed": int(config["seed"]), "contracts": len(ids), "feature_shape": list(diff.shape),
        "test_checked": False,
    }
    (OUTPUT / "extraction_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
