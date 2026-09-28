"""Eight-label E3/P11 architecture and train/valid loader preflight."""

import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evm_tokenizer import EVMOpcodeTokenizer
from light_label_data import collate_light_label
from polarity_query_model import build_model, loss_terms


def first_row(split):
    first = None
    seen = set()
    count = 0
    with (ROOT / "data/processed/DIVE_8_opcode_random_split" / f"{split}.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if first is None:
                first = row
            if len(row["multi_labels"]) != 8 or row["id"] in seen:
                raise ValueError(f"invalid labels or duplicate ID in {split}: {row['id']}")
            seen.add(row["id"])
            count += 1
    return first, count


def main():
    config = yaml.safe_load((ROOT / "configs/spor_phase25/E3_P11_8_A0.yaml").read_text(encoding="utf-8"))
    tokenizer = EVMOpcodeTokenizer.from_vocab_file(ROOT / config["vocab_path"])
    checked = {split: first_row(split) for split in ("train", "valid")}
    rows = {split: value[0] for split, value in checked.items()}
    assert checked["train"][1] == 17864 and checked["valid"][1] == 2233
    for split, row in rows.items():
        assert len(row["multi_labels"]) == 8 and len(tokenizer.encode(row["opcode"], add_special_tokens=False)) > 0, split
    torch.manual_seed(42)
    model = build_model("P11", config, len(tokenizer), tokenizer.pad_token_id)
    assert model.embedding.embedding_dim == 512
    assert model.encoder.hidden_size == 512 and model.encoder.bidirectional
    assert model.output_dim == 1024 and model.query_dim == 512
    assert model.queries.shape == (8, 2, 512)
    assert model.cross_attention.num_heads == 8 and model.cross_attention.head_dim == 64
    assert model.cross_attention.k_proj.weight.shape == (512, 1024)
    assert model.cross_attention.v_proj.weight.shape == (512, 1024)
    tokens = [tokenizer.encode(row["opcode"], add_special_tokens=False)[:32] for row in rows.values()]
    items = [{"id": row["id"], "input_ids": torch.tensor(seq), "length": len(seq),
              "original_length": len(seq), "labels": torch.tensor(row["multi_labels"], dtype=torch.float32)}
             for row, seq in zip(rows.values(), tokens)]
    batch = collate_light_label(items, tokenizer.pad_token_id)
    ids, lengths, mask, targets = batch["input_ids"], batch["lengths"], batch["mask"], batch["labels"]
    output = model(ids, lengths, mask, diagnostics=True)
    assert output["logits"].shape == (2, 8)
    assert output["energies"].shape == (2, 8, 2)
    assert output["representations"].shape == (2, 8, 2, 512)
    assert output["attention"].shape == (2, 8, 2, ids.shape[1])
    loss, classification, polarity = loss_terms(output, targets, torch.ones(8), config["auxiliary_weight"],
        positive_label_multiplier=config["positive_auxiliary_label_multiplier"],
        negative_label_multiplier=config["negative_auxiliary_label_multiplier"],
        dos_soft_targets={"positive_high": .8, "positive_low": .1, "negative_low": .1, "negative_high": .8})
    loss.backward()
    assert torch.isfinite(loss) and model.queries.grad is not None and model.queries.grad.abs().sum() > 0
    summary = {"route": "E3_P11_8_A0", "dataset": config["data_dir"], "train_valid_loader_checked": True,
               "train_rows": checked["train"][1], "valid_rows": checked["valid"][1],
               "test_checked": False, "parameter_count": sum(p.numel() for p in model.parameters()),
               "embedding_dim": 512, "bigru_output_dim": 1024, "query_shape": list(model.queries.shape),
               "kv_projection": [1024, 512], "attention_heads": 8, "head_dim": 64,
               "forward_backward_smoke": "passed", "loss": float(loss.item())}
    path = ROOT / "results/spor_phase25/E3_P11_8_A0/smoke.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
