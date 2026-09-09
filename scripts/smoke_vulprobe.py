"""Focused architecture checks for VulProbe-V1 without project checkpoints."""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import BertConfig, BertForMaskedLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vulprobe_dataset import VulProbeContractDataset, collate_contracts  # noqa: E402
from vulprobe_model import VulProbeModel, masked_logmeanexp  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/vulprobe_v1/smoke_report.json")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        config = BertConfig(vocab_size=16, hidden_size=32, num_hidden_layers=3,
                            num_attention_heads=4, intermediate_size=64)
        BertForMaskedLM(config).save_pretrained(root / "bert")
        vocab = {token: idx for idx, token in enumerate([
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "PUSH1", "ADD", "CALL",
            "SSTORE", "JUMP", "0x00", "0x01", "<HEX1>", "<HEX2>", "<HEX4>", "<HEX_SHORT>"])}
        (root / "vocab.json").write_text(json.dumps({"token_to_id": vocab}), encoding="utf-8")
        rows = [
            {"id":"a","opcode":"PUSH1 0x01 CALL SSTORE ADD","binary_label":1,"multi_labels":[1,0,0,0,0,0]},
            {"id":"b","opcode":"","binary_label":0,"multi_labels":[0,0,0,0,0,0]},
        ]
        (root / "train.jsonl").write_text("\n".join(json.dumps(x) for x in rows)+"\n", encoding="utf-8")
        data = VulProbeContractDataset(root / "train.jsonl", root / "vocab.json", max_len=8, stride=3, max_chunks=2)
        batch = collate_contracts([data[0], data[1]])
        assert batch["input_ids"].shape == (2, 2, 8)
        assert batch["content_mask"][1, 0].sum() == 0

        values = torch.tensor([[[1.0], [3.0], [100.0]]])
        pooled = masked_logmeanexp(values, torch.tensor([[1, 1, 0]], dtype=torch.bool), tau=1.0)
        expected = torch.log((torch.exp(torch.tensor(1.0)) + torch.exp(torch.tensor(3.0))) / 2)
        assert torch.allclose(pooled.squeeze(), expected)

        for variant in ("b0_shared_representation", "b1_shared_probe", "b2_label_probe"):
            model = VulProbeModel(root / "bert", variant, num_labels=6, num_heads=4, encoder_chunk_batch=2)
            model.set_training_stage("frozen")
            output = model(batch["input_ids"], batch["attention_mask"], batch["content_mask"], batch["chunk_mask"])
            assert output["logits"].shape == (2, 6)
            assert torch.isfinite(output["logits"]).all()
            assert not any(parameter.grad is not None for parameter in model.encoder.parameters())

        model = VulProbeModel(root / "bert", "b2_label_probe", num_labels=6, num_heads=4, encoder_chunk_batch=2)
        model.set_training_stage("frozen")
        output = model(batch["input_ids"], batch["attention_mask"], batch["content_mask"], batch["chunk_mask"])
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            mixed_output = model(
                batch["input_ids"], batch["attention_mask"],
                batch["content_mask"], batch["chunk_mask"],
            )
        assert mixed_output["logits"].shape == (2, 6)
        assert torch.isfinite(mixed_output["logits"]).all()
        valid_content = batch["content_mask"].reshape(-1, 8)[output["valid_chunk_flat_mask"]]
        nonempty = valid_content.any(dim=1)
        assert torch.all(
            output["valid_token_attention"][nonempty].masked_select(
                ~valid_content[nonempty].unsqueeze(1)
            ) == 0
        )
        label_id = 2
        loss = F.binary_cross_entropy_with_logits(output["logits"][:, label_id], batch["multi_labels"][:, label_id])
        loss.backward()
        norms = model.probes.grad.norm(dim=1)
        assert norms[label_id] > 0
        assert torch.all(norms[torch.arange(6) != label_id] == 0)
        assert not any(parameter.grad is not None for parameter in model.encoder.parameters())

        checkpoint = root / "roundtrip.pt"
        torch.save(model.state_dict(), checkpoint)
        restored = VulProbeModel(
            root / "bert", "b2_label_probe", num_labels=6,
            num_heads=4, encoder_chunk_batch=2,
        )
        restored.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=True)
        restored.eval()
        model.eval()
        with torch.no_grad():
            expected = model(
                batch["input_ids"], batch["attention_mask"],
                batch["content_mask"], batch["chunk_mask"],
            )["logits"]
            actual = restored(
                batch["input_ids"], batch["attention_mask"],
                batch["content_mask"], batch["chunk_mask"],
            )["logits"]
        assert torch.equal(expected, actual)

        model.set_training_stage("joint", unfreeze_last_layers=2)
        trainable_layers = [any(p.requires_grad for p in layer.parameters()) for layer in model.encoder.encoder.layer]
        assert trainable_layers == [False, True, True]
        data.close()
        report = {
            "status": "ok",
            "variants_checked": ["b0_shared_representation", "b1_shared_probe", "b2_label_probe"],
            "output_shape": [2, 6],
            "selected_probe_gradient_norm": float(norms[label_id]),
            "other_probe_gradient_norms": norms[torch.arange(6) != label_id].tolist(),
            "frozen_backbone_has_gradient": False,
            "joint_trainable_encoder_layers": trainable_layers,
            "checkpoint_roundtrip_exact": True,
            "mixed_precision_forward": True,
            "test_checked": False,
        }
        output_path = ROOT / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print("VulProbe smoke test: OK")


if __name__ == "__main__":
    main()
