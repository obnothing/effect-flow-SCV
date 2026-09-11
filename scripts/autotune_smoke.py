"""Run a small real-cache smoke test for the A0-A3 AutoTune family."""

import argparse
import sys
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autotune_model import AutoTuneSequenceNet  # noqa: E402
from evm_tokenizer import EVMOpcodeTokenizer  # noqa: E402
from light_label_data import LightLabelDataset, collate_light_label  # noqa: E402


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-len", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / "configs/light_label/b2_label_attention.yaml").read_text(encoding="utf-8"))
    tokenizer = EVMOpcodeTokenizer.from_vocab_file(resolve(config["vocab_path"]))
    dataset = LightLabelDataset(resolve(config["cache_dir"]) / f"train_max{args.max_len}.pt", runtime_max_len=args.max_len, indices=list(range(min(16, 16474))))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                        collate_fn=partial(collate_light_label, pad_id=tokenizer.pad_token_id))
    batch = next(iter(loader)); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    common = {"embedding_dim": 128, "encoder": "BiGRU", "hidden": 128, "layers": 1, "embedding_mlp": False,
              "self_attention": False, "heads": 1, "residual": True, "dropout": 0.1, "num_labels": 6}
    for variant in ("a0_mean", "a1_shared", "a2_b2", "a3_vsfs"):
        model_config = {**common, "variant": variant}
        model = AutoTuneSequenceNet(model_config, len(tokenizer), tokenizer.pad_token_id).to(device)
        output = model(batch["input_ids"].to(device), batch["lengths"], batch["mask"].to(device))
        loss = F.binary_cross_entropy_with_logits(output["logits"], batch["labels"].to(device))
        loss.backward()
        assert output["logits"].shape == (len(batch["ids"]), 6)
        assert torch.isfinite(loss) and all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)
        print(f"[{variant}] logits={tuple(output['logits'].shape)} loss={float(loss.detach()):.6f} params={sum(p.numel() for p in model.parameters())}", flush=True)
    print({"device": str(device), "max_len": args.max_len, "test_checked": False, "status": "ok"})


if __name__ == "__main__": main()
