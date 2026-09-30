"""Measure safe E3/P11 batch size on the requested local CUDA environment."""

import json
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from evm_tokenizer import EVMOpcodeTokenizer
from polarity_query_model import build_model, loss_terms


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("learnDL310 CUDA is required for the local pilot")
    pilot = yaml.safe_load((ROOT / "configs/fewshot_query_pilot/pilot0.yaml").read_text(encoding="utf-8"))
    config = yaml.safe_load((ROOT / pilot["e3_base_config"]).read_text(encoding="utf-8"))
    held = "Reentrancy"
    index = config["label_names"].index(held)
    config["label_names"] = [name for name in config["label_names"] if name != held]
    config["num_labels"] = 7
    config["positive_auxiliary_label_multiplier"] = [1.0] * 7
    config["negative_auxiliary_label_multiplier"] = [1.0] * 7
    tokenizer = EVMOpcodeTokenizer.from_vocab_file(ROOT / pilot["vocab_path"])
    torch.manual_seed(42)
    model = build_model("P11", config, len(tokenizer), tokenizer.pad_token_id).to("cuda").train()
    rows = []
    for batch_size in (1, 2, 4, 8, 16, 32):
        try:
            model.zero_grad(set_to_none=True)
            ids = torch.randint(5, len(tokenizer), (batch_size, 8192), device="cuda")
            lengths = torch.full((batch_size,), 8192, dtype=torch.long)
            mask = torch.ones_like(ids, dtype=torch.bool)
            labels = torch.randint(0, 2, (batch_size, 7), device="cuda").float()
            optimizer = torch.optim.AdamW(model.parameters(), lr=pilot["base_learning_rate"], weight_decay=pilot["base_weight_decay"])
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); start = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.float16):
                output = model(ids, lengths, mask)
                loss = loss_terms(output, labels, torch.ones(7, device="cuda"), config["auxiliary_weight"],
                    positive_label_multiplier=[1.0]*7, negative_label_multiplier=[1.0]*7,
                    dos_soft_targets={"positive_high": .8, "positive_low": .1, "negative_low": .1, "negative_high": .8},
                    dos_label_index=config["label_names"].index("DoS"))[0]
            loss.backward(); optimizer.step(); torch.cuda.synchronize()
            rows.append({"batch_size": batch_size, "status": "ok", "seconds": time.perf_counter()-start,
                "peak_memory_mib": torch.cuda.max_memory_allocated()/2**20, "loss": float(loss.detach())})
            del ids, lengths, mask, labels, optimizer, output, loss
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            rows.append({"batch_size": batch_size, "status": "oom"})
            torch.cuda.empty_cache()
    passed = [row for row in rows if row["status"] == "ok"]
    if not passed:
        raise RuntimeError(f"No 8192-token batch fits: {rows}")
    selected = min(passed, key=lambda item: item["seconds"] / item["batch_size"])
    result = {"gpu": torch.cuda.get_device_name(0), "vram_mib": torch.cuda.get_device_properties(0).total_memory/2**20,
        "torch": torch.__version__, "cuda": torch.version.cuda, "max_len": 8192, "trials": rows,
        "selected_physical_batch": selected["batch_size"], "effective_batch_target": 64,
        "gradient_accumulation": max(1, 64 // selected["batch_size"]), "test_checked": False}
    output = ROOT / "reports/fewshot_query_pilot/cuda_preflight.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
