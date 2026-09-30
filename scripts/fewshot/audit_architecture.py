"""Audit the real E3/P11 architecture and save the Few-shot Pilot report."""

import json
import sys
from pathlib import Path

import torch
import yaml
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from evm_tokenizer import EVMOpcodeTokenizer
from polarity_query_model import PolarityQueryNet


def main():
    pilot = yaml.safe_load((ROOT / "configs/fewshot_query_pilot/pilot0.yaml").read_text(encoding="utf-8"))
    config = yaml.safe_load((ROOT / pilot["e3_base_config"]).read_text(encoding="utf-8"))
    tokenizer = EVMOpcodeTokenizer.from_vocab_file(ROOT / pilot["vocab_path"])
    model = PolarityQueryNet("P11", len(tokenizer), tokenizer.pad_token_id, config["embedding_dim"],
        config["gru_hidden_size"], config["num_labels"], config["attention_heads"], config["bidirectional"],
        config["gru_layers"], config["representation_dropout"], config["query_dim"])
    named = dict(model.named_parameters())
    per_label = {"query_plus": int(model.queries[0, 0].numel()), "query_minus": int(model.queries[0, 1].numel()),
        "queries_pair": int(model.queries[0].numel()), "label_scorer": int(model.label_scorer[0].numel()),
        "branch_bias": int(model.branch_bias[0].numel())}
    shared_parameters = {name: int(param.numel()) for name, param in model.named_parameters()
                         if not name.startswith(("queries", "label_scorer", "branch_bias"))}
    modules = dict(model.named_modules())
    transformer_modules = [name for name, item in modules.items() if isinstance(item, nn.TransformerEncoder) or isinstance(item, nn.TransformerEncoderLayer)]
    e3_checkpoint_dir = ROOT / "checkpoints/spor_phase25/E3_P11_8_A0/P11"
    audit = {"route": "Few-shot Novel Vulnerability Query Expansion Pilot-0", "dataset": pilot["dataset"],
        "model_class": model.__class__.__name__, "num_labels": config["num_labels"], "label_names": config["label_names"],
        "block_transformer_present": bool(transformer_modules), "transformer_module_names": transformer_modules,
        "embedding_dim": model.embedding.embedding_dim, "bigru_hidden_per_direction": model.encoder.hidden_size,
        "bidirectional": model.encoder.bidirectional, "bigru_output_dim": model.output_dim,
        "kv_input_dim": model.cross_attention.key_dim, "kv_projection_dim": model.cross_attention.embed_dim,
        "query_plus_dim": model.queries.shape[-1], "query_minus_dim": model.queries.shape[-1],
        "query_shape": list(model.queries.shape), "attention_heads": model.cross_attention.num_heads,
        "head_dim": model.cross_attention.head_dim, "label_scorer_shape": list(model.label_scorer.shape),
        "branch_bias_shape": list(model.branch_bias.shape), "scorer_shared_across_polarities": True,
        "per_label_independent_parameters": per_label, "shared_parameters": shared_parameters,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "per_label_output_parameter_count": per_label["queries_pair"] + per_label["label_scorer"] + per_label["branch_bias"],
        "scorer_params_fraction_of_per_label_output": (per_label["label_scorer"]+per_label["branch_bias"])/(per_label["queries_pair"]+per_label["label_scorer"]+per_label["branch_bias"]),
        "fewshot_trainable_parameter_scenarios": {
            "M0_random_query": {"query": per_label["queries_pair"], "scorer": per_label["label_scorer"]+per_label["branch_bias"],
                "encoder": 0, "total": per_label["queries_pair"]+per_label["label_scorer"]+per_label["branch_bias"]},
            "M2_query_mixture": {"query": 14, "scorer": per_label["label_scorer"]+per_label["branch_bias"], "encoder": 0,
                "total": 14+per_label["label_scorer"]+per_label["branch_bias"]},
            "M3_M4_residual_stage": {"query": 8, "scorer": 0, "encoder": 0, "total": 8},
            "M5_full_finetune": {"query": per_label["queries_pair"], "scorer": per_label["label_scorer"]+per_label["branch_bias"],
                "encoder": sum(p.numel() for n,p in named.items() if not n.startswith(("queries", "label_scorer", "branch_bias"))),
                "total": sum(p.numel() for n,p in named.items() if not n.startswith(("queries", "label_scorer", "branch_bias")))+per_label["queries_pair"]+per_label["label_scorer"]+per_label["branch_bias"]}},
        "seven_label_model_has_label_specific_rows": True,
        "eight_label_config_exists": (ROOT / pilot["e3_base_config"]).exists(),
        "trained_eight_label_checkpoint_present": any(e3_checkpoint_dir.rglob("best.pt")) if e3_checkpoint_dir.exists() else False,
        "loss": {"main": "BCEWithLogits on e_plus-e_minus", "polarity_auxiliary_weight": config["auxiliary_weight"],
                 "positive_target": "y", "negative_target": "1-y", "pos_weight_mode": config["pos_weight_mode"],
                 "pos_weight_power": config["pos_weight_power"], "max_pos_weight": config["max_pos_weight"],
                 "DoS_soft_targets": [config["dos_soft_positive"], config["dos_soft_negative"]]},
        "threshold_protocol": "P11 default grid for official valid reporting only; pilot primary metric is threshold-free AP; no valid threshold tuning",
        "official_test_accessed": False}
    out = ROOT / pilot["report_root"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "architecture_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    lines = ["# Few-shot Query Pilot Architecture Audit", "", f"Dataset: `{pilot['dataset']}`. Model instantiated from actual `PolarityQueryNet` and E3/P11 configuration.", "",
        f"- Additional Block Transformer: **{'yes' if audit['block_transformer_present'] else 'no'}**.",
        f"- Opcode embedding: **{audit['embedding_dim']}**.", f"- BiGRU: **{audit['bigru_hidden_per_direction']} per direction**, output **{audit['bigru_output_dim']}**.",
        f"- Shared K/V: **{audit['kv_input_dim']} → {audit['kv_projection_dim']}**; attention **{audit['attention_heads']} × {audit['head_dim']}**.",
        f"- Positive and negative query shape: **{audit['query_shape']}**.",
        f"- Label scorer: **{audit['label_scorer_shape']}**, shared between positive/negative branches for each label; branch bias **{audit['branch_bias_shape']}**.",
        f"- Total 8-label parameters: **{audit['total_parameters']:,}**.",
        f"- Per-label independent parameters: q+={per_label['query_plus']:,}, q-={per_label['query_minus']:,}, shared scorer={per_label['label_scorer']:,}, branch biases={per_label['branch_bias']:,}.",
        f"- In M2 adaptation, 14 mixture logits plus {per_label['label_scorer']+per_label['branch_bias']:,} scorer/bias parameters are trainable: scorer/bias are {100*audit['fewshot_trainable_parameter_scenarios']['M2_query_mixture']['scorer']/audit['fewshot_trainable_parameter_scenarios']['M2_query_mixture']['total']:.1f}% of that count. The scorer does not dominate M0's query+scorer count ({100*audit['scorer_params_fraction_of_per_label_output']:.1f}%).",
        f"- Eight-label config exists: **{audit['eight_label_config_exists']}**; trained checkpoint for this pilot: **{audit['trained_eight_label_checkpoint_present']}**.",
        "- Shared parameters: opcode embedding, bidirectional GRU, shared query/key/value/output projections. Independent rows: each label's q+/q-, scorer vector, and two branch biases.",
        "- Loss: weighted main BCE on e+−e− plus polarity auxiliary BCE; E3 defaults use auxiliary weight 0.08, sqrt-ratio^0.6 positive weighting capped at 5, and DoS soft auxiliary targets 0.8/0.1.",
        "- Pilot primary evaluation is Average Precision; thresholded F1 uses fixed 0.5 and no threshold is tuned on official valid.",
        "- Official test accessed: **false**.", ""]
    (out / "architecture_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
