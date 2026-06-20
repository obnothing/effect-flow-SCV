import json
from pathlib import Path

from transformers import BertConfig, BertForMaskedLM


def load_evm_vocab_size(vocab_path):
    data = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
    token_to_id = data.get("token_to_id")
    if not token_to_id:
        raise ValueError(f"Invalid EVM vocab file: {vocab_path}")
    return len(token_to_id)


def build_bert_config(config, vocab_size):
    bert_config = dict(config.get("bert_config", {}))
    bert_config["vocab_size"] = int(vocab_size)
    return BertConfig(
        vocab_size=bert_config["vocab_size"],
        hidden_size=int(bert_config.get("hidden_size", 768)),
        num_hidden_layers=int(bert_config.get("num_hidden_layers", 12)),
        num_attention_heads=int(bert_config.get("num_attention_heads", 12)),
        intermediate_size=int(bert_config.get("intermediate_size", 3072)),
        max_position_embeddings=int(bert_config.get("max_position_embeddings", 512)),
        type_vocab_size=int(bert_config.get("type_vocab_size", 2)),
        hidden_dropout_prob=float(bert_config.get("hidden_dropout_prob", 0.1)),
        attention_probs_dropout_prob=float(
            bert_config.get("attention_probs_dropout_prob", 0.1)
        ),
        hidden_act=bert_config.get("hidden_act", "gelu"),
        initializer_range=float(bert_config.get("initializer_range", 0.02)),
    )


def build_evm_bert_mlm_model(config, vocab_size):
    base_hf_model_path = config.get("base_hf_model_path")
    if base_hf_model_path:
        model = BertForMaskedLM.from_pretrained(
            base_hf_model_path,
            local_files_only=True,
        )
        if int(model.config.vocab_size) != int(vocab_size):
            raise ValueError(
                "Continued MLM model/vocab mismatch: "
                f"model vocab_size={model.config.vocab_size}, EVM vocab_size={vocab_size}."
            )
        return model
    bert_config = build_bert_config(config, vocab_size)
    return BertForMaskedLM(bert_config)
