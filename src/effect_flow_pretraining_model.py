import json
from pathlib import Path

import torch
from torch import nn
from transformers import BertForMaskedLM


class EffectFlowBertForPreTraining(nn.Module):
    """Existing EVM-BERT with MOM, token-level ETP, and chunk-level EFPP heads."""

    def __init__(
        self,
        base_hf_model_path,
        vocab_size,
        num_effect_types=16,
        num_efpp_patterns=22,
        lambda_mom=1.0,
        lambda_etp=0.3,
        lambda_efpp=0.5,
        etp_class_weights=None,
        efpp_pos_weights=None,
        local_files_only=True,
    ):
        super().__init__()
        path = Path(base_hf_model_path)
        if not path.exists():
            raise FileNotFoundError(f"Base EVM-BERT path does not exist: {path}")
        try:
            base_model = BertForMaskedLM.from_pretrained(
                str(path), local_files_only=local_files_only
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load existing EVM-BERT from {path}: {exc}") from exc
        if int(base_model.config.vocab_size) != int(vocab_size):
            raise ValueError(
                f"Model/vocab mismatch: model={base_model.config.vocab_size}, "
                f"tokenizer={vocab_size}"
            )
        self.config = base_model.config
        self.bert = base_model.bert
        self.mom_head = base_model.cls
        self.num_effect_types = int(num_effect_types)
        self.num_efpp_patterns = int(num_efpp_patterns)
        self.etp_head = nn.Linear(self.config.hidden_size, self.num_effect_types)
        self.efpp_dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.efpp_head = nn.Linear(self.config.hidden_size, self.num_efpp_patterns)
        self.lambda_mom = float(lambda_mom)
        self.lambda_etp = float(lambda_etp)
        self.lambda_efpp = float(lambda_efpp)
        self.register_buffer(
            "etp_class_weights",
            torch.ones(self.num_effect_types, dtype=torch.float32)
            if etp_class_weights is None
            else torch.as_tensor(etp_class_weights, dtype=torch.float32),
        )
        self.register_buffer(
            "efpp_pos_weights",
            torch.ones(self.num_efpp_patterns, dtype=torch.float32)
            if efpp_pos_weights is None
            else torch.as_tensor(efpp_pos_weights, dtype=torch.float32),
        )
        if self.etp_class_weights.numel() != self.num_effect_types:
            raise ValueError("ETP class-weight width does not match num_effect_types.")
        if self.efpp_pos_weights.numel() != self.num_efpp_patterns:
            raise ValueError("EFPP pos-weight width does not match num_efpp_patterns.")

    def gradient_checkpointing_enable(self):
        self.bert.gradient_checkpointing_enable()

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
        pooling_mask=None,
        mom_labels=None,
        etp_labels=None,
        efpp_labels=None,
        source_dataset_id=None,
    ):
        del source_dataset_id
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        mom_logits = self.mom_head(hidden)
        etp_logits = self.etp_head(hidden)
        if pooling_mask is None:
            pooling_mask = attention_mask.bool()
        pooling_mask = pooling_mask.bool()
        pool_weights = pooling_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * pool_weights).sum(dim=1) / pool_weights.sum(dim=1).clamp_min(1.0)
        efpp_logits = self.efpp_head(self.efpp_dropout(pooled))

        mom_loss = None
        etp_loss = None
        efpp_loss = None
        total_loss = None
        if mom_labels is not None:
            mom_loss = nn.functional.cross_entropy(
                mom_logits.reshape(-1, self.config.vocab_size),
                mom_labels.reshape(-1),
                ignore_index=-100,
            )
        if etp_labels is not None:
            etp_loss = nn.functional.cross_entropy(
                etp_logits.reshape(-1, self.num_effect_types),
                etp_labels.reshape(-1),
                weight=self.etp_class_weights,
                ignore_index=-100,
            )
        if efpp_labels is not None:
            efpp_loss = nn.functional.binary_cross_entropy_with_logits(
                efpp_logits,
                efpp_labels.float(),
                pos_weight=self.efpp_pos_weights,
            )
        losses = [mom_loss, etp_loss, efpp_loss]
        if all(loss is not None for loss in losses):
            total_loss = (
                self.lambda_mom * mom_loss
                + self.lambda_etp * etp_loss
                + self.lambda_efpp * efpp_loss
            )
        return {
            "loss": total_loss,
            "mom_loss": mom_loss,
            "etp_loss": etp_loss,
            "efpp_loss": efpp_loss,
            "mom_logits": mom_logits,
            "etp_logits": etp_logits,
            "efpp_logits": efpp_logits,
        }

    def save_hf_model(self, output_dir, metadata=None):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        export_model = BertForMaskedLM(self.config)
        export_model.bert = self.bert
        export_model.cls = self.mom_head
        export_model.tie_weights()
        export_model.save_pretrained(output_dir)
        torch.save(
            {
                "etp_head_state_dict": self.etp_head.state_dict(),
                "efpp_head_state_dict": self.efpp_head.state_dict(),
                "etp_class_weights": self.etp_class_weights.detach().cpu(),
                "efpp_pos_weights": self.efpp_pos_weights.detach().cpu(),
                "num_effect_types": self.num_effect_types,
                "num_efpp_patterns": self.num_efpp_patterns,
                "lambda_mom": self.lambda_mom,
                "lambda_etp": self.lambda_etp,
                "lambda_efpp": self.lambda_efpp,
            },
            output_dir / "effect_flow_heads.pt",
        )
        payload = {
            "model_type": "effect-flow-aware EVM-BERT",
            "hf_encoder_and_mom_saved": True,
            "effect_flow_heads_file": "effect_flow_heads.pt",
            **(metadata or {}),
        }
        (output_dir / "effect_flow_pretraining_config.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
