import json
from pathlib import Path

import torch
from torch import nn
from transformers import BertForMaskedLM


class EffectFlowBertForPreTraining(nn.Module):
    """EVEF-MVD pretraining model with shared behavior + vulnerability heads."""

    def __init__(
        self,
        base_hf_model_path,
        vocab_size,
        num_effect_types=16,
        num_efpp_patterns=22,
        num_relation_types=6,
        num_vulnerability_templates=18,
        lambda_mom=1.0,
        lambda_etp=0.3,
        lambda_efpp=0.5,
        lambda_err=0.3,
        lambda_vep=0.4,
        lambda_vtm=0.4,
        etp_class_weights=None,
        efpp_pos_weights=None,
        relation_class_weights=None,
        vep_pos_weights=None,
        vtm_pos_weights=None,
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
        hidden_size = int(self.config.hidden_size)
        dropout = float(self.config.hidden_dropout_prob)

        self.num_effect_types = int(num_effect_types)
        self.num_efpp_patterns = int(num_efpp_patterns)
        self.num_relation_types = int(num_relation_types)
        self.num_vulnerability_templates = int(num_vulnerability_templates)

        self.etp_head = nn.Linear(hidden_size, self.num_effect_types)
        self.efpp_dropout = nn.Dropout(dropout)
        self.efpp_head = nn.Linear(hidden_size, self.num_efpp_patterns)
        self.err_head = nn.Linear(hidden_size, self.num_relation_types)
        self.vep_head = nn.Linear(hidden_size, self.num_vulnerability_templates)
        self.vtm_head = nn.Linear(hidden_size, self.num_vulnerability_templates)

        self.lambda_mom = float(lambda_mom)
        self.lambda_etp = float(lambda_etp)
        self.lambda_efpp = float(lambda_efpp)
        self.lambda_err = float(lambda_err)
        self.lambda_vep = float(lambda_vep)
        self.lambda_vtm = float(lambda_vtm)

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
        self.register_buffer(
            "relation_class_weights",
            torch.ones(self.num_relation_types, dtype=torch.float32)
            if relation_class_weights is None
            else torch.as_tensor(relation_class_weights, dtype=torch.float32),
        )
        self.register_buffer(
            "vep_pos_weights",
            torch.ones(self.num_vulnerability_templates, dtype=torch.float32)
            if vep_pos_weights is None
            else torch.as_tensor(vep_pos_weights, dtype=torch.float32),
        )
        self.register_buffer(
            "vtm_pos_weights",
            torch.ones(self.num_vulnerability_templates, dtype=torch.float32)
            if vtm_pos_weights is None
            else torch.as_tensor(vtm_pos_weights, dtype=torch.float32),
        )
        self._validate_weight_shapes()

    def _validate_weight_shapes(self):
        if self.etp_class_weights.numel() != self.num_effect_types:
            raise ValueError("ETP class-weight width does not match num_effect_types.")
        if self.efpp_pos_weights.numel() != self.num_efpp_patterns:
            raise ValueError("EFPP pos-weight width does not match num_efpp_patterns.")
        if self.relation_class_weights.numel() != self.num_relation_types:
            raise ValueError("ERR class-weight width does not match num_relation_types.")
        if self.vep_pos_weights.numel() != self.num_vulnerability_templates:
            raise ValueError("VEP pos-weight width does not match num_vulnerability_templates.")
        if self.vtm_pos_weights.numel() != self.num_vulnerability_templates:
            raise ValueError("VTM pos-weight width does not match num_vulnerability_templates.")

    def gradient_checkpointing_enable(self):
        self.bert.gradient_checkpointing_enable()

    @staticmethod
    def _masked_mean(hidden, pooling_mask):
        weights = pooling_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    @staticmethod
    def _masked_bce(logits, targets, mask, pos_weight):
        mask = mask.float()
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets.float(),
            pos_weight=pos_weight,
            reduction="none",
        )
        loss = loss * mask
        denom = mask.sum().clamp_min(1.0)
        return loss.sum() / denom

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
        pooling_mask=None,
        mom_labels=None,
        etp_labels=None,
        efpp_labels=None,
        err_labels=None,
        relation_labels=None,
        vep_labels=None,
        vtm_labels=None,
        vulnerability_loss_mask=None,
        source_dataset_id=None,
    ):
        del relation_labels
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
        pooled = self._masked_mean(hidden, pooling_mask)
        dropped = self.efpp_dropout(pooled)
        efpp_logits = self.efpp_head(dropped)
        err_logits = self.err_head(dropped)
        vep_logits = self.vep_head(dropped)
        vtm_logits = self.vtm_head(dropped)

        mom_loss = None
        etp_loss = None
        efpp_loss = None
        err_loss = None
        vep_loss = None
        vtm_loss = None
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
        if err_labels is not None:
            err_labels_flat = err_labels.reshape(-1)
            err_logits_flat = err_logits.reshape(-1, self.num_relation_types)
            # Check if there are any valid labels (not -100)
            valid_mask = err_labels_flat != -100
            if valid_mask.any():
                err_loss = nn.functional.cross_entropy(
                    err_logits_flat[valid_mask],
                    err_labels_flat[valid_mask],
                    weight=self.relation_class_weights,
                )
            else:
                # Keep ERR head parameters in the autograd graph for DDP batches
                # without an annotated relation.
                err_loss = err_logits.sum() * 0.0
        if vep_labels is not None:
            if vulnerability_loss_mask is None:
                vulnerability_loss_mask = torch.ones_like(vep_labels)
            vep_loss = self._masked_bce(
                vep_logits,
                vep_labels,
                vulnerability_loss_mask,
                self.vep_pos_weights,
            )
        if vtm_labels is not None:
            if vulnerability_loss_mask is None:
                vulnerability_loss_mask = torch.ones_like(vtm_labels)
            vtm_loss = self._masked_bce(
                vtm_logits,
                vtm_labels,
                vulnerability_loss_mask,
                self.vtm_pos_weights,
            )

        losses = [mom_loss, etp_loss, efpp_loss, err_loss, vep_loss, vtm_loss]
        if all(loss is not None for loss in losses):
            total_loss = (
                self.lambda_mom * mom_loss
                + self.lambda_etp * etp_loss
                + self.lambda_efpp * efpp_loss
                + self.lambda_err * err_loss
                + self.lambda_vep * vep_loss
                + self.lambda_vtm * vtm_loss
            )

        return {
            "loss": total_loss,
            "mom_loss": mom_loss,
            "etp_loss": etp_loss,
            "efpp_loss": efpp_loss,
            "err_loss": err_loss,
            "vep_loss": vep_loss,
            "vtm_loss": vtm_loss,
            "mom_logits": mom_logits,
            "etp_logits": etp_logits,
            "efpp_logits": efpp_logits,
            "err_logits": err_logits,
            "vep_logits": vep_logits,
            "vtm_logits": vtm_logits,
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
                "err_head_state_dict": self.err_head.state_dict(),
                "vep_head_state_dict": self.vep_head.state_dict(),
                "vtm_head_state_dict": self.vtm_head.state_dict(),
                "etp_class_weights": self.etp_class_weights.detach().cpu(),
                "efpp_pos_weights": self.efpp_pos_weights.detach().cpu(),
                "relation_class_weights": self.relation_class_weights.detach().cpu(),
                "vep_pos_weights": self.vep_pos_weights.detach().cpu(),
                "vtm_pos_weights": self.vtm_pos_weights.detach().cpu(),
                "num_effect_types": self.num_effect_types,
                "num_efpp_patterns": self.num_efpp_patterns,
                "num_relation_types": self.num_relation_types,
                "num_vulnerability_templates": self.num_vulnerability_templates,
                "lambda_mom": self.lambda_mom,
                "lambda_etp": self.lambda_etp,
                "lambda_efpp": self.lambda_efpp,
                "lambda_err": self.lambda_err,
                "lambda_vep": self.lambda_vep,
                "lambda_vtm": self.lambda_vtm,
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
