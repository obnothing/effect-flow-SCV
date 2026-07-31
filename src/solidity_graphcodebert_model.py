"""GraphCodeBERT with label-conditioned contract-level multi-slot MIL."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class SolidityGraphCodeBERTMultiSlotMIL(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.num_labels = int(config["num_labels"])
        self.hidden_dim = int(config.get("hidden_dim", 768))
        self.num_slots = int(config.get("num_slots", 3))
        self.use_dfg = bool(config.get("use_dfg", True))
        self.use_mil = bool(config.get("use_mil", True))
        self.encoder = AutoModel.from_pretrained(
            config["source_model_path"], local_files_only=True
        )
        if bool(config.get("gradient_checkpointing", False)):
            # Reentrant checkpointing reuses the same LoRA parameter across
            # window microbatches and is incompatible with PyTorch 2.0 DDP.
            try:
                self.encoder.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                self.encoder.gradient_checkpointing_enable()
            self.encoder.enable_input_require_grads()
            self.encoder.config.use_cache = False
        if self.encoder.config.hidden_size != self.hidden_dim:
            raise ValueError("GraphCodeBERT hidden size does not match hidden_dim")
        self.dropout = nn.Dropout(float(config.get("dropout", 0.1)))
        self.label_queries = nn.Parameter(torch.empty(self.num_labels, self.num_slots, self.hidden_dim))
        self.slot_logits = nn.Parameter(torch.zeros(self.num_labels, self.num_slots))
        self.label_heads = nn.Parameter(torch.empty(self.num_labels, self.hidden_dim))
        self.label_bias = nn.Parameter(torch.zeros(self.num_labels))
        self.mean_head = nn.Linear(self.hidden_dim, self.num_labels)
        self.projection = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.GELU(), nn.Linear(self.hidden_dim, 256))
        nn.init.normal_(self.label_queries, std=0.02)
        nn.init.normal_(self.label_heads, std=0.02)

    def apply_lora(self, config):
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:
            raise RuntimeError("LoRA requires peft==0.12.0; run scripts/install_source_main6_dependencies.sh") from exc
        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=int(config.get("lora_rank", 8)), lora_alpha=int(config.get("lora_alpha", 16)),
            lora_dropout=float(config.get("lora_dropout", 0.05)), bias="none",
            target_modules=list(config.get("lora_target_modules", ["query", "key", "value"])),
        )
        self.encoder = get_peft_model(self.encoder, lora_config)

    def encode_units(self, input_ids, token_mask, graph_mask, unit_mask):
        batch, units, length = input_ids.shape
        flat_active = unit_mask.reshape(-1)
        flat_ids = input_ids.reshape(batch * units, length)
        flat_tokens = token_mask.reshape(batch * units, length)
        flat_graph = graph_mask.reshape(batch * units, length, length) if self.use_dfg else None
        embeddings = input_ids.new_zeros((batch * units, self.hidden_dim), dtype=torch.float32)
        if flat_active.any():
            active_indices = flat_active.nonzero(as_tuple=False).flatten()
            microbatch = int(self.config.get("encoder_microbatch_size", 8))
            for start in range(0, active_indices.numel(), microbatch):
                index = active_indices[start:start + microbatch]
                attention_mask = flat_graph[index].long() if self.use_dfg else flat_tokens[index].long()
                output = self.encoder(input_ids=flat_ids[index], attention_mask=attention_mask, return_dict=True)
                embeddings[index] = output.last_hidden_state[:, 0].float()
        return embeddings.reshape(batch, units, self.hidden_dim)

    @staticmethod
    def masked_mean(values, mask):
        weight = mask.unsqueeze(-1).type_as(values)
        return (values * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)

    def forward(self, input_ids, token_mask, graph_mask, unit_mask, return_attention=False):
        unit_embeddings = self.dropout(self.encode_units(input_ids, token_mask, graph_mask, unit_mask))
        contract_embedding = self.masked_mean(unit_embeddings, unit_mask)
        if not self.use_mil:
            logits = self.mean_head(contract_embedding)
            return {"logits": logits, "contract_embedding": contract_embedding, "projection": F.normalize(self.projection(contract_embedding), dim=-1)}
        scores = torch.einsum("buh,lsh->buls", unit_embeddings, self.label_queries)
        scores = scores.masked_fill(~unit_mask[:, :, None, None], -1e9)
        attention = torch.softmax(scores, dim=1)
        slot_repr = torch.einsum("buls,buh->blsh", attention, unit_embeddings)
        slot_weights = torch.softmax(self.slot_logits, dim=-1)
        label_repr = torch.einsum("ls,blsh->blh", slot_weights, slot_repr)
        logits = torch.einsum("blh,lh->bl", label_repr, self.label_heads) + self.label_bias
        result = {"logits": logits, "contract_embedding": contract_embedding, "projection": F.normalize(self.projection(contract_embedding), dim=-1)}
        if return_attention:
            result["slot_unit_attention"] = attention
            result["slot_weights"] = slot_weights
        return result


def multilabel_supervised_contrastive(projection, labels, temperature=0.1):
    """Jaccard-weighted SupCon, with all-zero contracts as denominator-only samples."""
    if projection.shape[0] < 2:
        return projection.sum() * 0.0
    similarity = torch.matmul(projection, projection.T) / float(temperature)
    diagonal = torch.eye(similarity.shape[0], device=similarity.device, dtype=torch.bool)
    similarity = similarity.masked_fill(diagonal, -1e9)
    overlap = torch.matmul(labels, labels.T)
    union = labels.sum(dim=1, keepdim=True) + labels.sum(dim=1).unsqueeze(0) - overlap
    weights = torch.where(union > 0, overlap / union.clamp_min(1.0), torch.zeros_like(overlap))
    weights = weights.masked_fill(diagonal, 0.0)
    log_prob = similarity - torch.logsumexp(similarity, dim=1, keepdim=True)
    active = weights.sum(dim=1) > 0
    if not active.any():
        return projection.sum() * 0.0
    per_anchor = -(weights * log_prob).sum(dim=1) / weights.sum(dim=1).clamp_min(1e-8)
    return per_anchor[active].mean()


def symmetric_kl(first_logits, second_logits):
    first_prob, second_prob = torch.sigmoid(first_logits), torch.sigmoid(second_logits)
    eps = 1e-6
    def binary_kl(left, right):
        return left * torch.log((left + eps) / (right + eps)) + (1 - left) * torch.log((1 - left + eps) / (1 - right + eps))
    return 0.5 * (binary_kl(first_prob, second_prob).mean() + binary_kl(second_prob, first_prob).mean())
