"""Parameter-efficient execution-structure adaptation for EVM-BERT.

The base EVM-BERT remains frozen. A small structural encoder produces a token
aligned execution representation, a gated additive adapter injects it at the
embedding boundary, and LoRA is applied only to attention Q/V projections.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
from transformers import BertForMaskedLM
from transformers.models.bert.modeling_bert import BertSelfAttention


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=16.0, dropout=0.0):
        super().__init__()
        self.base = base
        self.rank = int(rank)
        self.scaling = float(alpha) / max(1, self.rank)
        self.dropout = nn.Dropout(float(dropout))
        self.lora_a = nn.Parameter(torch.empty(self.rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_a, a=5 ** 0.5)

    def forward(self, hidden_states):
        base = self.base(hidden_states)
        update = self.dropout(hidden_states).matmul(self.lora_a.t()).matmul(self.lora_b.t())
        return base + self.scaling * update


class StackStructuralEncoder(nn.Module):
    """Encode token-aligned stack state and typed relation neighborhoods."""

    def __init__(self, output_dim=768, hidden_dim=256, layers=1, heads=8, dropout=0.1):
        super().__init__()
        self.field_embeddings = nn.ModuleList([
            nn.Embedding(32, hidden_dim),
            nn.Embedding(32, hidden_dim),
            nn.Embedding(256, hidden_dim),
            nn.Embedding(16, hidden_dim),
            nn.Embedding(2, hidden_dim),
        ])
        self.relation_embedding = nn.Embedding(8, hidden_dim)
        self.source_relation_embedding = nn.Embedding(8, hidden_dim)
        self.target_relation_embedding = nn.Embedding(8, hidden_dim)
        self.slot_embedding = nn.Embedding(16, hidden_dim)
        self.distance_embedding = nn.Embedding(9, hidden_dim)
        self.boundary_embeddings = nn.ModuleList([nn.Embedding(16, hidden_dim) for _ in range(4)])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, stack_state, attention_mask, edge_offsets, edge_src, edge_dst,
                edge_type, edge_slot, edge_distance, edge_confidence,
                boundary_state=None):
        state = stack_state.long()
        hidden = sum(embedding(state[..., index]) for index, embedding in enumerate(self.field_embeddings))
        relation = hidden.new_zeros(hidden.shape)
        for chunk in range(hidden.shape[0]):
            left, right = int(edge_offsets[chunk]), int(edge_offsets[chunk + 1])
            if right <= left:
                continue
            src = edge_src[left:right].long().to(hidden.device)
            dst = edge_dst[left:right].long().to(hidden.device)
            typ = edge_type[left:right].long().to(hidden.device).clamp(0, 7)
            slot = edge_slot[left:right].long().to(hidden.device).clamp(0, 15)
            distance = edge_distance[left:right].long().to(hidden.device).clamp(0, 8)
            weight = edge_confidence[left:right].to(hidden.device, dtype=hidden.dtype).unsqueeze(-1)
            values = (self.relation_embedding(typ) + self.slot_embedding(slot) + self.distance_embedding(distance)) * weight
            relation[chunk].index_add_(0, src, values + self.source_relation_embedding(typ) * weight)
            relation[chunk].index_add_(0, dst, values + self.target_relation_embedding(typ) * weight)
        hidden = self.norm(hidden + relation)
        if boundary_state is not None:
            boundary = sum(
                embedding(boundary_state[:, index].long().clamp(0, 15))
                for index, embedding in enumerate(self.boundary_embeddings)
            )
            hidden[:, 0] = hidden[:, 0] + boundary
        padding = ~attention_mask.bool()
        hidden = self.encoder(self.dropout(hidden), src_key_padding_mask=padding)
        return self.output(self.norm(hidden))


class AdapterSelfAttention(BertSelfAttention):
    def enable_lora(self, rank=8, alpha=16.0, dropout=0.05):
        self.query = LoRALinear(self.query, rank=rank, alpha=alpha, dropout=dropout)
        self.value = LoRALinear(self.value, rank=rank, alpha=alpha, dropout=dropout)

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                past_key_value=None, output_attentions=False, **kwargs):
        if encoder_hidden_states is not None or past_key_value is not None:
            raise ValueError("AdapterSelfAttention supports encoder-only self attention")
        query = self.transpose_for_scores(self.query(hidden_states))
        key = self.transpose_for_scores(self.key(hidden_states))
        value = self.transpose_for_scores(self.value(hidden_states))
        scores = torch.matmul(query, key.transpose(-1, -2)) / (self.attention_head_size ** 0.5)
        if attention_mask is not None:
            scores = scores + attention_mask
        probs = torch.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        if head_mask is not None:
            probs = probs * head_mask
        context = torch.matmul(probs, value).permute(0, 2, 1, 3).contiguous()
        context = context.view(context.size()[:-2] + (self.all_head_size,))
        return (context, probs) if output_attentions else (context,)


def replace_attention_with_lora(base, rank=8, alpha=16.0, dropout=0.05):
    for layer in base.bert.encoder.layer:
        original = layer.attention.self
        replacement = AdapterSelfAttention(base.config)
        replacement.load_state_dict(original.state_dict(), strict=True)
        replacement.enable_lora(rank=rank, alpha=alpha, dropout=dropout)
        layer.attention.self = replacement


class StackAdapterBertForMaskedLM(nn.Module):
    def __init__(self, base_model, stack_hidden_dim=256, stack_layers=1,
                 lora_rank=8, lora_alpha=16.0, lora_dropout=0.05,
                 fusion_init=0.05):
        super().__init__()
        self.base = base_model
        replace_attention_with_lora(self.base, lora_rank, lora_alpha, lora_dropout)
        hidden = int(self.base.config.hidden_size)
        self.stack_encoder = StackStructuralEncoder(hidden, stack_hidden_dim, stack_layers)
        self.stack_projection = nn.Linear(hidden, hidden)
        self.fusion_gate_raw = nn.Parameter(torch.tensor(float(torch.logit(torch.tensor(fusion_init).clamp(1e-4, 1 - 1e-4)))))

    @classmethod
    def from_pretrained(cls, path, local_files_only=True, **kwargs):
        return cls(BertForMaskedLM.from_pretrained(path, local_files_only=local_files_only), **kwargs)

    @property
    def config(self):
        return self.base.config

    def trainable_parameters(self):
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def freeze_base(self):
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        for layer in self.base.bert.encoder.layer:
            for name in ("query", "value"):
                module = getattr(layer.attention.self, name)
                module.lora_a.requires_grad = True
                module.lora_b.requires_grad = True
        for parameter in self.stack_encoder.parameters():
            parameter.requires_grad = True
        for parameter in self.stack_projection.parameters():
            parameter.requires_grad = True
        self.fusion_gate_raw.requires_grad = True

    def _attention_mask(self, attention_mask, dtype):
        mask = attention_mask.bool()[:, None, None, :].to(dtype)
        return (1.0 - mask) * torch.finfo(dtype).min

    def forward(self, input_ids, attention_mask, stack_state, boundary_state=None,
                edge_offsets=None, edge_src=None, edge_dst=None, edge_type=None,
                edge_slot=None, edge_distance=None, edge_confidence=None,
                labels=None, output_attentions=False):
        input_ids = input_ids.long()
        attention_mask = attention_mask.bool()
        embeddings = self.base.bert.embeddings(input_ids=input_ids)
        structural = self.stack_encoder(
            stack_state, attention_mask, edge_offsets, edge_src, edge_dst,
            edge_type, edge_slot, edge_distance, edge_confidence,
            boundary_state=boundary_state,
        )
        gate = torch.sigmoid(self.fusion_gate_raw).to(embeddings.dtype)
        embeddings = embeddings + gate * self.stack_projection(structural).to(embeddings.dtype)
        extended_mask = self._attention_mask(attention_mask, embeddings.dtype)
        hidden = embeddings
        attentions = []
        for layer in self.base.bert.encoder.layer:
            output = layer(hidden, attention_mask=extended_mask, head_mask=None, output_attentions=output_attentions)
            hidden = output[0]
            if output_attentions:
                attentions.append(output[1])
        logits = self.base.cls(hidden) if labels is not None else None
        loss = None
        if labels is not None:
            valid = labels.reshape(-1).ne(-100)
            if valid.any():
                loss = nn.functional.cross_entropy(logits.float().reshape(-1, logits.size(-1))[valid], labels.reshape(-1)[valid])
            else:
                loss = hidden.float().sum() * 0.0
        return SimpleNamespace(loss=loss, logits=logits, last_hidden_state=hidden, attentions=tuple(attentions) if output_attentions else None)
