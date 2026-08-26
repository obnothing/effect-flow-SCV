"""BERT with direct EVM stack-state embeddings and sparse relation bias."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from transformers import BertForMaskedLM
from transformers.models.bert.modeling_bert import BertSelfAttention


class StackAwareSelfAttention(BertSelfAttention):
    def __init__(self, config):
        super().__init__(config)
        self.relation_scale = nn.Parameter(torch.zeros(()))
        self._relation_bias = None

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        **kwargs,
    ):
        if encoder_hidden_states is not None or past_key_value is not None:
            raise ValueError("StackAwareSelfAttention supports encoder-only self attention")
        mixed_query_layer = self.query(hidden_states)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / (self.attention_head_size ** 0.5)
        if self._relation_bias is not None:
            scale = torch.tanh(self.relation_scale).to(attention_scores.dtype)
            attention_scores = attention_scores + scale * self._relation_bias.to(attention_scores.dtype)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_shape)
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


def _replace_attention_modules(base):
    for layer in base.bert.encoder.layer:
        original = layer.attention.self
        replacement = StackAwareSelfAttention(base.config)
        replacement.load_state_dict(original.state_dict(), strict=False)
        layer.attention.self = replacement


class StackAwareBertForMaskedLM(nn.Module):
    """Wrap a frozen-compatible HF MLM and inject stack relations before layers."""

    def __init__(self, base_model, num_relation_types=8, num_slots=16, num_distance_buckets=9):
        super().__init__()
        self.base = base_model
        # The downstream route freezes the base BERT weights but still needs
        # gradients through BERT to train the injected stack parameters. A
        # checkpointed layer recomputes activations during backward instead of
        # retaining every 512-token attention tensor for every chunk.
        self.gradient_checkpointing = True
        _replace_attention_modules(self.base)
        hidden = int(self.base.config.hidden_size)
        heads = int(self.base.config.num_attention_heads)
        self.stack_embeddings = nn.ModuleList([
            nn.Embedding(32, hidden),
            nn.Embedding(32, hidden),
            nn.Embedding(256, hidden),
            nn.Embedding(16, hidden),
            nn.Embedding(2, hidden),
        ])
        self.boundary_embeddings = nn.ModuleList([nn.Embedding(16, hidden) for _ in range(4)])
        self.relation_embedding = nn.Embedding(num_relation_types, heads)
        self.slot_embedding = nn.Embedding(num_slots, heads)
        self.distance_embedding = nn.Embedding(num_distance_buckets, heads)
        for module in list(self.stack_embeddings) + list(self.boundary_embeddings):
            nn.init.zeros_(module.weight)
        nn.init.zeros_(self.relation_embedding.weight)
        nn.init.zeros_(self.slot_embedding.weight)
        nn.init.zeros_(self.distance_embedding.weight)
        self.relation_embedding.weight.data[1:].normal_(std=0.01)

    @classmethod
    def from_pretrained(cls, path, local_files_only=True):
        base = BertForMaskedLM.from_pretrained(path, local_files_only=local_files_only)
        return cls(base)

    @property
    def config(self):
        return self.base.config

    def _embedding_delta(self, stack_state, boundary_state):
        delta = 0.0
        for index, embedding in enumerate(self.stack_embeddings):
            delta = delta + embedding(stack_state[..., index].clamp_min(0).clamp_max(embedding.num_embeddings - 1))
        if boundary_state is not None:
            boundary = 0.0
            for index, embedding in enumerate(self.boundary_embeddings):
                boundary = boundary + embedding(boundary_state[..., index].clamp_min(0).clamp_max(15))
            delta = delta.clone()
            delta[:, 0, :] = delta[:, 0, :] + boundary
        return delta

    def _relation_bias(self, edge_offsets, edge_src, edge_dst, edge_type, edge_slot, edge_distance, edge_confidence, chunks, length, device, dtype):
        heads = int(self.base.config.num_attention_heads)
        bias = torch.zeros((chunks, heads, length, length), device=device, dtype=dtype)
        if edge_src.numel() == 0:
            return bias
        edge_offsets = edge_offsets.to(device=device)
        for chunk in range(chunks):
            left = int(edge_offsets[chunk])
            right = int(edge_offsets[chunk + 1])
            if right <= left:
                continue
            values = self.relation_embedding(edge_type[left:right].to(device))
            values = values + self.slot_embedding(edge_slot[left:right].to(device))
            values = values + self.distance_embedding(edge_distance[left:right].to(device))
            values = values * edge_confidence[left:right].to(device=device, dtype=values.dtype).unsqueeze(-1)
            src = edge_src[left:right].long().to(device)
            dst = edge_dst[left:right].long().to(device)
            for head in range(heads):
                bias[chunk, head].index_put_((src, dst), values[:, head], accumulate=True)
        return bias

    def forward(
        self,
        input_ids,
        attention_mask,
        stack_state,
        boundary_state=None,
        edge_offsets=None,
        edge_src=None,
        edge_dst=None,
        edge_type=None,
        edge_slot=None,
        edge_distance=None,
        edge_confidence=None,
        labels=None,
        output_attentions=False,
    ):
        input_ids = input_ids.long()
        attention_mask = attention_mask.bool()
        inputs = self.base.bert.embeddings(input_ids=input_ids)
        inputs = inputs + self._embedding_delta(stack_state.long(), boundary_state.long() if boundary_state is not None else None).to(inputs.dtype)
        extended_mask = attention_mask[:, None, None, :].to(inputs.dtype)
        extended_mask = (1.0 - extended_mask) * torch.finfo(inputs.dtype).min
        relation_bias = None
        if edge_offsets is not None:
            relation_bias = self._relation_bias(edge_offsets, edge_src, edge_dst, edge_type, edge_slot, edge_distance, edge_confidence, input_ids.shape[0], input_ids.shape[1], input_ids.device, inputs.dtype)
        hidden = inputs
        attentions = []
        for layer_index, layer in enumerate(self.base.bert.encoder.layer):
            layer.attention.self._relation_bias = relation_bias if layer_index >= max(0, len(self.base.bert.encoder.layer) - 4) else None
            if self.gradient_checkpointing and self.training and not output_attentions:
                def run_layer(layer_hidden, current_layer=layer):
                    return current_layer(
                        layer_hidden,
                        attention_mask=extended_mask,
                        head_mask=None,
                        output_attentions=False,
                    )[0]

                hidden = checkpoint(run_layer, hidden)
                outputs = (hidden,)
            else:
                outputs = layer(hidden, attention_mask=extended_mask, head_mask=None, output_attentions=output_attentions)
                hidden = outputs[0]
            if output_attentions:
                attentions.append(outputs[1])
        token_hidden = outputs[0]
        # Downstream MIL only consumes last_hidden_state. Avoid materializing
        # the vocabulary projection unless MLM labels are actually supplied.
        logits = self.base.cls(token_hidden) if labels is not None else None
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        return SimpleNamespace(loss=loss, logits=logits, last_hidden_state=token_hidden, attentions=tuple(attentions) if output_attentions else None)
