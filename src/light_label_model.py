"""Minimal BiGRU models for testing label-specific opcode aggregation."""

import math

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def validate_model_config(model, config):
    """Fail before training if a requested architecture was not constructed."""
    checks = {
        "embedding_dim": (int(model.embedding.embedding_dim), int(config["embedding_dim"])),
        "gru_hidden_size": (int(model.encoder.hidden_size), int(config["gru_hidden_size"])),
        "gru_layers": (int(model.encoder.num_layers), int(config.get("gru_layers", 1))),
        "num_labels": (int(model.num_labels), int(config["num_labels"])),
        "bidirectional": (bool(model.encoder.bidirectional), bool(config["bidirectional"])),
    }
    mismatches = {key: values for key, values in checks.items() if values[0] != values[1]}
    if mismatches:
        raise RuntimeError(f"effective model/config mismatch: {mismatches}")


class LabelGuidedOpcodeNet(nn.Module):
    """Shared opcode encoder with mean, shared-attention, or label-attention pooling.

    For B2, u_(l,i)=q_l^T W_a h_i, alpha_(l,i)=softmax_i(u_(l,i)),
    z_l=sum_i alpha_(l,i)h_i, and s_l=w_l^T z_l+b_l.
    """

    def __init__(self, variant, vocab_size, pad_id, embedding_dim=128,
                 gru_hidden_size=128, num_labels=6, bidirectional=True, local_radius=8,
                 gru_layers=1):
        super().__init__()
        self.variant = str(variant)
        self.num_labels = int(num_labels)
        self.local_radius = int(local_radius)
        self.gru_layers = int(gru_layers)
        if self.gru_layers < 1:
            raise ValueError("gru_layers must be at least 1")
        self.embedding = nn.Embedding(int(vocab_size), int(embedding_dim), padding_idx=int(pad_id))
        self.encoder = nn.GRU(int(embedding_dim), int(gru_hidden_size), num_layers=self.gru_layers,
                              batch_first=True, bidirectional=bool(bidirectional), dropout=0.0)
        self.output_dim = int(gru_hidden_size) * (2 if bidirectional else 1)
        if self.variant == "b0_mean":
            self.classifier = nn.Linear(self.output_dim, self.num_labels)
        elif self.variant == "b1_shared_attention":
            self.attention_projection = nn.Linear(self.output_dim, self.output_dim, bias=False)
            self.shared_query = nn.Parameter(torch.empty(self.output_dim))
            self.classifier = nn.Linear(self.output_dim, self.num_labels)
            nn.init.normal_(self.shared_query, std=0.02)
        elif self.variant in {"b2_label_attention", "l1_symmetric_local", "l2_directional_local"}:
            self.attention_projection = nn.Linear(self.output_dim, self.output_dim, bias=False)
            self.label_queries = nn.Parameter(torch.empty(self.num_labels, self.output_dim))
            self.label_scorer = nn.Parameter(torch.empty(self.num_labels, self.output_dim))
            self.label_bias = nn.Parameter(torch.zeros(self.num_labels))
            nn.init.normal_(self.label_queries, std=0.02)
            nn.init.xavier_uniform_(self.label_scorer)
            if self.variant == "l1_symmetric_local":
                self.local_neighbor_gate = nn.Parameter(torch.zeros(self.output_dim))
                self.local_norm = nn.LayerNorm(self.output_dim)
            elif self.variant == "l2_directional_local":
                self.local_left_gate = nn.Parameter(torch.zeros(self.output_dim))
                self.local_right_gate = nn.Parameter(torch.zeros(self.output_dim))
                self.local_norm = nn.LayerNorm(self.output_dim)
        else:
            raise ValueError(f"unknown variant: {variant}")

    def encode(self, input_ids, lengths):
        embedded = self.embedding(input_ids)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        encoded, _ = self.encoder(packed)
        encoded, _ = pad_packed_sequence(encoded, batch_first=True, total_length=input_ids.shape[1])
        return encoded

    @staticmethod
    def masked_softmax(scores, mask):
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        return torch.softmax(scores, dim=-1)

    def forward(self, input_ids, lengths, mask, query_permutation=None):
        hidden = self.encode(input_ids, lengths)
        if self.variant == "b0_mean":
            weights = mask.unsqueeze(-1).to(hidden.dtype)
            representation = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1)
            logits = self.classifier(representation)
            attention = mask.to(hidden.dtype) / mask.sum(1, keepdim=True).clamp_min(1)
            return {"logits": logits, "attention": attention.unsqueeze(1), "representations": representation.unsqueeze(1).expand(-1, self.num_labels, -1)}
        if self.variant in {"l1_symmetric_local", "l2_directional_local"}:
            hidden = self._add_local_context(hidden, mask)
        projected = self.attention_projection(hidden)
        if self.variant == "b1_shared_attention":
            scores = torch.einsum("bth,h->bt", projected, self.shared_query) / math.sqrt(self.output_dim)
            attention = self.masked_softmax(scores, mask)
            representation = torch.einsum("bt,bth->bh", attention, hidden)
            return {"logits": self.classifier(representation), "attention": attention.unsqueeze(1).expand(-1, self.num_labels, -1),
                    "representations": representation.unsqueeze(1).expand(-1, self.num_labels, -1)}
        queries = self.label_queries
        if query_permutation is not None:
            queries = queries.index_select(0, torch.as_tensor(query_permutation, device=queries.device))
        scores = torch.einsum("bth,lh->blt", projected, queries) / math.sqrt(self.output_dim)
        attention = self.masked_softmax(scores, mask.unsqueeze(1))
        representation = torch.einsum("blt,bth->blh", attention, hidden)
        logits = torch.einsum("blh,lh->bl", representation, self.label_scorer) + self.label_bias
        return {"logits": logits, "attention": attention, "representations": representation}

    def _add_local_context(self, hidden, mask):
        """Add masked left/right context with a fixed radius and zero-init gates."""
        values = hidden * mask.unsqueeze(-1).to(hidden.dtype)
        counts = mask.to(hidden.dtype)
        prefix_values = torch.cat([torch.zeros_like(values[:, :1]), values.cumsum(dim=1)], dim=1)
        prefix_counts = torch.cat([torch.zeros_like(counts[:, :1]), counts.cumsum(dim=1)], dim=1)
        positions = torch.arange(hidden.shape[1], device=hidden.device)
        left_index = (positions - self.local_radius).clamp_min(0)
        right_index = (positions + self.local_radius + 1).clamp_max(hidden.shape[1])
        left_sum = prefix_values.index_select(1, positions) - prefix_values.index_select(1, left_index)
        left_count = prefix_counts.index_select(1, positions) - prefix_counts.index_select(1, left_index)
        right_sum = prefix_values.index_select(1, right_index) - prefix_values.index_select(1, positions + 1)
        right_count = prefix_counts.index_select(1, right_index) - prefix_counts.index_select(1, positions + 1)
        left = left_sum / left_count.clamp_min(1).unsqueeze(-1)
        right = right_sum / right_count.clamp_min(1).unsqueeze(-1)
        if self.variant == "l1_symmetric_local":
            context = 0.5 * (left + right)
            return hidden + self.local_neighbor_gate.view(1, 1, -1) * self.local_norm(context)
        return hidden + self.local_left_gate.view(1, 1, -1) * self.local_norm(left) + self.local_right_gate.view(1, 1, -1) * self.local_norm(right)
