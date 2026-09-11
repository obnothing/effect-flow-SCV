"""Searchable standard sequence backbones with frozen A0-A3 aggregation semantics."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class AutoTuneSequenceNet(nn.Module):
    """Configurable opcode sequence model for the persistent AutoTune study.

    A0 uses masked mean pooling, A1 uses shared attention, A2 is the current
    B2-style label attention, and A3 uses the same label-specific aggregation
    after the searchable stronger backbone. No retrieval, graph, prototype, or
    test data is involved.
    """

    def __init__(self, config, vocab_size, pad_id):
        super().__init__()
        self.config = dict(config)
        self.variant = str(config.get("aggregation", config.get("variant", "a3_vsfs")))
        self.num_labels = int(config.get("num_labels", 6))
        self.embedding_dim = int(config["embedding_dim"])
        self.hidden_size = int(config["hidden"])
        self.layers = int(config["layers"])
        self.dropout_rate = float(config.get("dropout", 0.1))
        encoder = str(config["encoder"])
        self.bidirectional = encoder.startswith("Bi")
        rnn_class = nn.LSTM if "LSTM" in encoder else nn.GRU
        self.embedding = nn.Embedding(int(vocab_size), self.embedding_dim, padding_idx=int(pad_id))
        self.embedding_dropout = nn.Dropout(self.dropout_rate)
        if bool(config.get("embedding_mlp", False)):
            self.embedding_mlp = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim), nn.LayerNorm(self.embedding_dim), nn.ELU(),
                nn.Dropout(self.dropout_rate), nn.Linear(self.embedding_dim, self.embedding_dim),
            )
        else:
            self.embedding_mlp = nn.Identity()
        rnn_dropout = self.dropout_rate if self.layers > 1 else 0.0
        self.encoder = rnn_class(self.embedding_dim, self.hidden_size, num_layers=self.layers,
                                 batch_first=True, bidirectional=self.bidirectional, dropout=rnn_dropout)
        self.output_dim = self.hidden_size * (2 if self.bidirectional else 1)
        self.self_attention_enabled = bool(config.get("self_attention", False))
        self.residual_enabled = bool(config.get("residual", False))
        if self.self_attention_enabled:
            heads = int(config.get("heads", 1))
            if self.output_dim % heads != 0:
                raise ValueError(f"attention heads={heads} must divide output_dim={self.output_dim}")
            self.self_attention = nn.MultiheadAttention(self.output_dim, heads, batch_first=True, dropout=self.dropout_rate)
            self.self_attention_norm = nn.LayerNorm(self.output_dim)
        if self.variant in {"a1_shared", "a2_b2", "a3_vsfs"}:
            self.attention_projection = nn.Linear(self.output_dim, self.output_dim, bias=False)
        if self.variant == "a1_shared":
            self.shared_query = nn.Parameter(torch.empty(self.output_dim))
            nn.init.normal_(self.shared_query, std=0.02)
        elif self.variant in {"a2_b2", "a3_vsfs"}:
            self.label_queries = nn.Parameter(torch.empty(self.num_labels, self.output_dim))
            self.label_scorer = nn.Parameter(torch.empty(self.num_labels, self.output_dim))
            self.label_bias = nn.Parameter(torch.zeros(self.num_labels))
            nn.init.normal_(self.label_queries, std=0.02)
            nn.init.xavier_uniform_(self.label_scorer)
        if self.variant in {"a0_mean", "a1_shared"}:
            middle = max(self.output_dim // 2, self.num_labels)
            self.output_mlp = nn.Sequential(nn.Linear(self.output_dim, middle), nn.GELU(),
                                            nn.Dropout(self.dropout_rate), nn.Linear(middle, self.num_labels))
        elif self.variant not in {"a2_b2", "a3_vsfs"}:
            raise ValueError(f"unknown aggregation variant: {self.variant}")

    def encode(self, input_ids, lengths, mask):
        embedded = self.embedding_dropout(self.embedding_mlp(self.embedding(input_ids)))
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        encoded, _ = self.encoder(packed)
        hidden, _ = pad_packed_sequence(encoded, batch_first=True, total_length=input_ids.shape[1])
        if self.self_attention_enabled:
            # Fixed stride keeps MHA linear in the number of long-contract
            # summaries instead of quadratic in the raw opcode length.
            stride = 16
            context = hidden[:, ::stride]
            context_mask = mask[:, ::stride]
            attended, _ = self.self_attention(context, context, context, key_padding_mask=~context_mask, need_weights=False)
            attended = self.self_attention_norm(attended + context if self.residual_enabled else attended)
            expanded = F.interpolate(attended.transpose(1, 2), size=hidden.shape[1], mode="linear", align_corners=False).transpose(1, 2)
            hidden = self.self_attention_norm(hidden + expanded if self.residual_enabled else expanded)
        return hidden

    @staticmethod
    def masked_softmax(scores, mask):
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        return torch.softmax(scores, dim=-1)

    def forward(self, input_ids, lengths, mask):
        hidden = self.encode(input_ids, lengths, mask)
        if self.variant == "a0_mean":
            weights = mask.unsqueeze(-1).to(hidden.dtype)
            representation = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1)
            logits = self.output_mlp(representation)
            attention = mask.to(hidden.dtype).unsqueeze(1) / mask.sum(1, keepdim=True).clamp_min(1).unsqueeze(1)
            return {"logits": logits, "attention": attention, "representations": representation.unsqueeze(1).expand(-1, self.num_labels, -1)}
        projected = self.attention_projection(hidden)
        if self.variant == "a1_shared":
            scores = torch.einsum("bth,h->bt", projected, self.shared_query) / math.sqrt(self.output_dim)
            attention = self.masked_softmax(scores, mask)
            representation = torch.einsum("bt,bth->bh", attention, hidden)
            return {"logits": self.output_mlp(representation), "attention": attention.unsqueeze(1).expand(-1, self.num_labels, -1),
                    "representations": representation.unsqueeze(1).expand(-1, self.num_labels, -1)}
        scores = torch.einsum("bth,lh->blt", projected, self.label_queries) / math.sqrt(self.output_dim)
        attention = self.masked_softmax(scores, mask.unsqueeze(1))
        representation = torch.einsum("blt,bth->blh", attention, hidden)
        logits = torch.einsum("blh,lh->bl", representation, self.label_scorer) + self.label_bias
        return {"logits": logits, "attention": attention, "representations": representation}
