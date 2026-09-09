"""Minimal BiGRU models for testing label-specific opcode aggregation."""

import math

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LabelGuidedOpcodeNet(nn.Module):
    """Shared opcode encoder with mean, shared-attention, or label-attention pooling.

    For B2, u_(l,i)=q_l^T W_a h_i, alpha_(l,i)=softmax_i(u_(l,i)),
    z_l=sum_i alpha_(l,i)h_i, and s_l=w_l^T z_l+b_l.
    """

    def __init__(self, variant, vocab_size, pad_id, embedding_dim=128,
                 gru_hidden_size=128, num_labels=6, bidirectional=True):
        super().__init__()
        self.variant = str(variant)
        self.num_labels = int(num_labels)
        self.embedding = nn.Embedding(int(vocab_size), int(embedding_dim), padding_idx=int(pad_id))
        self.encoder = nn.GRU(int(embedding_dim), int(gru_hidden_size), num_layers=1,
                              batch_first=True, bidirectional=bool(bidirectional), dropout=0.0)
        self.output_dim = int(gru_hidden_size) * (2 if bidirectional else 1)
        if self.variant == "b0_mean":
            self.classifier = nn.Linear(self.output_dim, self.num_labels)
        elif self.variant == "b1_shared_attention":
            self.attention_projection = nn.Linear(self.output_dim, self.output_dim, bias=False)
            self.shared_query = nn.Parameter(torch.empty(self.output_dim))
            self.classifier = nn.Linear(self.output_dim, self.num_labels)
            nn.init.normal_(self.shared_query, std=0.02)
        elif self.variant == "b2_label_attention":
            self.attention_projection = nn.Linear(self.output_dim, self.output_dim, bias=False)
            self.label_queries = nn.Parameter(torch.empty(self.num_labels, self.output_dim))
            self.label_scorer = nn.Parameter(torch.empty(self.num_labels, self.output_dim))
            self.label_bias = nn.Parameter(torch.zeros(self.num_labels))
            nn.init.normal_(self.label_queries, std=0.02)
            nn.init.xavier_uniform_(self.label_scorer)
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
