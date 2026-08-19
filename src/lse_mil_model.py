"""Label-specific evidence routing MIL for the isolated opcode route.

This first implementation routes evidence at chunk/view level using the
existing MLM8 cache. It deliberately does not claim token-level localization.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class LabelSpecificEvidenceRoutingMIL(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_labels = int(config.get("num_labels", 6))
        self.num_views = int(config.get("num_views", 8))
        self.context_view_index = int(config.get("context_view_index", 0))
        self.input_dim = int(config.get("feature_dim", 768))
        self.hidden_dim = int(config.get("hidden_dim", 256))
        self.slots = int(config.get("evidence_slots", 3))
        self.topk = int(config.get("evidence_topk_chunks", 0))
        dropout = float(config.get("dropout", 0.1))

        self.input_norm = nn.LayerNorm(self.input_dim)
        self.view_projection = nn.Linear(self.input_dim, self.hidden_dim)
        self.context = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=int(config.get("context_heads", 8)),
                dim_feedforward=self.hidden_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=int(config.get("context_layers", 2)),
        )
        self.label_queries = nn.Parameter(
            torch.empty(self.num_labels, self.slots, self.hidden_dim)
        )
        self.chunk_key = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.chunk_value = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.view_key = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.view_value = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.view_query = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.slot_score = nn.Linear(self.hidden_dim, 1)
        self.slot_merge = nn.Linear(self.hidden_dim, 1)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(self.hidden_dim),
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_dim // 2, 1),
            ) for _ in range(self.num_labels)
        ])
        nn.init.normal_(self.label_queries, mean=0.0, std=0.02)

    def _topk_mask(self, scores, chunk_mask):
        if self.topk <= 0 or scores.shape[1] <= self.topk:
            return chunk_mask.unsqueeze(-1).unsqueeze(-1)
        valid_count = chunk_mask.sum(dim=1).max().item()
        if int(valid_count) <= self.topk:
            return chunk_mask.unsqueeze(-1).unsqueeze(-1)
        masked = scores.masked_fill(~chunk_mask.unsqueeze(-1).unsqueeze(-1), -torch.inf)
        chosen = masked.topk(self.topk, dim=1).indices
        result = torch.zeros_like(scores, dtype=torch.bool)
        result.scatter_(1, chosen, True)
        return result & chunk_mask.unsqueeze(-1).unsqueeze(-1)

    def forward(self, features, chunk_mask, multi_labels=None, return_attention=False):
        if features.ndim != 4 or features.shape[2] != self.num_views:
            raise ValueError(
                f"features must be [B,C,{self.num_views},H], got {tuple(features.shape)}"
            )
        chunk_mask = chunk_mask.bool()
        views = self.view_projection(self.input_norm(features))
        if self.context_view_index < 0 or self.context_view_index >= self.num_views:
            raise ValueError("context_view_index is outside the selected view set")
        base = views[:, :, self.context_view_index, :]
        context = self.context(base, src_key_padding_mask=~chunk_mask)
        keys = self.view_key(views)
        values = self.view_value(views)
        queries = self.label_queries.view(1, 1, self.num_labels, self.slots, self.hidden_dim)
        queries = queries + self.view_query(context).unsqueeze(2).unsqueeze(3)
        view_logits = torch.einsum("bclsh,bcvh->bclsv", queries, keys)
        view_attention = torch.softmax(view_logits / (self.hidden_dim ** 0.5), dim=-1)
        view_summary = torch.einsum("bclsv,bcvh->bclsh", view_attention, values)
        slot_chunks = context.unsqueeze(2).unsqueeze(3) + view_summary

        chunk_keys = self.chunk_key(context)
        chunk_scores = torch.einsum("bclsh,bch->bcls", queries, chunk_keys)
        chunk_scores = chunk_scores / (self.hidden_dim ** 0.5)
        allowed = self._topk_mask(chunk_scores, chunk_mask)
        chunk_scores = chunk_scores.masked_fill(~allowed, torch.finfo(chunk_scores.dtype).min)
        chunk_attention = torch.softmax(chunk_scores, dim=1)
        slot_repr = torch.einsum("bcls,bclsh->blsh", chunk_attention, slot_chunks)
        slot_weights = torch.softmax(self.slot_merge(slot_repr).squeeze(-1), dim=-1)
        label_repr = torch.einsum("bls,blsh->blh", slot_weights, slot_repr)
        logits = torch.cat([
            head(label_repr[:, label_id]).reshape(-1, 1)
            for label_id, head in enumerate(self.heads)
        ], dim=1)

        # Penalize slots of the same label repeatedly selecting the same chunks.
        overlap = chunk_attention.permute(0, 2, 3, 1)
        diversity_loss = logits.new_zeros(())
        if self.slots > 1:
            terms = []
            for left in range(self.slots):
                for right in range(left + 1, self.slots):
                    terms.append((overlap[:, :, left] * overlap[:, :, right]).sum(dim=-1))
            diversity_loss = torch.stack(terms, dim=-1).mean()

        result = {
            "recognition_logits": logits,
            "chunk_attention": torch.einsum("bcls,bls->bcl", chunk_attention, slot_weights),
            "slot_chunk_attention": chunk_attention,
            "view_attention": view_attention,
            "slot_weights": slot_weights,
            "diversity_loss": diversity_loss,
        }
        if multi_labels is not None:
            result["bce_loss"] = F.binary_cross_entropy_with_logits(logits, multi_labels)
        return result
