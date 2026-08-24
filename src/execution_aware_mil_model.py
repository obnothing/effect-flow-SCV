"""Execution-aware gated MIL over frozen MLM8 chunk views."""

import torch
from torch import nn
import torch.nn.functional as F


class ExecutionAwareMIL(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_labels = int(config.get("num_labels", 6))
        self.num_views = int(config.get("num_views", 8))
        self.input_dim = int(config.get("feature_dim", 768))
        self.exec_dim = int(config.get("execution_feature_dim", 56))
        self.hidden_dim = int(config.get("hidden_dim", 256))
        self.slots = int(config.get("evidence_slots", 3))
        dropout = float(config.get("dropout", 0.1))
        self.input_norm = nn.LayerNorm(self.input_dim)
        self.view_projection = nn.Linear(self.input_dim, self.hidden_dim)
        self.exec_norm = nn.LayerNorm(self.exec_dim)
        self.exec_projection = nn.Sequential(nn.Linear(self.exec_dim, self.hidden_dim), nn.GELU(), nn.Linear(self.hidden_dim, self.hidden_dim))
        self.exec_gate = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        nn.init.zeros_(self.exec_gate.weight)
        nn.init.zeros_(self.exec_gate.bias)
        self.context = nn.TransformerEncoder(nn.TransformerEncoderLayer(self.hidden_dim, int(config.get("context_heads", 8)), self.hidden_dim * 4, dropout=dropout, activation="gelu", batch_first=True, norm_first=True), num_layers=int(config.get("context_layers", 2)))
        self.label_queries = nn.Parameter(torch.empty(self.num_labels, self.slots, self.hidden_dim))
        self.chunk_key = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.slot_merge = nn.Linear(self.hidden_dim, 1)
        self.heads = nn.ModuleList([nn.Sequential(nn.LayerNorm(self.hidden_dim), nn.Linear(self.hidden_dim, self.hidden_dim // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(self.hidden_dim // 2, 1)) for _ in range(self.num_labels)])
        nn.init.normal_(self.label_queries, std=0.02)

    def forward(self, features, execution_features, chunk_mask, return_attention=False):
        if features.ndim != 4 or execution_features.ndim != 3:
            raise ValueError("features must be [B,C,V,H] and execution_features [B,C,F]")
        mask = chunk_mask.bool()
        views = self.view_projection(self.input_norm(features))
        base = views.mean(dim=2)
        context = self.context(base, src_key_padding_mask=~mask)
        execution = self.exec_projection(self.exec_norm(execution_features))
        gate = torch.sigmoid(self.exec_gate(torch.cat([context, execution], dim=-1)))
        fused = context + gate * execution
        queries = self.label_queries.view(1, 1, self.num_labels, self.slots, self.hidden_dim)
        queries = queries + fused.unsqueeze(2).unsqueeze(3)
        scores = torch.einsum("bclsh,bch->bcls", queries, self.chunk_key(fused)) / (self.hidden_dim ** 0.5)
        scores = scores.masked_fill(~mask[:, :, None, None], torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=1)
        slot_repr = torch.einsum("bcls,bch->blsh", attention, fused)
        weights = torch.softmax(self.slot_merge(slot_repr).squeeze(-1), dim=-1)
        label_repr = torch.einsum("bls,blsh->blh", weights, slot_repr)
        logits = torch.cat([head(label_repr[:, i]).view(-1, 1) for i, head in enumerate(self.heads)], dim=1)
        overlap = attention.permute(0, 2, 3, 1)
        diversity = logits.new_zeros(())
        if self.slots > 1:
            diversity = torch.stack([(overlap[:, :, i] * overlap[:, :, j]).sum(-1) for i in range(self.slots) for j in range(i + 1, self.slots)], dim=-1).mean()
        return {"recognition_logits": logits, "chunk_attention": torch.einsum("bcls,bls->bcl", attention, weights), "execution_gate": gate, "diversity_loss": diversity}
