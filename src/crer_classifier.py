"""Standalone Counterfactual Evidence Routing classifier.

This module is deliberately independent from the historical M0 model.  It
accepts only the cached eight-view chunk features and a valid-chunk mask, and
returns recognition logits directly.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedChunkEncoder(nn.Module):
    """Contextualize chunk features without a contract-level shortcut."""

    def __init__(self, feature_dim, hidden_dim, max_chunks, num_layers, num_heads, dropout):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.max_chunks = int(max_chunks)
        self.input_norm = nn.LayerNorm(feature_dim)
        self.input_projection = nn.Linear(feature_dim, hidden_dim)
        self.position_embedding = nn.Parameter(torch.empty(max_chunks, hidden_dim))
        self.dropout = nn.Dropout(float(dropout))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=int(num_heads),
            dim_feedforward=hidden_dim * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=int(num_layers),
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def forward(self, features, mask):
        if features.ndim != 3:
            raise ValueError("features must be [B, C, D]")
        mask = mask.bool()
        if features.shape[:2] != mask.shape:
            raise ValueError("chunk mask does not match feature shape")
        if features.shape[1] > self.max_chunks:
            raise ValueError("received more chunks than max_chunks")
        if (~mask).all(dim=1).any():
            raise ValueError("each sample must contain at least one valid chunk")
        h = self.input_projection(self.input_norm(features))
        h = self.dropout(h + self.position_embedding[: features.shape[1]].unsqueeze(0))
        h = self.encoder(h, src_key_padding_mask=(~mask).contiguous())
        return self.output_norm(h).masked_fill(~mask.unsqueeze(-1), 0.0)


class LabelConditionedEvidenceBlock(nn.Module):
    """One differentiable multi-head label-to-chunk cross-attention block."""

    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = hidden_dim // num_heads
        self.query_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.attn_dropout = nn.Dropout(float(dropout))
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(float(dropout)),
        )

    def forward(self, queries, chunks, mask):
        batch, labels, _ = queries.shape
        keys = self.key_projection(chunks).view(
            batch, chunks.shape[1], self.num_heads, self.head_dim
        ).transpose(1, 2)
        values = self.value_projection(chunks).view(
            batch, chunks.shape[1], self.num_heads, self.head_dim
        ).transpose(1, 2)
        q = self.query_projection(queries).view(
            batch, labels, self.num_heads, self.head_dim
        ).transpose(1, 2)
        scores = torch.einsum("bhld,bhcd->bhlc", q, keys)
        scores = scores / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~mask[:, None, None, :], torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        attended = torch.einsum("bhlc,bhcd->bhld", attention, values)
        attended = attended.transpose(1, 2).contiguous().view(batch, labels, self.hidden_dim)
        updated = self.attn_norm(queries + self.attn_dropout(self.output_projection(attended)))
        updated = self.ffn_norm(updated + self.ffn(updated))
        return updated, attention.mean(dim=1)


class CRERClassifier(nn.Module):
    """Standalone label-conditioned evidence detector.

    The forward path depends only on ``chunk_features`` and ``chunk_mask``.
    No M0 model, M0 logits, residual fusion, retrieval memory, or local labels
    are used here.
    """

    def __init__(
        self,
        feature_dim=768,
        num_labels=6,
        max_chunks=64,
        chunk_view_index=1,
        hidden_dim=384,
        num_heads=8,
        shared_encoder_layers=1,
        evidence_blocks=2,
        dropout=0.1,
        temperature=1.0,
        use_label_queries=True,
        sparsity_target=0.1,
        view_indices=None,
    ):
        super().__init__()
        if not 0.0 < float(temperature):
            raise ValueError("temperature must be positive")
        if int(evidence_blocks) < 1:
            raise ValueError("evidence_blocks must be positive")
        self.feature_dim = int(feature_dim)
        self.num_labels = int(num_labels)
        self.chunk_view_index = int(chunk_view_index)
        self.view_indices = [int(value) for value in (view_indices or [self.chunk_view_index])]
        if not self.view_indices or any(value < 0 or value >= 8 for value in self.view_indices):
            raise ValueError("view_indices must contain values in [0, 7]")
        self.temperature = float(temperature)
        self.use_label_queries = bool(use_label_queries)
        self.sparsity_target = float(sparsity_target)
        self.encoder = SharedChunkEncoder(
            feature_dim,
            hidden_dim,
            max_chunks,
            shared_encoder_layers,
            num_heads,
            dropout,
        )
        self.label_embedding = nn.Parameter(torch.empty(num_labels, hidden_dim))
        self.shared_query = nn.Parameter(torch.empty(1, hidden_dim))
        self.query_projection = nn.Linear(hidden_dim, hidden_dim)
        self.gate_key_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gate_value_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gate_threshold = nn.Parameter(torch.zeros(num_labels))
        self.evidence_blocks = nn.ModuleList(
            [LabelConditionedEvidenceBlock(hidden_dim, num_heads, dropout) for _ in range(int(evidence_blocks))]
        )
        self.evidence_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        branch_hidden = max(32, hidden_dim // 2)
        self.label_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, branch_hidden),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(branch_hidden, 1),
                )
                for _ in range(num_labels)
            ]
        )
        nn.init.normal_(self.label_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.shared_query, mean=0.0, std=0.02)

    def _queries(self, batch_size):
        if self.use_label_queries:
            base = self.label_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            base = self.shared_query.unsqueeze(0).expand(batch_size, self.num_labels, -1)
        return self.query_projection(base)

    def _label_heads(self, evidence):
        return torch.cat(
            [head(evidence[:, label_id, :]) for label_id, head in enumerate(self.label_heads)],
            dim=1,
        )

    def _route(self, queries, chunks, mask):
        key = self.gate_key_projection(chunks)
        value = self.gate_value_projection(chunks)
        scores = torch.einsum("blh,bch->bcl", queries, key) / math.sqrt(key.shape[-1])
        scores = scores / self.temperature
        scores = scores.masked_fill(~mask.unsqueeze(-1), torch.finfo(scores.dtype).min)
        threshold = self.gate_threshold.view(1, 1, -1)
        gates = torch.sigmoid(scores - threshold)
        gates = gates * mask.unsqueeze(-1).to(dtype=gates.dtype)
        denom = gates.sum(dim=1).clamp_min(1e-6)
        weights = gates / denom.unsqueeze(1)
        evidence = torch.einsum("bcl,bch->blh", weights, value)
        complement_gates = (1.0 - gates) * mask.unsqueeze(-1).to(dtype=gates.dtype)
        complement_denom = complement_gates.sum(dim=1).clamp_min(1e-6)
        complement_weights = complement_gates / complement_denom.unsqueeze(1)
        complement = torch.einsum("bcl,bch->blh", complement_weights, value)
        return scores, gates, weights, evidence, complement

    def forward(
        self,
        chunk_features,
        chunk_mask,
        return_diagnostics=False,
        routing_override=None,
        evidence_override=None,
    ):
        if chunk_features.ndim != 4 or chunk_features.shape[2] <= self.chunk_view_index:
            raise ValueError("CRER expects [B, C, V, D] features with configured chunk view")
        if chunk_features.shape[-1] != self.feature_dim:
            raise ValueError("feature_dim does not match CRER configuration")
        mask = chunk_mask.bool()
        if (~mask).all(dim=1).any():
            raise ValueError("each sample must contain at least one valid chunk")
        chunks = chunk_features[:, :, self.view_indices, :].mean(dim=2)
        contextual = self.encoder(chunks, mask)
        queries = self._queries(chunks.shape[0])
        block_attentions = []
        for block in self.evidence_blocks:
            queries, attention = block(queries, contextual, mask)
            block_attentions.append(attention)
        scores, gates, weights, evidence_raw, complement_raw = self._route(
            queries, contextual, mask
        )
        if routing_override not in (None, "learned", "uniform", "zero"):
            raise ValueError("routing_override must be learned, uniform, zero, or None")
        if routing_override == "uniform":
            weights = mask.unsqueeze(-1).to(dtype=contextual.dtype)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
            evidence_raw = torch.einsum("bcl,bch->blh", weights, self.gate_value_projection(contextual))
        elif routing_override == "zero":
            evidence_raw = torch.zeros_like(evidence_raw)
        evidence = self.evidence_projection(evidence_raw)
        if routing_override == "zero":
            evidence = torch.zeros_like(evidence)
        if evidence_override is not None:
            if evidence_override.shape != evidence.shape:
                raise ValueError(
                    "evidence_override must match [batch, labels, hidden_dim]"
                )
            evidence = evidence_override.to(dtype=evidence.dtype)
        complement = self.evidence_projection(complement_raw)
        logits = self._label_heads(evidence)
        complement_logits = self._label_heads(complement)
        gate_distribution = weights.clamp_min(1e-12)
        entropy = -(gate_distribution * gate_distribution.log()).sum(dim=1)
        valid_count = mask.sum(dim=1, keepdim=True).to(dtype=weights.dtype)
        top_k = min(10, contextual.shape[1])
        rank_scores, rank_indices = torch.topk(scores, k=top_k, dim=1)
        result = {
            "recognition_logits": logits,
            "complement_logits": complement_logits,
            "counterfactual_delta": logits - complement_logits,
            "chunk_representation": contextual,
            "evidence_representation": evidence,
            "complement_representation": complement,
            "routing_scores": scores,
            "routing_weights": weights,
            "gate": gates,
            "gate_fraction": gates.sum(dim=1) / valid_count,
            "evidence_entropy": entropy,
            "effective_evidence_count": entropy.exp(),
            "ranked_indices": rank_indices.permute(0, 2, 1),
            "ranked_scores": rank_scores.permute(0, 2, 1),
            "block_attentions": block_attentions,
            "chunk_mask": mask,
        }
        if return_diagnostics:
            normalized = F.normalize(evidence, dim=-1)
            result["label_evidence_cosine"] = torch.einsum("blh,bmh->blm", normalized, normalized)
        return result

    def compute_loss(
        self,
        output,
        labels,
        pos_weight,
        lambda_cf=0.1,
        lambda_sparse=0.01,
        lambda_entropy=0.0,
        margin=0.1,
    ):
        labels = labels.float()
        cls_loss = F.binary_cross_entropy_with_logits(
            output["recognition_logits"], labels, pos_weight=pos_weight
        )
        delta = output["counterfactual_delta"]
        positive_loss = labels * F.relu(float(margin) - delta)
        negative_loss = (1.0 - labels) * F.relu(delta)
        cf_loss = (positive_loss + negative_loss).mean()
        fractions = output["gate_fraction"]
        sparse_loss = (fractions - float(self.sparsity_target)).pow(2).mean()
        distribution = output["routing_weights"].clamp_min(1e-12)
        entropy = -(distribution * distribution.log()).sum(dim=1)
        entropy_loss = entropy.mean()
        total = (
            cls_loss
            + float(lambda_cf) * cf_loss
            + float(lambda_sparse) * sparse_loss
            + float(lambda_entropy) * entropy_loss
        )
        return {
            "loss": total,
            "classification_loss": cls_loss,
            "counterfactual_loss": cf_loss,
            "sparsity_loss": sparse_loss,
            "entropy_loss": entropy_loss,
        }


def parameter_group_specs():
    return {
        "label_embedding": lambda name: name.startswith("label_embedding"),
        "query_projection": lambda name: name.startswith("query_projection") or ".query_projection" in name,
        "key_projection": lambda name: name.startswith("gate_key_projection") or ".key_projection" in name,
        "value_projection": lambda name: name.startswith("gate_value_projection") or ".value_projection" in name,
        "evidence_block_1": lambda name: name.startswith("evidence_blocks.0."),
        "evidence_block_2": lambda name: name.startswith("evidence_blocks.1."),
        "feature_encoder": lambda name: name.startswith("encoder."),
        "prediction_head": lambda name: name.startswith("label_heads."),
    }


def parameter_group_diagnostics(model, initial=None):
    rows = []
    for group, matches in parameter_group_specs().items():
        params = [(name, p) for name, p in model.named_parameters() if matches(name)]
        grad_values = [p.grad.detach().float().norm().pow(2) for _, p in params if p.grad is not None]
        norm_values = [p.detach().float().norm().pow(2) for _, p in params]
        grad_sq = torch.stack(grad_values).sum() if grad_values else torch.tensor(0.0)
        norm_sq = torch.stack(norm_values).sum() if norm_values else torch.tensor(0.0)
        update_values = []
        if initial is not None:
            update_values = [
                (p.detach().float().cpu() - initial[name]).norm().pow(2)
                for name, p in params
            ]
        update_sq = torch.stack(update_values).sum() if update_values else torch.tensor(0.0)
        rows.append(
            {
                "group": group,
                "grad_norm": float(grad_sq.sqrt()),
                "param_norm": float(norm_sq.sqrt()),
                "update_norm": float(update_sq.sqrt()),
            }
        )
    return rows
