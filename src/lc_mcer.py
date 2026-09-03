"""Label-Conditioned Multi-Chunk Evidence Retrieval (LC-MCER)."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LCMCER(nn.Module):
    """A frozen M0 plus a classification-supervised evidence residual path."""

    def __init__(
        self,
        base_model,
        feature_dim,
        num_labels,
        retrieval_dim=256,
        top_k=5,
        diversity_lambda=0.1,
        use_competition=True,
        residual_scale=0.1,
        residual_hidden_dim=256,
        dropout=0.1,
    ):
        super().__init__()
        self.base_model = base_model
        for parameter in self.base_model.parameters():
            parameter.requires_grad = False
        self.feature_dim = int(feature_dim)
        self.num_labels = int(num_labels)
        self.retrieval_dim = int(retrieval_dim)
        self.top_k = int(top_k)
        self.diversity_lambda = float(diversity_lambda)
        self.use_competition = bool(use_competition)
        self.query_projection = nn.Linear(self.feature_dim, self.retrieval_dim)
        self.chunk_projection = nn.Linear(self.feature_dim, self.retrieval_dim)
        self.label_embedding = nn.Parameter(torch.empty(self.num_labels, self.retrieval_dim))
        self.residual_classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim * 2),
            nn.Linear(self.feature_dim * 2, int(residual_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(residual_hidden_dim), 1),
        )
        self.residual_scale = float(residual_scale)
        nn.init.normal_(self.label_embedding, mean=0.0, std=0.02)

    def train(self, mode=True):
        super().train(mode)
        self.base_model.eval()
        return self

    @staticmethod
    def _masked_mean(values, mask):
        weights = mask.unsqueeze(-1).to(dtype=values.dtype)
        return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def _select_diverse(self, scores, projected, mask):
        batch, chunks, labels = scores.shape
        k = min(self.top_k, chunks)
        normalized = F.normalize(projected, dim=-1)
        selected_indices = torch.zeros(batch, labels, k, dtype=torch.long, device=scores.device)
        selected_scores = torch.zeros(batch, labels, k, dtype=scores.dtype, device=scores.device)
        selected_mask = torch.zeros(batch, labels, k, dtype=torch.bool, device=scores.device)
        for batch_id in range(batch):
            active = mask[batch_id]
            active_count = int(active.sum().detach().cpu().item())
            take = min(k, active_count)
            if take == 0:
                continue
            for label_id in range(labels):
                available = active.clone()
                chosen = []
                for rank in range(take):
                    adjusted = scores[batch_id, :, label_id].clone()
                    if chosen and self.diversity_lambda > 0:
                        similarity = normalized[batch_id] @ normalized[batch_id, chosen].transpose(0, 1)
                        adjusted = adjusted - self.diversity_lambda * similarity.max(dim=1).values
                    adjusted = adjusted.masked_fill(~available, torch.finfo(adjusted.dtype).min)
                    index = int(torch.argmax(adjusted).detach().cpu().item())
                    chosen.append(index)
                    available[index] = False
                selected_indices[batch_id, label_id, :take] = torch.tensor(chosen, device=scores.device)
                selected_scores[batch_id, label_id, :take] = scores[batch_id, chosen, label_id]
                selected_mask[batch_id, label_id, :take] = True
        return selected_indices, selected_scores, selected_mask

    def forward(self, chunk_features, chunk_mask, return_diagnostics=False):
        if chunk_features.ndim != 4 or chunk_features.shape[2] != 8:
            raise ValueError("LC-MCER expects [B, C, 8, feature_dim] chunk features")
        mask = chunk_mask.to(dtype=torch.bool)
        chunks = chunk_features[:, :, 1, :]
        if chunks.shape[-1] != self.feature_dim:
            raise ValueError(f"expected feature_dim={self.feature_dim}, got {chunks.shape[-1]}")
        if (~mask).all(dim=1).any():
            raise ValueError("each sample must contain at least one valid chunk")
        with torch.no_grad():
            baseline = self.base_model(chunk_features, mask)
            base_logits = baseline["recognition_logits"].float()
        contract = self._masked_mean(chunks, mask)
        projected = self.chunk_projection(chunks)
        query = self.query_projection(contract).unsqueeze(1) + self.label_embedding.unsqueeze(0)
        scores = torch.einsum("bld,bcd->bcl", query, projected) / math.sqrt(self.retrieval_dim)
        scores = scores.masked_fill(~mask.unsqueeze(-1), torch.finfo(scores.dtype).min)
        competition_scores = scores
        if self.use_competition:
            competition_scores = scores - torch.logsumexp(scores, dim=-1, keepdim=True)
        analysis_k = min(10, chunks.shape[1])
        ranked_scores, ranked_indices = torch.topk(competition_scores, k=analysis_k, dim=1)
        selected_indices, selected_scores, selected_mask = self._select_diverse(
            competition_scores, projected, mask
        )
        weighted_scores = selected_scores.masked_fill(
            ~selected_mask, torch.finfo(selected_scores.dtype).min
        )
        weights = torch.softmax(weighted_scores, dim=-1)
        weights = weights * selected_mask.to(dtype=weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        expanded = chunks.unsqueeze(2).expand(-1, -1, self.num_labels, -1)
        gather_index = selected_indices.permute(0, 2, 1).unsqueeze(-1).expand(
            -1, -1, -1, self.feature_dim
        )
        gathered = torch.gather(expanded, 1, gather_index).permute(0, 2, 1, 3)
        evidence = (gathered * weights.unsqueeze(-1)).sum(dim=2)
        residual_input = torch.cat(
            [contract.unsqueeze(1).expand(-1, self.num_labels, -1), evidence], dim=-1
        )
        residual = self.residual_classifier(residual_input).squeeze(-1)
        logits = base_logits + self.residual_scale * residual
        result = {
            "recognition_logits": logits,
            "base_logits": base_logits,
            "residual_logits": residual,
            "evidence_indices": selected_indices,
            "evidence_scores": selected_scores,
            "evidence_weights": weights,
            "ranked_indices": ranked_indices.permute(0, 2, 1),
            "ranked_scores": ranked_scores.permute(0, 2, 1),
            "competition_scores": competition_scores,
        }
        if return_diagnostics:
            result["contract_representation"] = contract
            result["chunk_representation"] = chunks
            result["selected_mask"] = selected_mask
        return result
