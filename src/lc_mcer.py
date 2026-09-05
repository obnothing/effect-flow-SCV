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
        include_base_logit=False,
        routing_mode="hard",
        routing_temperature=0.5,
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
        self.include_base_logit = bool(include_base_logit)
        if routing_mode not in ("hard", "soft"):
            raise ValueError("routing_mode must be 'hard' or 'soft'")
        self.routing_mode = str(routing_mode)
        self.routing_temperature = float(routing_temperature)
        if self.routing_temperature <= 0:
            raise ValueError("routing_temperature must be positive")
        self.query_projection = nn.Linear(self.feature_dim, self.retrieval_dim)
        self.chunk_projection = nn.Linear(self.feature_dim, self.retrieval_dim)
        self.label_embedding = nn.Parameter(torch.empty(self.num_labels, self.retrieval_dim))
        residual_input_dim = self.feature_dim * 2 + (1 if self.include_base_logit else 0)
        self.residual_classifier = nn.Sequential(
            nn.LayerNorm(residual_input_dim),
            nn.Linear(residual_input_dim, int(residual_hidden_dim)),
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

    def forward(
        self,
        chunk_features,
        chunk_mask,
        return_diagnostics=False,
        evidence_override=None,
    ):
        if chunk_features.ndim != 4 or chunk_features.shape[2] != 8:
            raise ValueError("LC-MCER expects [B, C, 8, feature_dim] chunk features")
        mask = chunk_mask.to(dtype=torch.bool)
        batch = chunk_features.shape[0]
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
        use_soft_routing = self.routing_mode == "soft" and self.training
        if use_soft_routing:
            routing_weights = torch.softmax(
                competition_scores / self.routing_temperature, dim=1
            )
            evidence = torch.einsum("bcl,bch->blh", routing_weights, chunks)
        else:
            routing_weights = torch.zeros(
                batch, chunks.shape[1], self.num_labels,
                dtype=weights.dtype, device=weights.device,
            )
            routing_weights.scatter_add_(
                1,
                selected_indices.permute(0, 2, 1),
                (weights * selected_mask.to(dtype=weights.dtype)).permute(0, 2, 1),
            )
        if evidence_override is not None:
            if evidence_override.shape != (batch, self.num_labels, self.feature_dim):
                raise ValueError(
                    "evidence_override must have shape "
                    f"[{batch}, {self.num_labels}, {self.feature_dim}]"
                )
            evidence = evidence_override.to(dtype=chunks.dtype)
        residual_parts = [
            contract.unsqueeze(1).expand(-1, self.num_labels, -1),
            evidence,
        ]
        if self.include_base_logit:
            residual_parts.append(base_logits.unsqueeze(-1))
        residual_input = torch.cat(
            residual_parts, dim=-1
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
            "evidence_representation": evidence,
            "routing_weights": routing_weights,
            "routing_scores": competition_scores,
            "routing_mode_used": "soft" if use_soft_routing else "hard",
        }
        if return_diagnostics:
            result["contract_representation"] = contract
            result["chunk_representation"] = chunks
            result["selected_mask"] = selected_mask
        return result


class CRER(nn.Module):
    """Counterfactual Residual Evidence Routing.

    The M0 detector remains frozen. Evidence is learned only through the
    classification objective and optional counterfactual losses; no local
    evidence labels or prototype memory are used.
    """

    def __init__(
        self,
        base_model,
        feature_dim,
        num_labels,
        retrieval_dim=256,
        gate_temperature=0.5,
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
        self.gate_temperature = float(gate_temperature)
        if self.gate_temperature <= 0:
            raise ValueError("gate_temperature must be positive")
        self.residual_scale = float(residual_scale)
        self.query_projection = nn.Linear(self.feature_dim, self.retrieval_dim)
        self.chunk_projection = nn.Linear(self.feature_dim, self.retrieval_dim)
        self.label_embedding = nn.Parameter(
            torch.empty(self.num_labels, self.retrieval_dim)
        )
        self.gate_threshold = nn.Parameter(torch.zeros(self.num_labels))
        residual_input_dim = self.feature_dim * 2 + 1
        self.residual_classifier = nn.Sequential(
            nn.LayerNorm(residual_input_dim),
            nn.Linear(residual_input_dim, int(residual_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(residual_hidden_dim), 1),
        )
        nn.init.normal_(self.label_embedding, mean=0.0, std=0.02)

    def train(self, mode=True):
        super().train(mode)
        self.base_model.eval()
        return self

    @staticmethod
    def _masked_mean(values, mask):
        weights = mask.unsqueeze(-1).to(dtype=values.dtype)
        return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def _residual(self, contract, evidence, base_logits):
        contract = contract.unsqueeze(1).expand(-1, self.num_labels, -1)
        base = base_logits.unsqueeze(-1).expand(-1, -1, 1)
        inputs = torch.cat([contract, evidence, base], dim=-1)
        return self.residual_classifier(inputs).squeeze(-1)

    def _counterfactual_residuals(self, contract, evidence_full, evidence_empty, base_logits):
        # Use the same deterministic head for both worlds so Delta measures
        # evidence effect rather than two independent dropout masks.
        was_training = self.residual_classifier.training
        self.residual_classifier.eval()
        residual_full = self._residual(contract, evidence_full, base_logits)
        residual_empty = self._residual(contract, evidence_empty, base_logits)
        self.residual_classifier.train(was_training)
        return residual_full, residual_empty

    def forward(self, chunk_features, chunk_mask, return_diagnostics=False):
        if chunk_features.ndim != 4 or chunk_features.shape[2] != 8:
            raise ValueError("CRER expects [B, C, 8, feature_dim] chunk features")
        mask = chunk_mask.to(dtype=torch.bool)
        if (~mask).all(dim=1).any():
            raise ValueError("each sample must contain at least one valid chunk")
        chunks = chunk_features[:, :, 1, :]
        if chunks.shape[-1] != self.feature_dim:
            raise ValueError(
                f"expected feature_dim={self.feature_dim}, got {chunks.shape[-1]}"
            )
        with torch.no_grad():
            base_output = self.base_model(chunk_features, mask)
            base_logits = base_output["recognition_logits"].float()
        contract = self._masked_mean(chunks, mask)
        projected = self.chunk_projection(chunks)
        query = self.query_projection(contract).unsqueeze(1)
        query = query + self.label_embedding.unsqueeze(0)
        scores = torch.einsum("bld,bcd->bcl", query, projected)
        scores = scores / math.sqrt(self.retrieval_dim)
        scores = scores.masked_fill(~mask.unsqueeze(-1), 0.0)
        thresholds = self.gate_threshold.view(1, 1, -1)
        gates = torch.sigmoid((scores - thresholds) / self.gate_temperature)
        gates = gates * mask.unsqueeze(-1).to(dtype=gates.dtype)
        denominator = gates.sum(dim=1).clamp_min(1e-6)
        evidence_full = torch.einsum("bcl,bch->blh", gates, chunks) / denominator.unsqueeze(-1)
        evidence_empty = torch.zeros_like(evidence_full)
        residual_full, residual_empty = self._counterfactual_residuals(
            contract, evidence_full, evidence_empty, base_logits
        )
        counterfactual_delta = residual_full - residual_empty
        logits = base_logits + self.residual_scale * residual_full
        active_count = mask.sum(dim=1, keepdim=True).to(dtype=gates.dtype)
        gate_distribution = gates / denominator.unsqueeze(1)
        gate_entropy = -(
            gate_distribution * gate_distribution.clamp_min(1e-12).log()
        ).sum(dim=1)
        result = {
            "recognition_logits": logits,
            "base_logits": base_logits,
            "residual_logits": residual_full,
            "residual_empty": residual_empty,
            "counterfactual_delta": counterfactual_delta,
            "evidence_representation": evidence_full,
            "gate": gates,
            "gate_entropy": gate_entropy,
            "active_chunk_count": (gates > 0.5).sum(dim=1).float(),
            "routing_scores": scores,
            "routing_weights": gate_distribution,
            "chunk_mask": mask,
        }
        if return_diagnostics:
            result["contract_representation"] = contract
            result["chunk_representation"] = chunks
            result["chunk_mask"] = mask
        return result

    def auxiliary_loss(
        self,
        output,
        labels,
        pos_weight,
        *,
        counterfactual_weight=0.0,
        error_weighting=False,
        preservation_weight=0.0,
        sparsity_weight=0.0,
        sparsity_target=0.1,
        margin=0.1,
    ):
        cls_loss = F.binary_cross_entropy_with_logits(
            output["recognition_logits"], labels, pos_weight=pos_weight
        )
        delta = output["counterfactual_delta"]
        target_direction = labels.mul(2.0).sub(1.0)
        if error_weighting:
            error_weight = (labels - torch.sigmoid(output["base_logits"])).abs().detach()
        else:
            error_weight = torch.ones_like(labels)
        cf_loss = (
            error_weight * F.relu(float(margin) - target_direction * delta)
        ).mean()
        preserve_weight = float(preservation_weight)
        preserve_loss = (
            (1.0 - error_weight) * delta.abs()
        ).mean()
        gates = output["gate"]
        active = output["chunk_mask"].to(dtype=gates.dtype)
        gate_fraction = gates.sum(dim=1) / active.sum(dim=1, keepdim=True).clamp_min(1.0)
        sparse_loss = (gate_fraction - float(sparsity_target)).abs().mean()
        total = (
            cls_loss
            + float(counterfactual_weight) * cf_loss
            + preserve_weight * preserve_loss
            + float(sparsity_weight) * sparse_loss
        )
        return {
            "loss": total,
            "classification_loss": cls_loss,
            "counterfactual_loss": cf_loss,
            "preservation_loss": preserve_loss,
            "sparsity_loss": sparse_loss,
        }


class LCMCERV2(LCMCER):
    """LC-MCER-v2 with train-only EMA prototypes for weak evidence learning."""

    def __init__(self, *args, prototype_momentum=0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.prototype_momentum = float(prototype_momentum)
        if not 0.0 <= self.prototype_momentum < 1.0:
            raise ValueError("prototype_momentum must be in [0, 1)")
        self.register_buffer(
            "label_prototypes",
            torch.zeros(self.num_labels, self.feature_dim),
        )
        self.register_buffer("prototype_counts", torch.zeros(self.num_labels))

    def evidence_discrimination_loss(self, evidence, labels, temperature=0.1):
        if temperature <= 0:
            raise ValueError("evidence contrastive temperature must be positive")
        normalized = F.normalize(evidence, dim=-1)
        prototypes = F.normalize(self.label_prototypes, dim=-1)
        losses = []
        for row in range(evidence.shape[0]):
            for label_id in torch.where(labels[row] > 0.5)[0].tolist():
                if self.prototype_counts[label_id] <= 0:
                    continue
                positive = torch.sum(normalized[row, label_id] * prototypes[label_id])
                logits = [positive]
                for other_id in range(self.num_labels):
                    if other_id == label_id or self.prototype_counts[other_id] <= 0:
                        continue
                    logits.append(torch.sum(normalized[row, label_id] * prototypes[other_id]))
                negative_rows = torch.where(labels[:, label_id] < 0.5)[0]
                if len(negative_rows):
                    hard_scores = torch.sum(
                        normalized[negative_rows, label_id] * normalized[row, label_id], dim=-1
                    )
                    logits.append(hard_scores.max())
                if len(logits) > 1:
                    values = torch.stack(logits) / float(temperature)
                    losses.append(-values[0] + torch.logsumexp(values, dim=0))
        if not losses:
            return evidence.sum() * 0.0
        return torch.stack(losses).mean()

    @torch.no_grad()
    def update_prototypes(self, evidence, labels):
        normalized = F.normalize(evidence.detach(), dim=-1)
        for label_id in range(self.num_labels):
            rows = torch.where(labels[:, label_id] > 0.5)[0]
            if len(rows) == 0:
                continue
            batch_value = F.normalize(normalized[rows, label_id].mean(dim=0), dim=0)
            if self.prototype_counts[label_id] <= 0:
                self.label_prototypes[label_id].copy_(batch_value)
            else:
                value = (
                    self.prototype_momentum * self.label_prototypes[label_id]
                    + (1.0 - self.prototype_momentum) * batch_value
                )
                self.label_prototypes[label_id].copy_(F.normalize(value, dim=0))
            self.prototype_counts[label_id] += float(len(rows))
