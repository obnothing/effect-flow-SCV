"""Label-wise pairwise and EMA dual-prototype representation objectives."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelDecoupledPrototype(nn.Module):
    """Training-only label-decoupled contrast for Z shaped [B, L, D]."""

    def __init__(self, num_labels, dim, momentum=0.95, temperature=0.1):
        super().__init__()
        self.num_labels = int(num_labels)
        self.dim = int(dim)
        self.momentum = float(momentum)
        self.temperature = float(temperature)
        self.register_buffer("positive_prototypes", torch.zeros(self.num_labels, self.dim))
        self.register_buffer("negative_prototypes", torch.zeros(self.num_labels, self.dim))
        self.register_buffer("initialized", torch.zeros(self.num_labels, 2, dtype=torch.bool))

    @torch.no_grad()
    def update(self, representations, labels):
        normalized = F.normalize(representations.detach().float(), dim=-1)
        labels = labels.detach().float()
        for label_id in range(self.num_labels):
            for state, target, storage_index in ((1, labels[:, label_id] > 0.5, 0), (0, labels[:, label_id] <= 0.5, 1)):
                if not bool(target.any()):
                    continue
                mean = F.normalize(normalized[target, label_id].mean(dim=0, keepdim=True), dim=-1).squeeze(0)
                prototype = self.positive_prototypes if state == 1 else self.negative_prototypes
                if bool(self.initialized[label_id, storage_index]):
                    prototype[label_id].mul_(self.momentum).add_(mean, alpha=1.0 - self.momentum)
                    prototype[label_id].copy_(F.normalize(prototype[label_id], dim=0))
                else:
                    prototype[label_id].copy_(mean)
                    self.initialized[label_id, storage_index] = True

    def prototype_loss(self, representations, labels):
        normalized = F.normalize(representations, dim=-1)
        losses = []
        active = []
        for label_id in range(self.num_labels):
            if not bool(self.initialized[label_id].all()):
                continue
            positive = normalized[:, label_id] @ self.positive_prototypes[label_id]
            negative = normalized[:, label_id] @ self.negative_prototypes[label_id]
            logits = torch.stack([negative, positive], dim=-1) / self.temperature
            target = labels[:, label_id].long()
            parts = []
            for class_id in (0, 1):
                selected = target == class_id
                if bool(selected.any()):
                    parts.append(F.cross_entropy(logits[selected], target[selected]))
            if parts:
                losses.append(torch.stack(parts).mean())
                active.append(label_id)
        if not losses:
            return representations.sum() * 0.0, active
        return torch.stack(losses).mean(), active

    def diagnostics(self, representations, labels):
        normalized = F.normalize(representations.detach().float(), dim=-1)
        result = {"prototype_separation": [], "positive_margin": [], "negative_margin": [], "positive_intra_cosine": [], "negative_intra_cosine": [], "positive_negative_cosine": []}
        for label_id in range(self.num_labels):
            if bool(self.initialized[label_id].all()):
                result["prototype_separation"].append(float(self.positive_prototypes[label_id] @ self.negative_prototypes[label_id]))
            else:
                result["prototype_separation"].append(None)
            pos = labels[:, label_id] > 0.5
            neg = ~pos
            pos_repr = normalized[pos, label_id]
            neg_repr = normalized[neg, label_id]
            if bool(self.initialized[label_id].all()):
                pos_plus = pos_repr @ self.positive_prototypes[label_id] if len(pos_repr) else torch.empty(0)
                pos_minus = pos_repr @ self.negative_prototypes[label_id] if len(pos_repr) else torch.empty(0)
                neg_minus = neg_repr @ self.negative_prototypes[label_id] if len(neg_repr) else torch.empty(0)
                neg_plus = neg_repr @ self.positive_prototypes[label_id] if len(neg_repr) else torch.empty(0)
                result["positive_margin"].append(float((pos_plus - pos_minus).mean()) if len(pos_repr) else None)
                result["negative_margin"].append(float((neg_minus - neg_plus).mean()) if len(neg_repr) else None)
            else:
                result["positive_margin"].append(None); result["negative_margin"].append(None)
            result["positive_intra_cosine"].append(float((pos_repr @ pos_repr.T).mean()) if len(pos_repr) else None)
            result["negative_intra_cosine"].append(float((neg_repr @ neg_repr.T).mean()) if len(neg_repr) else None)
            result["positive_negative_cosine"].append(float((pos_repr @ neg_repr.T).mean()) if len(pos_repr) and len(neg_repr) else None)
        return result


def pairwise_supervised_contrast(representations, labels, temperature=0.1):
    """Batch-only label-decoupled supervised contrastive loss."""
    normalized = F.normalize(representations, dim=-1)
    values = []
    for label_id in range(labels.shape[1]):
        similarity = normalized[:, label_id] @ normalized[:, label_id].T / float(temperature)
        target = labels[:, label_id] > 0.5
        for anchor in range(len(target)):
            positive = target == target[anchor]
            positive[anchor] = False
            denominator = torch.ones(len(target), dtype=torch.bool, device=target.device)
            denominator[anchor] = False
            if not bool(positive.any()) or not bool(denominator.any()):
                continue
            log_probability = similarity[anchor] - torch.logsumexp(similarity[anchor][denominator], dim=0)
            values.append(-log_probability[positive].mean())
    if not values:
        return representations.sum() * 0.0
    return torch.stack(values).mean()
