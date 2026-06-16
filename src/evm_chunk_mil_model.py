import torch
import torch.nn as nn
import torch.nn.functional as F


class EVMChunkMILClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_labels = int(config.get("num_labels", 10))
        feature_dim = int(config.get("feature_dim", 768))
        hidden_dim = int(config.get("hidden_dim", 512))
        dropout = float(config.get("dropout", 0.1))
        self.recognition_aggregation = config.get("recognition_aggregation", "topk_mean")
        self.top_k = int(config.get("top_k", 2))

        self.chunk_projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.chunk_classifier = nn.Linear(hidden_dim, self.num_labels)
        self.detection_classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.detection_loss_fn = nn.BCEWithLogitsLoss()
        self.recognition_pos_weight = None

    def set_recognition_pos_weight(self, pos_weight):
        self.recognition_pos_weight = pos_weight

    def masked_mean(self, h, chunk_mask):
        mask = chunk_mask.unsqueeze(-1).type_as(h)
        return (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    def aggregate_recognition(self, chunk_logits, chunk_mask):
        mask = chunk_mask.unsqueeze(-1)
        masked_logits = chunk_logits.masked_fill(~mask, -1e9)
        if self.recognition_aggregation == "max":
            return masked_logits.max(dim=1).values
        if self.recognition_aggregation == "topk_mean":
            k = min(self.top_k, chunk_logits.shape[1])
            top_values = torch.topk(masked_logits, k=k, dim=1).values
            valid = top_values > -1e8
            return (top_values * valid.type_as(top_values)).sum(dim=1) / valid.sum(dim=1).clamp(min=1).type_as(top_values)
        if self.recognition_aggregation == "noisy_or":
            probs = torch.sigmoid(chunk_logits).masked_fill(~mask, 0.0)
            contract_probs = 1.0 - torch.prod(1.0 - probs.clamp(1e-6, 1 - 1e-6), dim=1)
            return torch.logit(contract_probs.clamp(1e-6, 1 - 1e-6))
        raise ValueError(f"Unsupported recognition_aggregation: {self.recognition_aggregation}")

    def top_chunk_indices(self, chunk_logits, chunk_mask, k=None):
        k = int(k or self.top_k)
        k = min(k, chunk_logits.shape[1])
        masked_logits = chunk_logits.masked_fill(~chunk_mask.unsqueeze(-1), -1e9)
        scores, indices = torch.topk(masked_logits, k=k, dim=1)
        return indices, scores

    def forward(self, chunk_features, chunk_mask, binary_label=None, multi_labels=None):
        chunk_mask = chunk_mask.bool()
        h = self.chunk_projection(chunk_features)
        chunk_logits = self.chunk_classifier(h)
        recognition_logits = self.aggregate_recognition(chunk_logits, chunk_mask)
        global_h = self.masked_mean(h, chunk_mask)
        detection_logits = self.detection_classifier(global_h).squeeze(-1)

        loss = None
        if binary_label is not None and multi_labels is not None:
            detection_loss = self.detection_loss_fn(detection_logits, binary_label.float())
            if self.recognition_pos_weight is None:
                recognition_loss = F.binary_cross_entropy_with_logits(
                    recognition_logits,
                    multi_labels.float(),
                )
            else:
                recognition_loss = F.binary_cross_entropy_with_logits(
                    recognition_logits,
                    multi_labels.float(),
                    pos_weight=self.recognition_pos_weight,
                )
            loss = (detection_loss + recognition_loss) / 2
        return {
            "loss": loss,
            "detection_logits": detection_logits,
            "recognition_logits": recognition_logits,
            "chunk_logits": chunk_logits,
        }

