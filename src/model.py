import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class CorrelaScan(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_labels = config["num_labels"]
        self.encoder = AutoModel.from_pretrained(
            config["model_name"],
            local_files_only=config.get("local_files_only", True),
        )
        if config.get("freeze_encoder", False):
            for param in self.encoder.parameters():
                param.requires_grad = False

        encoder_hidden_size = self.encoder.config.hidden_size
        bigru_hidden_size = config["bigru_hidden_size"]
        self.use_vte = config.get("use_vte", False)
        self.vte_fusion = config.get("vte_fusion", "concat")

        self.detection_bigru = nn.GRU(
            input_size=encoder_hidden_size,
            hidden_size=bigru_hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.recognition_bigru = nn.GRU(
            input_size=encoder_hidden_size,
            hidden_size=bigru_hidden_size,
            batch_first=True,
            bidirectional=True,
        )

        self.detection_classifier = nn.Linear(bigru_hidden_size * 2, 1)
        recognition_feature_size = bigru_hidden_size * 2
        if self.use_vte:
            if self.vte_fusion != "concat":
                raise ValueError("Only vte_fusion=concat is currently supported.")
            self.vte_label_embeddings = nn.Embedding(
                self.num_labels,
                encoder_hidden_size,
            )
            self.vte_layer_norm = nn.LayerNorm(encoder_hidden_size)
            self.vte_dropout = nn.Dropout(float(config.get("vte_dropout", 0.1)))
            self.vte_temperature = float(config.get("vte_temperature", 1.0))
            recognition_feature_size += encoder_hidden_size

        self.recognition_classifier = nn.Linear(
            recognition_feature_size,
            self.num_labels,
        )

        self.detection_loss_fn = nn.BCEWithLogitsLoss()
        self.recognition_loss_fn = nn.BCEWithLogitsLoss()
        self.recognition_pos_weight = None

    def set_recognition_pos_weight(self, pos_weight):
        self.recognition_pos_weight = pos_weight

    def _masked_mean_pool(self, sequence_output, attention_mask):
        mask = attention_mask.unsqueeze(-1).type_as(sequence_output)
        summed = (sequence_output * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-9)
        return summed / denom

    def _vulnerability_type_context(self, sequence_output, attention_mask):
        token_features = F.normalize(sequence_output, p=2, dim=-1)
        label_features = F.normalize(self.vte_label_embeddings.weight, p=2, dim=-1)
        similarity = torch.matmul(token_features, label_features.transpose(0, 1))
        token_scores = similarity.max(dim=-1).values / max(self.vte_temperature, 1e-6)
        token_scores = token_scores.masked_fill(attention_mask == 0, -1e4)
        attention_weights = torch.softmax(token_scores, dim=1)
        context = torch.bmm(
            attention_weights.unsqueeze(1),
            sequence_output,
        ).squeeze(1)
        return self.vte_layer_norm(context)

    def get_vte_label_correlation(self):
        if not self.use_vte:
            return None
        label_features = F.normalize(self.vte_label_embeddings.weight.detach(), p=2, dim=-1)
        return torch.matmul(label_features, label_features.transpose(0, 1))

    def forward(
        self,
        input_ids,
        attention_mask,
        binary_label=None,
        multi_labels=None,
    ):
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        last_hidden_state = encoder_outputs.last_hidden_state

        detection_output, _ = self.detection_bigru(last_hidden_state)
        recognition_output, _ = self.recognition_bigru(last_hidden_state)

        detection_pooled = self._masked_mean_pool(detection_output, attention_mask)
        recognition_pooled = self._masked_mean_pool(recognition_output, attention_mask)

        detection_logits = self.detection_classifier(detection_pooled).squeeze(-1)
        recognition_features = recognition_pooled
        if self.use_vte:
            vte_context = self._vulnerability_type_context(
                last_hidden_state,
                attention_mask,
            )
            recognition_features = torch.cat(
                [recognition_pooled, self.vte_dropout(vte_context)],
                dim=-1,
            )
        recognition_logits = self.recognition_classifier(recognition_features)

        loss = None
        if binary_label is not None and multi_labels is not None:
            detection_loss = self.detection_loss_fn(
                detection_logits, binary_label.float()
            )
            if self.recognition_pos_weight is None:
                recognition_loss = self.recognition_loss_fn(
                    recognition_logits, multi_labels.float()
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
        }
