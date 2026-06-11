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
        self.recognition_classifier = nn.Linear(bigru_hidden_size * 2, self.num_labels)

        self.detection_loss_fn = nn.BCEWithLogitsLoss()
        self.recognition_loss_fn = nn.BCEWithLogitsLoss()
        self.recognition_pos_weight = None

        # TODO: add vulnerability type embedding module in stage 2.

    def set_recognition_pos_weight(self, pos_weight):
        self.recognition_pos_weight = pos_weight

    def _masked_mean_pool(self, sequence_output, attention_mask):
        mask = attention_mask.unsqueeze(-1).type_as(sequence_output)
        summed = (sequence_output * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-9)
        return summed / denom

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
        recognition_logits = self.recognition_classifier(recognition_pooled)

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
