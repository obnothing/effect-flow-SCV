import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertForMaskedLM


class EVMBertChunkCorrelaScan(nn.Module):
    def __init__(self, config, pad_token_id=None):
        super().__init__()
        self.num_labels = int(config["num_labels"])
        self.hf_model_path = config["hf_model_path"]
        self.chunk_pooling = config.get("chunk_pooling", "attention")
        self.use_vte = config.get("use_vte", False)
        self.vte_fusion = config.get("vte_fusion", "concat")
        self.encoder_chunk_batch_size = int(config.get("encoder_chunk_batch_size", 16))

        mlm_model = BertForMaskedLM.from_pretrained(
            self.hf_model_path,
            local_files_only=True,
        )
        self.encoder = mlm_model.bert
        if config.get("gradient_checkpointing", False) and hasattr(
            self.encoder,
            "gradient_checkpointing_enable",
        ):
            self.encoder.gradient_checkpointing_enable()
        if config.get("freeze_encoder", False):
            for param in self.encoder.parameters():
                param.requires_grad = False

        if pad_token_id is not None and self.encoder.config.pad_token_id != pad_token_id:
            raise ValueError(
                "EVM tokenizer pad_token_id does not match BertConfig.pad_token_id: "
                f"{pad_token_id} vs {self.encoder.config.pad_token_id}"
            )

        encoder_hidden_size = self.encoder.config.hidden_size
        bigru_hidden_size = int(config.get("bigru_hidden_size", 256))
        self.dropout = nn.Dropout(float(config.get("dropout", 0.1)))
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

        chunk_feature_size = bigru_hidden_size * 2
        if self.chunk_pooling == "attention":
            self.detection_chunk_attention = nn.Linear(chunk_feature_size, 1)
            self.recognition_chunk_attention = nn.Linear(chunk_feature_size, 1)
        elif self.chunk_pooling not in {"mean", "max"}:
            raise ValueError(f"Unsupported chunk_pooling: {self.chunk_pooling}")

        self.detection_classifier = nn.Linear(chunk_feature_size, 1)
        recognition_feature_size = chunk_feature_size
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

    def _masked_mean_pool_tokens(self, hidden_states, attention_mask):
        mask = attention_mask.unsqueeze(-1).type_as(hidden_states)
        summed = (hidden_states * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-9)
        return summed / denom

    def _encode_active_chunks(self, input_ids, attention_mask, token_type_ids):
        batch_size, max_chunks, chunk_size = input_ids.shape
        flat_input_ids = input_ids.view(batch_size * max_chunks, chunk_size)
        flat_attention_mask = attention_mask.view(batch_size * max_chunks, chunk_size)
        flat_token_type_ids = (
            token_type_ids.view(batch_size * max_chunks, chunk_size)
            if token_type_ids is not None
            else None
        )
        active_mask = flat_attention_mask.sum(dim=1) > 0
        active_indices = active_mask.nonzero(as_tuple=False).squeeze(-1)
        if active_indices.numel() == 0:
            raise ValueError("No active chunks found in batch.")

        active_input_ids = flat_input_ids.index_select(0, active_indices)
        active_attention_mask = flat_attention_mask.index_select(0, active_indices)
        active_token_type_ids = (
            flat_token_type_ids.index_select(0, active_indices)
            if flat_token_type_ids is not None
            else None
        )

        pooled_parts = []
        micro_batch = max(1, self.encoder_chunk_batch_size)
        for start in range(0, active_input_ids.size(0), micro_batch):
            end = start + micro_batch
            outputs = self.encoder(
                input_ids=active_input_ids[start:end],
                attention_mask=active_attention_mask[start:end],
                token_type_ids=(
                    active_token_type_ids[start:end]
                    if active_token_type_ids is not None
                    else None
                ),
            )
            pooled_parts.append(
                self._masked_mean_pool_tokens(
                    outputs.last_hidden_state,
                    active_attention_mask[start:end],
                )
            )

        active_embeddings = torch.cat(pooled_parts, dim=0)
        flat_embeddings = active_embeddings.new_zeros(
            batch_size * max_chunks,
            active_embeddings.size(-1),
        )
        flat_embeddings[active_indices] = active_embeddings
        return flat_embeddings.view(batch_size, max_chunks, -1)

    def _pool_chunks(self, chunk_features, chunk_mask, attention_layer=None):
        mask = chunk_mask.unsqueeze(-1).type_as(chunk_features)
        if self.chunk_pooling == "mean":
            summed = (chunk_features * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1e-9)
            return summed / denom
        if self.chunk_pooling == "max":
            return chunk_features.masked_fill(mask == 0, -1e9).max(dim=1).values

        scores = attention_layer(chunk_features).squeeze(-1)
        scores = scores.masked_fill(chunk_mask == 0, -1e4)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (chunk_features * weights).sum(dim=1)

    def _vulnerability_type_context(self, chunk_embeddings, chunk_mask):
        chunk_features = F.normalize(chunk_embeddings, p=2, dim=-1)
        label_features = F.normalize(self.vte_label_embeddings.weight, p=2, dim=-1)
        similarity = torch.matmul(chunk_features, label_features.transpose(0, 1))
        chunk_scores = similarity.max(dim=-1).values / max(self.vte_temperature, 1e-6)
        chunk_scores = chunk_scores.masked_fill(chunk_mask == 0, -1e4)
        weights = torch.softmax(chunk_scores, dim=1)
        context = torch.bmm(
            weights.unsqueeze(1),
            chunk_embeddings,
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
        token_type_ids=None,
        chunk_mask=None,
        binary_label=None,
        multi_labels=None,
    ):
        if input_ids.dim() != 3:
            raise ValueError(
                "EVMBertChunkCorrelaScan expects input_ids with shape "
                "[batch, max_chunks, chunk_size]."
            )
        if input_ids.max().item() >= self.encoder.config.vocab_size:
            raise ValueError(
                f"input_ids.max()={input_ids.max().item()} >= "
                f"vocab_size={self.encoder.config.vocab_size}"
            )
        if chunk_mask is None:
            chunk_mask = attention_mask.sum(dim=-1).gt(0).long()

        chunk_embeddings = self.dropout(
            self._encode_active_chunks(input_ids, attention_mask, token_type_ids)
        )
        detection_sequence, _ = self.detection_bigru(chunk_embeddings)
        recognition_sequence, _ = self.recognition_bigru(chunk_embeddings)

        detection_pooled = self._pool_chunks(
            detection_sequence,
            chunk_mask,
            getattr(self, "detection_chunk_attention", None),
        )
        recognition_pooled = self._pool_chunks(
            recognition_sequence,
            chunk_mask,
            getattr(self, "recognition_chunk_attention", None),
        )

        detection_logits = self.detection_classifier(detection_pooled).squeeze(-1)
        recognition_features = recognition_pooled
        if self.use_vte:
            vte_context = self._vulnerability_type_context(
                chunk_embeddings,
                chunk_mask,
            )
            recognition_features = torch.cat(
                [recognition_pooled, self.vte_dropout(vte_context)],
                dim=-1,
            )
        recognition_logits = self.recognition_classifier(recognition_features)

        loss = None
        if binary_label is not None and multi_labels is not None:
            detection_loss = self.detection_loss_fn(
                detection_logits,
                binary_label.float(),
            )
            if self.recognition_pos_weight is None:
                recognition_loss = self.recognition_loss_fn(
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
        }
