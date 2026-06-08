import torch
from torch import nn


class EVMChunkCorrelaScan(nn.Module):
    def __init__(self, config, vocab_size, pad_token_id):
        super().__init__()
        self.num_labels = config["num_labels"]
        self.encoder_type = config.get("encoder_type", "bigru")
        self.chunk_pooling = config.get("chunk_pooling", "attention")

        self.embedding = nn.Embedding(
            vocab_size,
            config["embedding_dim"],
            padding_idx=pad_token_id,
        )
        self.dropout = nn.Dropout(config.get("dropout", 0.1))

        if self.encoder_type == "bigru":
            self.chunk_encoder = nn.GRU(
                input_size=config["embedding_dim"],
                hidden_size=config["bigru_hidden_size"],
                batch_first=True,
                bidirectional=True,
            )
            chunk_dim = config["bigru_hidden_size"] * 2
        elif self.encoder_type == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config["embedding_dim"],
                nhead=config["num_attention_heads"],
                dim_feedforward=config["embedding_dim"] * 4,
                dropout=config.get("dropout", 0.1),
                batch_first=True,
            )
            self.chunk_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=config["num_transformer_layers"],
            )
            chunk_dim = config["embedding_dim"]
        else:
            raise ValueError(f"Unsupported encoder_type: {self.encoder_type}")

        if self.chunk_pooling == "attention":
            self.chunk_attention = nn.Linear(chunk_dim, 1)
        elif self.chunk_pooling not in {"mean", "max"}:
            raise ValueError(f"Unsupported chunk_pooling: {self.chunk_pooling}")

        self.detection_classifier = nn.Linear(chunk_dim, 1)
        self.recognition_classifier = nn.Linear(chunk_dim, self.num_labels)
        self.detection_loss_fn = nn.BCEWithLogitsLoss()
        self.recognition_loss_fn = nn.BCEWithLogitsLoss()

    def _masked_mean_pool(self, hidden_states, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        summed = (hidden_states * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return summed / denom

    def _encode_chunks(self, input_ids, attention_mask):
        batch_size, max_chunks, chunk_size = input_ids.shape
        flat_input_ids = input_ids.view(batch_size * max_chunks, chunk_size)
        flat_attention_mask = attention_mask.view(batch_size * max_chunks, chunk_size)
        embedded = self.dropout(self.embedding(flat_input_ids))

        if self.encoder_type == "bigru":
            encoded, _ = self.chunk_encoder(embedded)
        else:
            key_padding_mask = flat_attention_mask == 0
            all_pad = key_padding_mask.all(dim=1)
            if all_pad.any():
                key_padding_mask[all_pad, 0] = False
            encoded = self.chunk_encoder(
                embedded,
                src_key_padding_mask=key_padding_mask,
            )

        chunk_embeddings = self._masked_mean_pool(encoded, flat_attention_mask)
        return chunk_embeddings.view(batch_size, max_chunks, -1)

    def _pool_chunks(self, chunk_embeddings, chunk_mask):
        mask = chunk_mask.unsqueeze(-1).float()
        if self.chunk_pooling == "mean":
            summed = (chunk_embeddings * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1.0)
            return summed / denom
        if self.chunk_pooling == "max":
            return chunk_embeddings.masked_fill(mask == 0, -1e9).max(dim=1).values

        scores = self.chunk_attention(chunk_embeddings).squeeze(-1)
        scores = scores.masked_fill(chunk_mask == 0, -1e9)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (chunk_embeddings * weights).sum(dim=1)

    def forward(
        self,
        input_ids,
        attention_mask,
        chunk_mask,
        binary_label=None,
        multi_labels=None,
    ):
        chunk_embeddings = self._encode_chunks(input_ids, attention_mask)
        contract_embedding = self.dropout(
            self._pool_chunks(chunk_embeddings, chunk_mask)
        )
        detection_logits = self.detection_classifier(contract_embedding).squeeze(-1)
        recognition_logits = self.recognition_classifier(contract_embedding)

        outputs = {
            "detection_logits": detection_logits,
            "recognition_logits": recognition_logits,
        }
        if binary_label is not None and multi_labels is not None:
            detection_loss = self.detection_loss_fn(detection_logits, binary_label)
            recognition_loss = self.recognition_loss_fn(
                recognition_logits, multi_labels
            )
            outputs["loss"] = (detection_loss + recognition_loss) / 2
        return outputs
