import torch
import torch.nn as nn
import torch.nn.functional as F


class ChunkContextEncoder(nn.Module):
    def __init__(
        self,
        feature_dim=768,
        hidden_dim=512,
        max_chunks=32,
        num_layers=2,
        num_heads=8,
        dropout=0.1,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.max_chunks = int(max_chunks)
        self.input_norm = nn.LayerNorm(feature_dim)
        self.input_projection = nn.Linear(feature_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.position_embedding = nn.Parameter(
            torch.empty(self.max_chunks, hidden_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        # PyTorch 2.0.x can fail in the fused eval-only Transformer path on
        # some CUDA 11.8/A10 combinations. The regular path is stable and uses
        # the same model parameters and masks.
        if hasattr(torch.backends, "mha") and hasattr(
            torch.backends.mha,
            "set_fastpath_enabled",
        ):
            torch.backends.mha.set_fastpath_enabled(False)
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def forward(self, chunk_features, chunk_mask):
        # DataParallel executes replicas in worker threads. In PyTorch 2.0.x
        # the MHA fastpath flag is thread-local, so disable it in every replica
        # before entering the Transformer rather than only during __init__.
        if hasattr(torch.backends, "mha") and hasattr(
            torch.backends.mha,
            "set_fastpath_enabled",
        ):
            torch.backends.mha.set_fastpath_enabled(False)
        if chunk_features.ndim != 3:
            raise ValueError(
                f"chunk_features must be [B, C, H], got {tuple(chunk_features.shape)}"
            )
        chunk_mask = chunk_mask.bool()
        if chunk_mask.shape != chunk_features.shape[:2]:
            raise ValueError(
                "chunk_mask shape must match the first two chunk_features dimensions"
            )
        if chunk_features.shape[1] > self.max_chunks:
            raise ValueError(
                f"received {chunk_features.shape[1]} chunks, max_chunks={self.max_chunks}"
            )
        if (~chunk_mask).all(dim=1).any():
            raise ValueError("each sample must contain at least one valid chunk")

        h = self.input_norm(chunk_features)
        h = self.input_projection(h)
        h = self.activation(h)
        h = self.dropout(h)
        positions = self.position_embedding[: h.shape[1]].unsqueeze(0)
        h = h + positions.type_as(h)
        padding_mask = (~chunk_mask).contiguous().bool()
        h = self.transformer(h, src_key_padding_mask=padding_mask)
        return h.masked_fill(~chunk_mask.unsqueeze(-1), 0.0)


class EVMChunkMILClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_labels = int(config.get("num_labels", 10))
        feature_dim = int(config.get("feature_dim", 768))
        hidden_dim = int(config.get("hidden_dim", 512))
        attn_dim = int(config.get("attn_dim", 256))
        dropout = float(config.get("dropout", 0.1))
        self.recognition_aggregation = config.get("recognition_aggregation", "topk_mean")
        self.top_k = int(config.get("top_k", 2))
        self.use_chunk_context = bool(config.get("use_chunk_context", False))

        if self.use_chunk_context:
            self.chunk_context_encoder = ChunkContextEncoder(
                feature_dim=feature_dim,
                hidden_dim=hidden_dim,
                max_chunks=int(config.get("max_chunks", 32)),
                num_layers=int(config.get("chunk_context_num_layers", 2)),
                num_heads=int(config.get("chunk_context_num_heads", 8)),
                dropout=float(config.get("chunk_context_dropout", dropout)),
            )
            self.chunk_projection = None
        else:
            self.chunk_context_encoder = None
            self.chunk_projection = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        self.chunk_classifier = nn.Linear(hidden_dim, self.num_labels)
        self.attn_v = nn.Linear(hidden_dim, attn_dim)
        self.attn_u = nn.Linear(hidden_dim, attn_dim)
        self.label_attn = nn.Parameter(torch.empty(self.num_labels, attn_dim))
        self.label_out = nn.Parameter(torch.empty(self.num_labels, hidden_dim))
        self.label_bias = nn.Parameter(torch.zeros(self.num_labels))
        self.detection_classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.detection_loss_fn = nn.BCEWithLogitsLoss()
        self.register_buffer("recognition_pos_weight", None, persistent=False)
        nn.init.xavier_uniform_(self.label_attn)
        nn.init.xavier_uniform_(self.label_out)

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

    def label_gated_attention(self, h, chunk_mask):
        v = torch.tanh(self.attn_v(h))
        u = torch.sigmoid(self.attn_u(h))
        gated = v * u
        attn_logits = torch.einsum("bca,ka->bck", gated, self.label_attn)
        attn_logits = attn_logits.masked_fill(~chunk_mask.unsqueeze(-1), -1e9)
        attn_weights = torch.softmax(attn_logits, dim=1)
        z = torch.einsum("bck,bch->bkh", attn_weights, h)
        recognition_logits = (z * self.label_out.unsqueeze(0)).sum(dim=-1) + self.label_bias
        return recognition_logits, attn_weights, attn_logits

    def top_chunk_indices(self, chunk_logits, chunk_mask, k=None):
        k = int(k or self.top_k)
        k = min(k, chunk_logits.shape[1])
        masked_logits = chunk_logits.masked_fill(~chunk_mask.unsqueeze(-1), -1e9)
        scores, indices = torch.topk(masked_logits, k=k, dim=1)
        return indices, scores

    def forward(self, chunk_features, chunk_mask, binary_label=None, multi_labels=None):
        chunk_mask = chunk_mask.bool()
        if self.use_chunk_context:
            h = self.chunk_context_encoder(chunk_features, chunk_mask)
        else:
            h = self.chunk_projection(chunk_features)
        chunk_logits = self.chunk_classifier(h)
        chunk_scores = None
        if self.recognition_aggregation == "label_gated_attention":
            recognition_logits, chunk_scores, chunk_logits = self.label_gated_attention(
                h,
                chunk_mask,
            )
        else:
            recognition_logits = self.aggregate_recognition(chunk_logits, chunk_mask)
            chunk_scores = torch.sigmoid(chunk_logits).masked_fill(
                ~chunk_mask.unsqueeze(-1),
                0.0,
            )
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
            loss = ((detection_loss + recognition_loss) / 2).reshape(1)
        return {
            "loss": loss,
            "detection_logits": detection_logits,
            "recognition_logits": recognition_logits,
            "chunk_logits": chunk_logits,
            "chunk_scores": chunk_scores,
        }


class EffectFlowGuidedChunkMIL(EVMChunkMILClassifier):
    """Chunk-context MIL with explicit effect-flow semantic evidence.

    The model keeps contract-level MIL supervision: effect-flow probabilities are
    auxiliary chunk evidence, not chunk-level vulnerability labels.
    """

    def __init__(self, config):
        super().__init__(config)
        hidden_dim = int(config.get("hidden_dim", 512))
        efpp_dim = int(config.get("efpp_dim", 22))
        etp_dim = int(config.get("etp_dim", 16))
        semantic_hidden_dim = int(config.get("semantic_projection_dim", 128))
        dropout = float(config.get("dropout", 0.1))
        self.semantic_input_dim = efpp_dim + etp_dim
        self.semantic_projection = nn.Sequential(
            nn.LayerNorm(self.semantic_input_dim),
            nn.Linear(self.semantic_input_dim, semantic_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(semantic_hidden_dim, hidden_dim),
        )
        self.semantic_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.semantic_fusion = config.get("semantic_fusion", "gated_add")
        if self.semantic_fusion != "gated_add":
            raise ValueError(
                "EffectFlowGuidedChunkMIL currently supports semantic_fusion=gated_add only"
            )

    def fuse_semantics(self, h, chunk_mask, efpp_probs, etp_distribution):
        if efpp_probs is None or etp_distribution is None:
            raise ValueError(
                "EffectFlowGuidedChunkMIL requires efpp_probs and etp_distribution"
            )
        if efpp_probs.shape[:2] != h.shape[:2]:
            raise ValueError("efpp_probs [B, C] prefix must match chunk features")
        if etp_distribution.shape[:2] != h.shape[:2]:
            raise ValueError("etp_distribution [B, C] prefix must match chunk features")
        semantic_input = torch.cat(
            [efpp_probs.to(dtype=h.dtype), etp_distribution.to(dtype=h.dtype)],
            dim=-1,
        )
        if semantic_input.shape[-1] != self.semantic_input_dim:
            raise ValueError(
                f"semantic feature dim mismatch: got {semantic_input.shape[-1]}, "
                f"expected {self.semantic_input_dim}"
            )
        semantic_h = self.semantic_projection(semantic_input)
        gate = self.semantic_gate(torch.cat([h, semantic_h], dim=-1))
        fused = h + gate * semantic_h
        return fused.masked_fill(~chunk_mask.unsqueeze(-1), 0.0), gate

    def forward(
        self,
        chunk_features,
        chunk_mask,
        efpp_probs=None,
        etp_distribution=None,
        binary_label=None,
        multi_labels=None,
    ):
        chunk_mask = chunk_mask.bool()
        if self.use_chunk_context:
            h = self.chunk_context_encoder(chunk_features, chunk_mask)
        else:
            h = self.chunk_projection(chunk_features)
        h, semantic_gate = self.fuse_semantics(
            h,
            chunk_mask,
            efpp_probs,
            etp_distribution,
        )
        chunk_logits = self.chunk_classifier(h)
        chunk_scores = None
        if self.recognition_aggregation == "label_gated_attention":
            recognition_logits, chunk_scores, chunk_logits = self.label_gated_attention(
                h,
                chunk_mask,
            )
        else:
            recognition_logits = self.aggregate_recognition(chunk_logits, chunk_mask)
            chunk_scores = torch.sigmoid(chunk_logits).masked_fill(
                ~chunk_mask.unsqueeze(-1),
                0.0,
            )
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
            loss = ((detection_loss + recognition_loss) / 2).reshape(1)
        return {
            "loss": loss,
            "detection_logits": detection_logits,
            "recognition_logits": recognition_logits,
            "chunk_logits": chunk_logits,
            "chunk_scores": chunk_scores,
            "semantic_gate": semantic_gate,
        }
