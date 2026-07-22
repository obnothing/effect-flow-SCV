import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import json
import yaml

from effect_flow_ontology import (
    build_template_tensor_bundle,
    load_ontology,
    load_vulnerability_templates,
)
from effect_flow_utils import GLOBAL_VULNERABILITY_LABELS


def softplus_inverse(value):
    value = torch.tensor(float(value), dtype=torch.float32)
    return float(torch.log(torch.expm1(value)).item())


def asymmetric_multilabel_loss(
    logits,
    targets,
    gamma_neg=4.0,
    gamma_pos=0.0,
    clip=0.05,
    eps=1e-8,
):
    targets = targets.float()
    probs_pos = torch.sigmoid(logits)
    probs_neg = 1.0 - probs_pos
    if clip and clip > 0:
        probs_neg = (probs_neg + float(clip)).clamp(max=1.0)

    log_pos = torch.log(probs_pos.clamp(min=float(eps)))
    log_neg = torch.log(probs_neg.clamp(min=float(eps)))
    loss = targets * log_pos + (1.0 - targets) * log_neg

    gamma_neg = float(gamma_neg)
    gamma_pos = float(gamma_pos)
    if gamma_neg > 0 or gamma_pos > 0:
        pt = probs_pos * targets + probs_neg * (1.0 - targets)
        gamma = gamma_pos * targets + gamma_neg * (1.0 - targets)
        loss = loss * torch.pow((1.0 - pt).clamp(min=0.0), gamma)
    return -loss.mean()


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


class ClassicPreNormTransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, h, padding_mask):
        normalized = self.attn_norm(h)
        attended, _ = self.self_attn(
            normalized,
            normalized,
            normalized,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        h = h + self.attn_dropout(attended)
        return h + self.ffn(self.ffn_norm(h))


class ClassicLocalGlobalContextEncoder(nn.Module):
    """Shallow local-convolution plus global pre-norm Transformer encoder."""

    def __init__(
        self,
        feature_dim=768,
        hidden_dim=512,
        max_chunks=64,
        num_layers=2,
        num_heads=8,
        dropout=0.1,
        local_residual_enabled=True,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.max_chunks = int(max_chunks)
        self.local_residual_enabled = bool(local_residual_enabled)
        self.input_norm = nn.LayerNorm(feature_dim)
        self.input_projection = nn.Linear(feature_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.position_embedding = nn.Parameter(
            torch.empty(self.max_chunks, hidden_dim)
        )
        self.local_norm = nn.LayerNorm(hidden_dim)
        self.local_depthwise = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
            groups=hidden_dim,
        )
        self.local_pointwise = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.local_activation = nn.GELU()
        self.local_dropout = nn.Dropout(dropout)
        self.transformer = nn.ModuleList(
            [
                ClassicPreNormTransformerBlock(hidden_dim, num_heads, dropout)
                for _ in range(int(num_layers))
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def forward(self, chunk_features, chunk_mask):
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

        h = self.dropout(self.activation(self.input_projection(self.input_norm(chunk_features))))
        h = h + self.position_embedding[: h.shape[1]].unsqueeze(0).type_as(h)
        h = h.masked_fill(~chunk_mask.unsqueeze(-1), 0.0)
        if self.local_residual_enabled:
            local = self.local_norm(h).transpose(1, 2)
            local = self.local_depthwise(local)
            local = self.local_pointwise(local).transpose(1, 2)
            local = self.local_dropout(self.local_activation(local))
            h = (h + local).masked_fill(~chunk_mask.unsqueeze(-1), 0.0)
        padding_mask = (~chunk_mask).contiguous()
        for layer in self.transformer:
            h = layer(h, padding_mask)
            h = h.masked_fill(~chunk_mask.unsqueeze(-1), 0.0)
        return self.output_norm(h).masked_fill(~chunk_mask.unsqueeze(-1), 0.0)


class ClassicEscortLabelBranch(nn.Module):
    def __init__(
        self,
        attn_dim,
        adapter_dim,
        bottleneck_dim,
        dropout,
        beta_init,
        gamma_init,
    ):
        super().__init__()
        self.attention_query = nn.Parameter(torch.empty(attn_dim))
        self.residual_norm = nn.LayerNorm(adapter_dim)
        self.down_projection = nn.Linear(adapter_dim, bottleneck_dim)
        self.activation = nn.GELU()
        self.down_dropout = nn.Dropout(dropout)
        self.up_projection = nn.Linear(bottleneck_dim, adapter_dim)
        self.up_dropout = nn.Dropout(min(float(dropout), 0.1))
        self.output_norm = nn.LayerNorm(adapter_dim)
        self.classifier = nn.Linear(adapter_dim, 1)
        self.beta_reliable_raw = nn.Parameter(torch.tensor(softplus_inverse(beta_init)))
        self.gamma_reliable_raw = nn.Parameter(torch.tensor(softplus_inverse(gamma_init)))
        self.evidence_logit_weight = nn.Parameter(torch.zeros(2))
        self.evidence_logit_bias = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.attention_query, mean=0.0, std=0.02)
        nn.init.zeros_(self.up_projection.weight)
        nn.init.zeros_(self.up_projection.bias)

    def forward(self, adapted):
        update = self.down_projection(self.residual_norm(adapted))
        update = self.down_dropout(self.activation(update))
        update = self.up_dropout(self.up_projection(update))
        return self.classifier(self.output_norm(adapted + update)).squeeze(-1)


class EVMChunkMILClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_labels = int(config.get("num_labels", 10))
        feature_dim = int(config.get("feature_dim", 768))
        hidden_dim = int(config.get("hidden_dim", 512))
        attn_dim = int(config.get("attn_dim", 256))
        dropout = float(config.get("dropout", 0.1))
        self.hidden_dim = hidden_dim
        self.label_names = config.get(
            "label_names",
            [f"label_{idx}" for idx in range(self.num_labels)],
        )
        self.loss_label_names = list(config.get("loss_label_names", self.label_names))
        unknown_loss_labels = [
            name for name in self.loss_label_names if name not in self.label_names
        ]
        if unknown_loss_labels:
            raise ValueError(f"Unknown loss_label_names: {unknown_loss_labels}")
        self.loss_label_ids = [
            self.label_names.index(name) for name in self.loss_label_names
        ]
        self.recognition_aggregation = config.get("recognition_aggregation", "topk_mean")
        self.top_k = int(config.get("top_k", 2))
        self.use_chunk_context = bool(config.get("use_chunk_context", False))
        self.task_mode = str(config.get("task_mode", "joint_detection_recognition"))
        self.derived_detection_from_multilabel = bool(
            config.get("derived_detection_from_multilabel", False)
        )
        self.detection_head_enabled = bool(
            config.get(
                "detection_head_enabled",
                self.task_mode != "recognition_only",
            )
        )
        self.detection_loss_weight = float(
            config.get(
                "detection_loss_weight",
                0.0 if self.task_mode == "recognition_only" else 1.0,
            )
        )
        self.recognition_loss_weight = float(
            config.get("recognition_loss_weight", 1.0)
        )
        if self.task_mode == "recognition_only" and self.detection_loss_weight != 0.0:
            raise ValueError(
                "task_mode=recognition_only requires detection_loss_weight=0.0"
            )
        if self.recognition_loss_weight <= 0:
            raise ValueError("recognition_loss_weight must be positive")
        if self.detection_loss_weight < 0:
            raise ValueError("detection_loss_weight must be non-negative")
        self.recognition_loss_type = config.get("recognition_loss_type", "bce")
        if self.recognition_loss_type not in {"bce", "asl"}:
            raise ValueError(
                f"Unsupported recognition_loss_type: {self.recognition_loss_type}"
            )
        self.asl_gamma_neg = float(config.get("asl_gamma_neg", 4.0))
        self.asl_gamma_pos = float(config.get("asl_gamma_pos", 0.0))
        self.asl_clip = float(config.get("asl_clip", 0.05))
        self.asl_eps = float(config.get("asl_eps", 1e-8))
        self.rare_negative_subsampling_enabled = bool(
            config.get("rare_negative_subsampling_enabled", False)
        )
        self.rare_negative_subsampling_labels = list(
            config.get(
                "rare_negative_subsampling_labels",
                ["Front Running", "Bad Randomness"],
            )
        )
        self.rare_negative_subsampling_label_ids = [
            self.label_names.index(name)
            for name in self.rare_negative_subsampling_labels
            if name in self.label_names
        ]
        self.rare_negative_subsampling_policy = str(
            config.get("rare_negative_subsampling_policy", "random_fraction")
        )
        self.rare_negative_subsampling_negative_fraction = float(
            config.get("rare_negative_subsampling_negative_fraction", 1.0)
        )
        self.rare_negative_subsampling_neg_per_pos = float(
            config.get("rare_negative_subsampling_neg_per_pos", 5.0)
        )
        self.rare_negative_subsampling_skip_when_no_positive = bool(
            config.get("rare_negative_subsampling_skip_when_no_positive", True)
        )
        if self.rare_negative_subsampling_enabled:
            if self.recognition_loss_type != "bce":
                raise ValueError(
                    "rare_negative_subsampling_enabled requires recognition_loss_type=bce"
                )
            if not self.rare_negative_subsampling_label_ids:
                raise ValueError(
                    "rare_negative_subsampling_enabled=true requires at least one "
                    "rare_negative_subsampling_label present in label_names"
                )
            if self.rare_negative_subsampling_policy not in {
                "random_fraction",
                "random_pos_ratio",
            }:
                raise ValueError(
                    "rare_negative_subsampling_policy must be one of: "
                    "random_fraction, random_pos_ratio"
                )
            if not (0.0 < self.rare_negative_subsampling_negative_fraction <= 1.0):
                raise ValueError(
                    "rare_negative_subsampling_negative_fraction must be in (0, 1]"
                )
            if self.rare_negative_subsampling_neg_per_pos <= 0:
                raise ValueError(
                    "rare_negative_subsampling_neg_per_pos must be positive"
                )

        if self.use_chunk_context:
            encoder_type = str(
                config.get("chunk_context_encoder_type", "transformer")
            )
            encoder_kwargs = {
                "feature_dim": feature_dim,
                "hidden_dim": hidden_dim,
                "max_chunks": int(config.get("max_chunks", 32)),
                "num_layers": int(config.get("chunk_context_num_layers", 2)),
                "num_heads": int(config.get("chunk_context_num_heads", 8)),
                "dropout": float(config.get("chunk_context_dropout", dropout)),
            }
            if encoder_type == "transformer":
                self.chunk_context_encoder = ChunkContextEncoder(**encoder_kwargs)
            elif encoder_type == "classic_local_global":
                self.chunk_context_encoder = ClassicLocalGlobalContextEncoder(
                    **encoder_kwargs,
                    local_residual_enabled=bool(
                        config.get("classic_local_residual_enabled", True)
                    ),
                )
            else:
                raise ValueError(
                    "chunk_context_encoder_type must be transformer or "
                    "classic_local_global"
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
        self.recognition_head_type = str(
            config.get("recognition_head_type", "linear_label_dot")
        )
        if self.recognition_head_type not in {
            "linear_label_dot",
            "label_branch_mlp",
            "classic_escort_residual",
        }:
            raise ValueError(
                "recognition_head_type must be one of: "
                "linear_label_dot, label_branch_mlp, classic_escort_residual"
            )
        self.label_branches = None
        if self.recognition_head_type == "label_branch_mlp":
            branch_hidden_dim = int(config.get("label_branch_hidden_dim", hidden_dim // 2))
            branch_dropout = float(config.get("label_branch_dropout", dropout))
            branch_use_layernorm = bool(config.get("label_branch_use_layernorm", True))
            if branch_hidden_dim <= 0:
                raise ValueError("label_branch_hidden_dim must be positive")
            branches = []
            for _ in range(self.num_labels):
                layers = []
                if branch_use_layernorm:
                    layers.append(nn.LayerNorm(hidden_dim))
                layers.extend(
                    [
                        nn.Linear(hidden_dim, branch_hidden_dim),
                        nn.GELU(),
                        nn.Dropout(branch_dropout),
                        nn.Linear(branch_hidden_dim, 1),
                    ]
                )
                branches.append(nn.Sequential(*layers))
            self.label_branches = nn.ModuleList(branches)
        self.classic_label_adapter = None
        self.classic_label_branches = None
        if self.recognition_head_type == "classic_escort_residual":
            adapter_dim = int(config.get("classic_label_adapter_dim", 256))
            major_bottleneck = int(
                config.get("classic_major_branch_bottleneck_dim", 64)
            )
            rare_bottleneck = int(
                config.get("classic_rare_branch_bottleneck_dim", 32)
            )
            rare_labels = set(
                config.get(
                    "classic_rare_label_names",
                    ["Front Running", "Bad Randomness"],
                )
            )
            self.classic_label_adapter = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, adapter_dim),
                nn.GELU(),
                nn.Dropout(float(config.get("classic_label_adapter_dropout", 0.1))),
            )
            beta_init = float(config.get("beta_reliable_init", 0.125))
            gamma_init = float(config.get("gamma_reliable_init", 0.125))
            branches = []
            for label_name in self.label_names:
                is_rare = label_name in rare_labels
                branches.append(
                    ClassicEscortLabelBranch(
                        attn_dim=attn_dim,
                        adapter_dim=adapter_dim,
                        bottleneck_dim=(
                            rare_bottleneck if is_rare else major_bottleneck
                        ),
                        dropout=float(
                            config.get(
                                "classic_rare_branch_dropout" if is_rare
                                else "classic_major_branch_dropout",
                                0.2 if is_rare else 0.1,
                            )
                        ),
                        beta_init=beta_init,
                        gamma_init=gamma_init,
                    )
                )
            self.classic_label_branches = nn.ModuleList(branches)
            for legacy_parameter in (
                self.label_attn,
                self.label_out,
                self.label_bias,
                self.chunk_classifier.weight,
                self.chunk_classifier.bias,
            ):
                legacy_parameter.requires_grad = False
        self.detection_classifier = None
        if self.detection_head_enabled:
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

    def compute_weighted_task_loss(self, detection_loss, recognition_loss):
        weighted_loss = self.recognition_loss_weight * recognition_loss
        if detection_loss is not None and self.detection_loss_weight > 0:
            weighted_loss = weighted_loss + self.detection_loss_weight * detection_loss
            total_weight = self.recognition_loss_weight + self.detection_loss_weight
            return weighted_loss / total_weight
        return weighted_loss

    def compute_detection_logits(self, global_h, recognition_logits):
        if self.detection_classifier is not None:
            return self.detection_classifier(global_h).squeeze(-1)
        return recognition_logits.max(dim=1).values

    def compute_label_logits(self, z):
        if self.recognition_head_type == "classic_escort_residual":
            adapted = self.classic_label_adapter(z)
            logits = [
                branch(adapted[:, label_idx, :])
                for label_idx, branch in enumerate(self.classic_label_branches)
            ]
            return torch.stack(logits, dim=1)
        if self.recognition_head_type == "label_branch_mlp":
            logits = [
                branch(z[:, label_idx, :]).squeeze(-1)
                for label_idx, branch in enumerate(self.label_branches)
            ]
            return torch.stack(logits, dim=1)
        return (z * self.label_out.unsqueeze(0)).sum(dim=-1) + self.label_bias

    def _empty_rare_negative_subsampling_stats(self, device):
        size = len(self.rare_negative_subsampling_label_ids)
        zero = torch.zeros(size, device=device, dtype=torch.float32)
        return {
            "enabled": bool(self.rare_negative_subsampling_enabled),
            "label_names": [
                self.label_names[idx] for idx in self.rare_negative_subsampling_label_ids
            ],
            "label_ids": list(self.rare_negative_subsampling_label_ids),
            "positive_count": zero.clone(),
            "available_negative_count": zero.clone(),
            "selected_negative_count": zero.clone(),
            "effective_neg_per_pos": zero.clone(),
            "selected_negative_fraction": zero.clone(),
            "active_label_batches": zero.clone(),
            "zero_positive_label_batches": zero.clone(),
        }

    def _build_rare_negative_subsampling_mask(self, multi_labels):
        labels = multi_labels.float()
        device = labels.device
        mask = torch.ones_like(labels, dtype=torch.bool, device=device)
        stats = self._empty_rare_negative_subsampling_stats(device)
        if not self.rare_negative_subsampling_enabled:
            return mask, stats

        for row_idx, label_id in enumerate(self.rare_negative_subsampling_label_ids):
            targets = labels[:, label_id] > 0.5
            negatives = ~targets
            positive_count = targets.float().sum()
            negative_count = negatives.float().sum()
            stats["positive_count"][row_idx] = positive_count
            stats["available_negative_count"][row_idx] = negative_count

            has_positive = bool((positive_count > 0).detach().cpu().item())
            if has_positive:
                stats["active_label_batches"][row_idx] = 1.0
            else:
                stats["zero_positive_label_batches"][row_idx] = 1.0
                if self.rare_negative_subsampling_skip_when_no_positive:
                    mask[:, label_id] = False
                    continue

            selected_negatives = torch.zeros_like(negatives)
            negative_indices = torch.nonzero(negatives, as_tuple=False).flatten()
            negative_count_int = int(negative_indices.numel())
            if negative_count_int > 0:
                if self.rare_negative_subsampling_policy == "random_fraction":
                    target_negative_count = int(
                        torch.ceil(
                            negative_count
                            * self.rare_negative_subsampling_negative_fraction
                        )
                        .clamp(min=0)
                        .detach()
                        .cpu()
                        .item()
                    )
                elif self.rare_negative_subsampling_policy == "random_pos_ratio":
                    target_negative_count = int(
                        torch.ceil(
                            positive_count
                            * self.rare_negative_subsampling_neg_per_pos
                        )
                        .clamp(min=0)
                        .detach()
                        .cpu()
                        .item()
                    )
                else:
                    raise ValueError(
                        "Unsupported rare_negative_subsampling_policy: "
                        f"{self.rare_negative_subsampling_policy}"
                    )
                target_negative_count = min(negative_count_int, target_negative_count)
                if target_negative_count > 0:
                    order = torch.randperm(negative_count_int, device=device)[
                        :target_negative_count
                    ]
                    selected_negatives[negative_indices.index_select(0, order)] = True

            label_mask = targets | selected_negatives
            mask[:, label_id] = label_mask
            selected_count = selected_negatives.float().sum()
            stats["selected_negative_count"][row_idx] = selected_count
            stats["selected_negative_fraction"][row_idx] = selected_count / negative_count.clamp_min(1.0)
            stats["effective_neg_per_pos"][row_idx] = selected_count / positive_count.clamp_min(1.0)
        return mask, stats

    def compute_recognition_loss(
        self,
        recognition_logits,
        multi_labels,
        return_stats=False,
    ):
        stats = self._empty_rare_negative_subsampling_stats(recognition_logits.device)
        loss_ids = torch.tensor(
            self.loss_label_ids,
            device=recognition_logits.device,
            dtype=torch.long,
        )
        loss_logits = recognition_logits.index_select(1, loss_ids)
        loss_targets = multi_labels.index_select(1, loss_ids).float()
        if self.recognition_loss_type == "asl":
            loss = asymmetric_multilabel_loss(
                loss_logits,
                loss_targets,
                gamma_neg=self.asl_gamma_neg,
                gamma_pos=self.asl_gamma_pos,
                clip=self.asl_clip,
                eps=self.asl_eps,
            )
            return (loss, stats) if return_stats else loss
        pos_weight = (
            None
            if self.recognition_pos_weight is None
            else self.recognition_pos_weight.to(
                device=recognition_logits.device,
                dtype=recognition_logits.dtype,
            )
        )
        if pos_weight is not None:
            pos_weight = pos_weight.index_select(0, loss_ids)
        raw_loss = F.binary_cross_entropy_with_logits(
            loss_logits,
            loss_targets,
            pos_weight=pos_weight,
            reduction="none",
        )
        if not (self.training and self.rare_negative_subsampling_enabled):
            loss = raw_loss.mean()
            return (loss, stats) if return_stats else loss

        loss_mask, stats = self._build_rare_negative_subsampling_mask(multi_labels)
        loss_mask = loss_mask.index_select(1, loss_ids)
        masked_loss = raw_loss * loss_mask.to(dtype=raw_loss.dtype)
        label_denominator = loss_mask.sum(dim=0).clamp_min(1).to(dtype=raw_loss.dtype)
        label_losses = masked_loss.sum(dim=0) / label_denominator
        loss = label_losses.mean()
        return (loss, stats) if return_stats else loss

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
        recognition_logits = self.compute_label_logits(z)
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
        detection_logits = self.compute_detection_logits(global_h, recognition_logits)

        loss = None
        detection_loss = None
        recognition_loss = None
        if binary_label is not None and multi_labels is not None:
            if self.detection_loss_weight > 0 and self.detection_classifier is not None:
                detection_loss = self.detection_loss_fn(
                    detection_logits,
                    binary_label.float(),
                )
            recognition_loss = self.compute_recognition_loss(
                recognition_logits,
                multi_labels,
            )
            loss = self.compute_weighted_task_loss(
                detection_loss,
                recognition_loss,
            ).reshape(1)
        return {
            "loss": loss,
            "detection_loss": detection_loss,
            "recognition_loss": recognition_loss,
            "detection_logits": detection_logits,
            "recognition_logits": recognition_logits,
            "chunk_logits": chunk_logits,
            "chunk_scores": chunk_scores,
        }


class EffectFlowGuidedChunkMIL(EVMChunkMILClassifier):
    """Template-aware evidence-guided MIL for EVEF-MVD."""

    def __init__(self, config):
        super().__init__(config)
        hidden_dim = int(config.get("hidden_dim", 512))
        efpp_dim = int(config.get("efpp_dim", 22))
        etp_dim = int(config.get("etp_dim", 16))
        relation_dim = int(config.get("relation_dim", 6))
        semantic_hidden_dim = int(config.get("semantic_projection_dim", 128))
        dropout = float(config.get("dropout", 0.1))
        ontology_path = self._resolve_config_path(config["ontology_path"])
        template_path = self._resolve_config_path(config["template_path"])
        ontology = load_ontology(ontology_path)
        label_names = config.get(
            "label_names",
            [f"label_{idx}" for idx in range(self.num_labels)],
        )
        templates = load_vulnerability_templates(
            template_path,
            ontology,
            expected_label_names=label_names,
        )
        template_bundle = build_template_tensor_bundle(label_names, templates, ontology)
        pattern_subset_path = self._resolve_config_path(
            config.get(
                "efpp_pattern_config",
                "configs/effect_flow_efpp_conservative_22.json",
            )
        )
        pattern_subset = json.loads(pattern_subset_path.read_text(encoding="utf-8"))
        pattern_index = {
            name: idx for idx, name in enumerate(template_bundle["pattern_names"])
        }
        included_pattern_indices = [
            pattern_index[name] for name in pattern_subset["included_pattern_names"]
        ]
        for key in (
            "required_patterns",
            "optional_patterns",
            "forbidden_or_counter_patterns",
            "weak_patterns",
            "role_risk_patterns",
            "role_protective_patterns",
            "role_missing_check_patterns",
        ):
            template_bundle[key] = template_bundle[key][:, included_pattern_indices]
        template_bundle["pattern_names"] = pattern_subset["included_pattern_names"]
        active_global_indices = [
            GLOBAL_VULNERABILITY_LABELS.index(label_name)
            for label_name in label_names
        ]
        self.global_template_dim = len(GLOBAL_VULNERABILITY_LABELS)
        self.active_template_dim = len(active_global_indices)
        self.semantic_input_dim = (
            efpp_dim + etp_dim + relation_dim + self.active_template_dim * 2
        )
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
        template_feature_dim = int(template_bundle["template_feature_vector"].shape[1])
        self.template_query_encoder = nn.Sequential(
            nn.Linear(template_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.alpha = nn.Parameter(torch.ones(self.num_labels))
        self.beta = nn.Parameter(torch.ones(self.num_labels))
        self.gamma = nn.Parameter(torch.ones(self.num_labels))
        self.delta = nn.Parameter(torch.ones(self.num_labels))
        self.evidence_bias = nn.Parameter(torch.zeros(self.num_labels))
        self.semantic_latent_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        self.semantic_fusion = config.get("semantic_fusion", "gated_add")
        self.evidence_align_weight = float(config.get("evidence_align_weight", 0.2))
        self.template_consistency_weight = float(
            config.get("template_consistency_weight", 0.1)
        )
        if self.semantic_fusion != "gated_add":
            raise ValueError(
                "EffectFlowGuidedChunkMIL currently supports semantic_fusion=gated_add only"
            )
        self.register_buffer(
            "active_global_indices",
            torch.tensor(active_global_indices, dtype=torch.long),
        )
        for name, tensor in template_bundle.items():
            if isinstance(tensor, torch.Tensor):
                self.register_buffer(f"template_{name}", tensor.float())

    @staticmethod
    def _resolve_config_path(path):
        path = Path(path)
        if path.is_absolute():
            return path
        return Path(__file__).resolve().parents[1] / path

    @staticmethod
    def _masked_role_mean(values, mask):
        weights = mask.unsqueeze(0).unsqueeze(0).to(values.dtype)
        denom = weights.sum(dim=-1).clamp_min(1.0)
        return (values * weights).sum(dim=-1) / denom

    def fuse_semantics(
        self,
        h,
        chunk_mask,
        efpp_probs,
        etp_distribution,
        relation_distribution,
        vulnerability_evidence_probs,
        template_match_scores,
    ):
        if (
            efpp_probs is None
            or etp_distribution is None
            or relation_distribution is None
            or vulnerability_evidence_probs is None
            or template_match_scores is None
        ):
            raise ValueError(
                "EffectFlowGuidedChunkMIL requires semantic cache v2 fields"
            )
        if efpp_probs.shape[:2] != h.shape[:2]:
            raise ValueError("efpp_probs [B, C] prefix must match chunk features")
        if etp_distribution.shape[:2] != h.shape[:2]:
            raise ValueError("etp_distribution [B, C] prefix must match chunk features")
        if relation_distribution.shape[:2] != h.shape[:2]:
            raise ValueError("relation_distribution [B, C] prefix must match chunk features")
        evidence_active = vulnerability_evidence_probs.index_select(
            -1, self.active_global_indices
        )
        template_active = template_match_scores.index_select(
            -1, self.active_global_indices
        )
        semantic_input = torch.cat(
            [
                efpp_probs.to(dtype=h.dtype),
                etp_distribution.to(dtype=h.dtype),
                relation_distribution.to(dtype=h.dtype),
                evidence_active.to(dtype=h.dtype),
                template_active.to(dtype=h.dtype),
            ],
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
        return (
            fused.masked_fill(~chunk_mask.unsqueeze(-1), 0.0),
            gate,
            evidence_active,
            template_active,
        )

    def template_aware_mil(
        self,
        h,
        chunk_mask,
        efpp_probs,
        etp_distribution,
        relation_distribution,
        evidence_active,
        template_active,
    ):
        query = self.template_query_encoder(self.template_template_feature_vector)
        query = F.normalize(query, dim=-1)
        h_norm = F.normalize(h, dim=-1)
        latent_similarity = torch.einsum("bch,lh->bcl", h_norm, query)
        risk_score = self._masked_role_mean(
            efpp_probs.unsqueeze(2),
            self.template_role_risk_patterns,
        )
        protective_score = self._masked_role_mean(
            efpp_probs.unsqueeze(2),
            self.template_role_protective_patterns,
        )
        missing_check_score = self._masked_role_mean(
            efpp_probs.unsqueeze(2),
            self.template_role_missing_check_patterns,
        )
        required_effect_score = self._masked_role_mean(
            etp_distribution.unsqueeze(2),
            self.template_required_effect_types,
        )
        relation_score = self._masked_role_mean(
            relation_distribution.unsqueeze(2),
            self.template_critical_relations,
        )
        template_prior = 0.5 * evidence_active + 0.5 * template_active
        latent_chunk_score = latent_similarity + template_prior
        risk_total = risk_score + 0.5 * required_effect_score + 0.5 * relation_score
        final_evidence_score = (
            self.alpha.view(1, 1, -1) * risk_total
            + self.beta.view(1, 1, -1) * missing_check_score
            - self.gamma.view(1, 1, -1) * protective_score
            + self.delta.view(1, 1, -1) * latent_chunk_score
            + self.evidence_bias.view(1, 1, -1)
        )
        final_evidence_score = final_evidence_score.masked_fill(
            ~chunk_mask.unsqueeze(-1),
            -1e9,
        )
        attn_weights = torch.softmax(final_evidence_score, dim=1)
        z = torch.einsum("bcl,bch->blh", attn_weights, h)
        recognition_logits = (
            z * query.unsqueeze(0)
        ).sum(dim=-1) + final_evidence_score.masked_fill(
            ~chunk_mask.unsqueeze(-1), 0.0
        ).amax(dim=1)
        return {
            "recognition_logits": recognition_logits,
            "attn_weights": attn_weights,
            "risk_evidence_scores": risk_total.masked_fill(~chunk_mask.unsqueeze(-1), 0.0),
            "protective_evidence_scores": protective_score.masked_fill(~chunk_mask.unsqueeze(-1), 0.0),
            "missing_check_evidence_scores": missing_check_score.masked_fill(~chunk_mask.unsqueeze(-1), 0.0),
            "final_evidence_scores": final_evidence_score,
            "latent_chunk_score": latent_chunk_score.masked_fill(~chunk_mask.unsqueeze(-1), 0.0),
        }

    def forward(
        self,
        chunk_features,
        chunk_mask,
        efpp_probs=None,
        etp_distribution=None,
        relation_distribution=None,
        vulnerability_evidence_probs=None,
        template_match_scores=None,
        binary_label=None,
        multi_labels=None,
        chunk_vulnerability_evidence=None,
        vulnerability_template_matches=None,
        active_vulnerability_label_mask=None,
    ):
        chunk_mask = chunk_mask.bool()
        if self.use_chunk_context:
            h = self.chunk_context_encoder(chunk_features, chunk_mask)
        else:
            h = self.chunk_projection(chunk_features)
        h, semantic_gate, evidence_active, template_active = self.fuse_semantics(
            h,
            chunk_mask,
            efpp_probs,
            etp_distribution,
            relation_distribution,
            vulnerability_evidence_probs,
            template_match_scores,
        )
        template_outputs = self.template_aware_mil(
            h,
            chunk_mask,
            efpp_probs,
            etp_distribution,
            relation_distribution,
            evidence_active,
            template_active,
        )
        recognition_logits = template_outputs["recognition_logits"]
        chunk_scores = torch.sigmoid(
            template_outputs["final_evidence_scores"].masked_fill(
                ~chunk_mask.unsqueeze(-1),
                -30.0,
            )
        ).masked_fill(~chunk_mask.unsqueeze(-1), 0.0)
        chunk_logits = template_outputs["final_evidence_scores"]
        global_h = self.masked_mean(h, chunk_mask)
        detection_logits = self.compute_detection_logits(global_h, recognition_logits)

        loss = None
        detection_loss = None
        recognition_loss = None
        evidence_align_loss = None
        template_consistency_loss = None
        if binary_label is not None and multi_labels is not None:
            if self.detection_loss_weight > 0 and self.detection_classifier is not None:
                detection_loss = self.detection_loss_fn(
                    detection_logits,
                    binary_label.float(),
                )
            recognition_loss = self.compute_recognition_loss(
                recognition_logits,
                multi_labels,
            )
            if chunk_vulnerability_evidence is None or vulnerability_template_matches is None:
                raise ValueError(
                    "EffectFlowGuidedChunkMIL requires pseudo evidence/template targets"
                )
            selected_chunk_targets = chunk_vulnerability_evidence.index_select(
                -1, self.active_global_indices
            )
            selected_template_targets = vulnerability_template_matches.index_select(
                -1, self.active_global_indices
            )
            active_mask = (
                active_vulnerability_label_mask.index_select(-1, self.active_global_indices)
                if active_vulnerability_label_mask is not None
                else torch.ones_like(multi_labels)
            )
            chunk_loss_mask = chunk_mask.unsqueeze(-1).float() * active_mask.unsqueeze(1).float()
            evidence_align_loss = F.binary_cross_entropy_with_logits(
                chunk_logits.masked_fill(~chunk_mask.unsqueeze(-1), 0.0),
                selected_chunk_targets.float(),
                reduction="none",
            )
            evidence_align_loss = (
                evidence_align_loss * chunk_loss_mask
            ).sum() / chunk_loss_mask.sum().clamp_min(1.0)
            contract_template_scores = template_outputs["final_evidence_scores"].masked_fill(
                ~chunk_mask.unsqueeze(-1),
                -1e9,
            ).amax(dim=1)
            template_targets = selected_template_targets.amax(dim=1)
            template_consistency_loss = F.binary_cross_entropy_with_logits(
                contract_template_scores,
                template_targets.float(),
                reduction="none",
            )
            template_consistency_loss = (
                template_consistency_loss * active_mask.float()
            ).sum() / active_mask.float().sum().clamp_min(1.0)
            loss = (
                self.compute_weighted_task_loss(detection_loss, recognition_loss)
                + self.evidence_align_weight * evidence_align_loss
                + self.template_consistency_weight * template_consistency_loss
            )
        return {
            "loss": loss,
            "detection_loss": detection_loss,
            "recognition_loss": recognition_loss,
            "evidence_align_loss": evidence_align_loss,
            "template_consistency_loss": template_consistency_loss,
            "detection_logits": detection_logits,
            "recognition_logits": recognition_logits,
            "chunk_logits": chunk_logits,
            "chunk_scores": chunk_scores,
            "semantic_gate": semantic_gate,
            "attn_weights": template_outputs["attn_weights"],
            "risk_evidence_scores": template_outputs["risk_evidence_scores"],
            "protective_evidence_scores": template_outputs["protective_evidence_scores"],
            "missing_check_evidence_scores": template_outputs["missing_check_evidence_scores"],
            "final_evidence_scores": template_outputs["final_evidence_scores"],
        }


class EVEFMVDV2SideEvidenceMIL(EVMChunkMILClassifier):
    """Strong-feature MIL with effect-flow side evidence.

    v2.1 keeps the strong chunk feature stream as the main representation. The
    effect-flow signals only bias label-wise attention and add a small
    label-specific evidence logit. VEP/VTM cache fields are intentionally not
    used because they are weak pseudo targets rather than reliable side
    evidence for this stage.
    """

    def __init__(self, config):
        super().__init__(config)
        efpp_dim = int(config.get("efpp_dim", 22))
        etp_dim = int(config.get("etp_dim", 16))
        relation_dim = int(config.get("relation_dim", 6))
        self.semantic_dropout_p = float(config.get("semantic_dropout", 0.2))
        self.lambda_gate = float(config.get("lambda_gate", 1e-4))
        self.warmup_epochs_neural_only = int(
            config.get("warmup_epochs_neural_only", 3)
        )
        self.enable_reliable_semantic_epoch = int(
            config.get(
                "enable_reliable_semantic_epoch",
                self.warmup_epochs_neural_only + 1,
            )
        )
        self.use_weak_semantic_features = bool(
            config.get("use_weak_semantic_features", False)
        )
        if self.use_weak_semantic_features:
            raise ValueError(
                "EVEFMVDV2SideEvidenceMIL v2.1 keeps VEP/VTM disabled. "
                "Set use_weak_semantic_features=false."
            )
        self.risk_weight = float(config.get("risk_evidence_weight", 1.0))
        self.missing_weight = float(config.get("missing_check_evidence_weight", 1.0))
        self.protective_weight = float(config.get("protective_evidence_weight", 1.0))
        self.effect_weight = float(config.get("effect_type_evidence_weight", 0.5))
        self.relation_weight = float(config.get("relation_evidence_weight", 0.5))

        ontology_path = self._resolve_config_path(config["ontology_path"])
        template_path = self._resolve_config_path(config["template_path"])
        ontology = load_ontology(ontology_path)
        label_names = config.get(
            "label_names",
            [f"label_{idx}" for idx in range(self.num_labels)],
        )
        self.new_branch_semantic_label_names = list(
            config.get("new_branch_semantic_label_names", [])
        )
        self.new_branch_semantic_label_ids = [
            label_names.index(name)
            for name in self.new_branch_semantic_label_names
            if name in label_names
        ]
        self.new_branch_semantic_enable_epoch = int(
            config.get(
                "new_branch_semantic_enable_epoch",
                self.enable_reliable_semantic_epoch,
            )
        )
        templates = load_vulnerability_templates(
            template_path,
            ontology,
            expected_label_names=label_names,
        )
        template_bundle = build_template_tensor_bundle(label_names, templates, ontology)
        pattern_subset_path = self._resolve_config_path(
            config.get(
                "efpp_pattern_config",
                "configs/effect_flow_efpp_conservative_22.json",
            )
        )
        pattern_subset = json.loads(pattern_subset_path.read_text(encoding="utf-8"))
        pattern_index = {
            name: idx for idx, name in enumerate(template_bundle["pattern_names"])
        }
        included_pattern_indices = [
            pattern_index[name] for name in pattern_subset["included_pattern_names"]
        ]
        for key in (
            "role_risk_patterns",
            "role_protective_patterns",
            "role_missing_check_patterns",
            "required_patterns",
            "optional_patterns",
            "forbidden_or_counter_patterns",
            "weak_patterns",
        ):
            template_bundle[key] = template_bundle[key][:, included_pattern_indices]
        template_bundle["pattern_names"] = pattern_subset["included_pattern_names"]

        if len(template_bundle["pattern_names"]) != efpp_dim:
            raise ValueError(
                f"EFPP template width {len(template_bundle['pattern_names'])} "
                f"does not match efpp_dim={efpp_dim}"
            )
        if template_bundle["required_effect_types"].shape[1] != etp_dim:
            raise ValueError("ETP template width does not match etp_dim")
        if template_bundle["critical_relations"].shape[1] != relation_dim:
            raise ValueError("ERR template width does not match relation_dim")

        active_global_indices = [
            GLOBAL_VULNERABILITY_LABELS.index(label_name)
            for label_name in label_names
        ]
        self.register_buffer(
            "active_global_indices",
            torch.tensor(active_global_indices, dtype=torch.long),
        )
        for name, tensor in template_bundle.items():
            if isinstance(tensor, torch.Tensor):
                self.register_buffer(f"template_{name}", tensor.float())

        self.behavior_weight_path = config.get("behavior_weight_path")
        self.use_weighted_behavior_scoring = bool(self.behavior_weight_path)
        if self.use_weighted_behavior_scoring:
            behavior_weights = self._load_behavior_weights(
                self.behavior_weight_path,
                label_names,
                template_bundle,
            )
            for name, tensor in behavior_weights.items():
                self.register_buffer(f"behavior_{name}", tensor.float())

        self.front_running_special_enabled = bool(
            config.get("front_running_special_enabled", False)
        )
        self.front_running_label_name = config.get(
            "front_running_label_name",
            "Front Running",
        )
        self.front_running_label_id = (
            label_names.index(self.front_running_label_name)
            if self.front_running_label_name in label_names
            else None
        )
        if self.front_running_special_enabled and self.front_running_label_id is None:
            raise ValueError(
                f"front_running_label_name={self.front_running_label_name} "
                "is not present in label_names"
            )
        self.front_running_generic_pattern_scale = float(
            config.get("front_running_generic_pattern_scale", 0.25)
        )
        self.front_running_special_risk_weight = float(
            config.get("front_running_special_risk_weight", 1.0)
        )
        self.front_running_special_protective_weight = float(
            config.get("front_running_special_protective_weight", 1.0)
        )
        self.front_running_confounder_suppression_weight = float(
            config.get("front_running_confounder_suppression_weight", 0.0)
        )
        self.front_running_confounder_margin = float(
            config.get("front_running_confounder_margin", 0.1)
        )
        self.front_special_feature_names = list(
            config.get("front_special_feature_names", [])
        )
        self.front_special_feature_dim = int(
            config.get("front_special_feature_dim", len(self.front_special_feature_names))
            or len(self.front_special_feature_names)
        )
        self.front_hard_negative_loss_enabled = bool(
            config.get("front_hard_negative_loss_enabled", False)
        )
        self.front_hard_negative_lambda = float(
            config.get("front_hard_negative_lambda", 0.0)
        )
        self.front_hard_negative_prob_threshold = float(
            config.get("front_hard_negative_prob_threshold", 0.5)
        )
        self.front_hard_negative_margin_logit = float(
            config.get("front_hard_negative_margin_logit", 0.0)
        )
        confounder_names = list(
            config.get(
                "front_running_confounder_labels",
                ["Access Control", "Reentrancy"],
            )
        )
        hard_negative_confounder_names = list(
            config.get("front_hard_negative_confounder_labels", confounder_names)
        )
        self.front_running_confounder_label_ids = [
            label_names.index(name) for name in confounder_names if name in label_names
        ]
        self.front_hard_negative_confounder_label_ids = [
            label_names.index(name)
            for name in hard_negative_confounder_names
            if name in label_names
        ]
        self.front_contrastive_loss_enabled = bool(
            config.get("front_contrastive_loss_enabled", False)
        )
        self.front_contrastive_lambda = float(
            config.get("front_contrastive_lambda", 0.0)
        )
        self.front_contrastive_temperature = float(
            config.get("front_contrastive_temperature", 0.1)
        )
        if self.front_contrastive_temperature <= 0:
            raise ValueError("front_contrastive_temperature must be positive")
        self.front_contrastive_enable_epoch = int(
            config.get("front_contrastive_enable_epoch", self.enable_reliable_semantic_epoch)
        )
        contrastive_confounder_names = list(
            config.get(
                "front_contrastive_confounder_labels",
                ["Access Control", "Reentrancy", "Time manipulation"],
            )
        )
        self.front_contrastive_confounder_label_ids = [
            label_names.index(name)
            for name in contrastive_confounder_names
            if name in label_names
        ]
        self.front_contrastive_negative_mining = str(
            config.get("front_contrastive_negative_mining", "all")
        )
        if self.front_contrastive_negative_mining not in {"all", "front_prob_topk"}:
            raise ValueError(
                "front_contrastive_negative_mining must be one of: "
                "all, front_prob_topk"
            )
        self.front_contrastive_hard_negative_ratio = float(
            config.get("front_contrastive_hard_negative_ratio", 2.0)
        )
        self.front_contrastive_hard_negative_min_k = int(
            config.get("front_contrastive_hard_negative_min_k", 16)
        )
        self.front_contrastive_hard_negative_max_k = int(
            config.get("front_contrastive_hard_negative_max_k", 128)
        )
        if self.front_contrastive_hard_negative_ratio <= 0:
            raise ValueError("front_contrastive_hard_negative_ratio must be positive")
        if self.front_contrastive_hard_negative_min_k <= 0:
            raise ValueError("front_contrastive_hard_negative_min_k must be positive")
        if self.front_contrastive_hard_negative_max_k <= 0:
            raise ValueError("front_contrastive_hard_negative_max_k must be positive")
        if (
            self.front_contrastive_hard_negative_max_k
            < self.front_contrastive_hard_negative_min_k
        ):
            raise ValueError(
                "front_contrastive_hard_negative_max_k must be >= "
                "front_contrastive_hard_negative_min_k"
            )
        if self.front_contrastive_loss_enabled and self.front_running_label_id is None:
            raise ValueError(
                f"front_running_label_name={self.front_running_label_name} "
                "is not present in label_names"
            )
        if self.front_contrastive_loss_enabled and not self.front_contrastive_confounder_label_ids:
            raise ValueError(
                "front_contrastive_loss_enabled=true requires at least one "
                "front_contrastive_confounder_label present in label_names"
            )
        if self.front_contrastive_loss_enabled and self.front_hard_negative_loss_enabled:
            raise ValueError(
                "front_contrastive_loss_enabled and front_hard_negative_loss_enabled "
                "must not be enabled together"
            )
        contrastive_dim = int(config.get("front_contrastive_dim", 128))
        if contrastive_dim <= 0:
            raise ValueError("front_contrastive_dim must be positive")
        self.front_contrastive_projector = None
        if self.front_contrastive_loss_enabled:
            self.front_contrastive_projector = nn.Sequential(
                nn.Linear(self.hidden_dim, contrastive_dim),
                nn.GELU(),
                nn.Linear(contrastive_dim, contrastive_dim),
            )
        if self.front_running_special_enabled:
            self._init_front_special_feature_masks()

        beta_init = float(config.get("beta_reliable_init", 0.1))
        gamma_init = float(config.get("gamma_reliable_init", 0.1))
        self.beta_reliable_raw = nn.Parameter(
            torch.full((self.num_labels,), self._softplus_inverse(beta_init))
        )
        self.gamma_reliable_raw = nn.Parameter(
            torch.full((self.num_labels,), self._softplus_inverse(gamma_init))
        )
        self.evidence_logit_weight = nn.Parameter(torch.zeros(self.num_labels, 2))
        self.evidence_logit_bias = nn.Parameter(torch.zeros(self.num_labels))
        if self.recognition_head_type == "classic_escort_residual":
            for legacy_parameter in (
                self.beta_reliable_raw,
                self.gamma_reliable_raw,
                self.evidence_logit_weight,
                self.evidence_logit_bias,
            ):
                legacy_parameter.requires_grad = False
        self.graph_evidence_enabled = bool(config.get("graph_evidence_enabled", False))
        self.graph_evidence_dim = int(config.get("graph_evidence_dim", 0))
        self.graph_evidence_scale = float(config.get("graph_evidence_scale", 1.0))
        self.graph_evidence_attention_scale = float(
            config.get("graph_evidence_attention_scale", self.graph_evidence_scale)
        )
        self.graph_evidence_logit_scale = float(
            config.get("graph_evidence_logit_scale", self.graph_evidence_scale)
        )
        self.graph_evidence_enable_epoch = int(
            config.get("graph_evidence_enable_epoch", self.enable_reliable_semantic_epoch)
        )
        graph_beta_init = float(config.get("beta_graph_init", beta_init))
        graph_gamma_init = float(config.get("gamma_graph_init", gamma_init))
        self.graph_contract_mlp = None
        self.beta_graph_raw = None
        self.gamma_graph_raw = None
        if self.graph_evidence_enabled:
            if self.graph_evidence_dim <= 0:
                raise ValueError(
                    "graph_evidence_enabled=true requires graph_evidence_dim > 0"
                )
            graph_hidden_dim = int(config.get("graph_evidence_hidden_dim", 32))
            if graph_hidden_dim <= 0:
                raise ValueError("graph_evidence_hidden_dim must be positive")
            graph_dropout = float(config.get("graph_evidence_dropout", 0.1))
            self.graph_contract_mlp = nn.Sequential(
                nn.LayerNorm(self.graph_evidence_dim),
                nn.Linear(self.graph_evidence_dim, graph_hidden_dim),
                nn.GELU(),
                nn.Dropout(graph_dropout),
                nn.Linear(graph_hidden_dim, 1),
            )
            self.beta_graph_raw = nn.Parameter(
                torch.full((self.num_labels,), self._softplus_inverse(graph_beta_init))
            )
            self.gamma_graph_raw = nn.Parameter(
                torch.full((self.num_labels,), self._softplus_inverse(graph_gamma_init))
            )
        default_major_labels = [
            "Reentrancy",
            "Access Control",
            "Arithmetic",
            "Unchecked Return Values",
            "DoS",
            "Time manipulation",
        ]
        major_label_names = list(
            config.get("major_enhancement_labels", default_major_labels)
        )
        major_label_mask = torch.zeros(self.num_labels, dtype=torch.float32)
        for name in major_label_names:
            if name in label_names:
                major_label_mask[label_names.index(name)] = 1.0
        self.register_buffer("major_enhancement_label_mask", major_label_mask)
        self.major_global_residual_enabled = bool(
            config.get("major_global_residual_enabled", False)
        )
        self.major_multipool_enabled = bool(
            config.get("major_multipool_enabled", False)
        )
        major_dropout = float(config.get("major_enhancement_dropout", config.get("dropout", 0.1)))
        major_scale_init = float(config.get("major_enhancement_scale_init", 0.1))
        self.major_global_residual_scale_raw = None
        self.major_multipool_scale_raw = None
        self.major_global_residual_mlp = None
        self.major_multipool_label_out = None
        self.major_multipool_label_bias = None
        self.major_multipool_top_k = int(config.get("major_multipool_top_k", 4))
        if self.major_multipool_top_k <= 0:
            raise ValueError("major_multipool_top_k must be positive")
        if self.major_global_residual_enabled:
            residual_hidden_dim = int(
                config.get("major_global_residual_hidden_dim", max(64, self.hidden_dim // 2))
            )
            if residual_hidden_dim <= 0:
                raise ValueError("major_global_residual_hidden_dim must be positive")
            self.major_global_residual_mlp = nn.Sequential(
                nn.LayerNorm(self.hidden_dim),
                nn.Linear(self.hidden_dim, residual_hidden_dim),
                nn.GELU(),
                nn.Dropout(major_dropout),
                nn.Linear(residual_hidden_dim, self.num_labels),
            )
            self.major_global_residual_scale_raw = nn.Parameter(
                torch.tensor(self._softplus_inverse(major_scale_init))
            )
        if self.major_multipool_enabled:
            self.major_multipool_label_out = nn.Parameter(
                torch.empty(self.num_labels, self.hidden_dim * 3)
            )
            self.major_multipool_label_bias = nn.Parameter(torch.zeros(self.num_labels))
            nn.init.xavier_uniform_(self.major_multipool_label_out)
            self.major_multipool_scale_raw = nn.Parameter(
                torch.tensor(self._softplus_inverse(major_scale_init))
            )
        self.current_epoch = 10**9

    def _init_front_special_feature_masks(self):
        if self.front_special_feature_dim <= 0:
            raise ValueError(
                "front_running_special_enabled=true requires front_special_feature_names"
            )
        if len(self.front_special_feature_names) != self.front_special_feature_dim:
            raise ValueError(
                "front_special_feature_names length must match front_special_feature_dim"
            )
        risk_names = {
            "approve_selector",
            "transfer_from_selector",
            "transfer_selector",
            "allowance_selector",
            "balance_of_selector",
            "swap_selector_family",
            "buy_sell_bid_order_text",
            "price_amount_balance_text",
            "calldata_to_state_write",
            "calldata_to_external_call",
            "external_call_then_state_write",
            "state_write_then_external_call",
        }
        protective_names = {
            "deadline_or_time_guard_text",
            "slippage_or_min_amount_text",
            "commit_reveal_text",
        }
        unknown_risk = sorted(risk_names - set(self.front_special_feature_names))
        unknown_protective = sorted(
            protective_names - set(self.front_special_feature_names)
        )
        if unknown_risk or unknown_protective:
            raise ValueError(
                "front_special_feature_names missing required features: "
                f"risk={unknown_risk}, protective={unknown_protective}"
            )
        risk_mask = torch.tensor(
            [1.0 if name in risk_names else 0.0 for name in self.front_special_feature_names],
            dtype=torch.float32,
        )
        protective_mask = torch.tensor(
            [
                1.0 if name in protective_names else 0.0
                for name in self.front_special_feature_names
            ],
            dtype=torch.float32,
        )
        self.register_buffer("front_special_risk_mask", risk_mask)
        self.register_buffer("front_special_protective_mask", protective_mask)

    @staticmethod
    def _resolve_config_path(path):
        path = Path(path)
        if path.is_absolute():
            return path
        return Path(__file__).resolve().parents[1] / path

    @staticmethod
    def _softplus_inverse(value):
        value = float(value)
        if value <= 0:
            raise ValueError("softplus inverse requires a positive value")
        return torch.log(torch.expm1(torch.tensor(value))).item()

    @staticmethod
    def _prior_mean(values, prior):
        weights = prior.unsqueeze(0).unsqueeze(0).to(values.dtype)
        denom = weights.sum(dim=-1).clamp_min(1.0)
        return (values.unsqueeze(2) * weights).sum(dim=-1) / denom

    def _load_behavior_weights(self, path, label_names, template_bundle):
        path = self._resolve_config_path(path)
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{path} must contain a YAML mapping.")
        if bool(payload.get("learnable", False)):
            raise ValueError("DIVE behavior weights must be fixed: learnable=false.")
        if payload.get("score_normalization", "none") != "none":
            raise ValueError("Only score_normalization=none is supported.")
        label_specs = payload.get("labels")
        if not isinstance(label_specs, dict):
            raise ValueError(f"{path} missing top-level labels mapping.")

        missing_labels = [name for name in label_names if name not in label_specs]
        extra_labels = sorted(set(label_specs) - set(label_names))
        if missing_labels or extra_labels:
            raise ValueError(
                "Behavior weight labels must exactly match config label_names. "
                f"missing={missing_labels}, extra={extra_labels}"
            )

        pattern_vocab = list(template_bundle["pattern_names"])
        effect_vocab = list(template_bundle["effect_type_names"])
        relation_vocab = list(template_bundle["relation_names"])
        matrices = {
            "risk_pattern_weights": torch.zeros(
                (len(label_names), len(pattern_vocab)), dtype=torch.float32
            ),
            "missing_pattern_weights": torch.zeros(
                (len(label_names), len(pattern_vocab)), dtype=torch.float32
            ),
            "protective_pattern_weights": torch.zeros(
                (len(label_names), len(pattern_vocab)), dtype=torch.float32
            ),
            "effect_weights": torch.zeros(
                (len(label_names), len(effect_vocab)), dtype=torch.float32
            ),
            "risk_relation_weights": torch.zeros(
                (len(label_names), len(relation_vocab)), dtype=torch.float32
            ),
            "missing_relation_weights": torch.zeros(
                (len(label_names), len(relation_vocab)), dtype=torch.float32
            ),
            "protective_relation_weights": torch.zeros(
                (len(label_names), len(relation_vocab)), dtype=torch.float32
            ),
        }
        field_specs = {
            "risk_patterns": ("risk_pattern_weights", pattern_vocab),
            "missing_check_patterns": ("missing_pattern_weights", pattern_vocab),
            "protective_patterns": ("protective_pattern_weights", pattern_vocab),
            "effect_types": ("effect_weights", effect_vocab),
            "risk_relations": ("risk_relation_weights", relation_vocab),
            "missing_relations": ("missing_relation_weights", relation_vocab),
            "protective_relations": ("protective_relation_weights", relation_vocab),
        }
        relation_schema = payload.get("relation_schema", "legacy")
        for label_idx, label_name in enumerate(label_names):
            spec = label_specs[label_name]
            if not isinstance(spec, dict):
                raise ValueError(f"{label_name} behavior weight spec must be a mapping.")
            for field, (matrix_name, vocab) in field_specs.items():
                missing_field_allowed = (
                    relation_schema != "split"
                    and field in {"missing_relations", "protective_relations"}
                )
                legacy_risk_relations_allowed = (
                    field == "risk_relations" and "relations" in spec
                )
                if field not in spec and not (
                    missing_field_allowed or legacy_risk_relations_allowed
                ):
                    raise ValueError(f"{label_name} missing behavior weight field: {field}")
                entries = spec.get(field)
                if entries is None and field == "risk_relations":
                    # Backward compatibility for v1 weight files. In v1 all
                    # relations were treated as risk-side evidence.
                    entries = spec.get("relations")
                if entries is None:
                    entries = {}
                if not isinstance(entries, dict):
                    raise ValueError(
                        f"{label_name}.{field} must be a name-to-weight mapping."
                    )
                vocab_index = {name: idx for idx, name in enumerate(vocab)}
                unknown = sorted(set(entries) - set(vocab_index))
                if unknown:
                    raise ValueError(
                        f"{label_name}.{field} contains unknown names: {unknown}"
                    )
                for name, weight in entries.items():
                    try:
                        value = float(weight)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            f"{label_name}.{field}.{name} weight must be numeric."
                        ) from exc
                    matrices[matrix_name][label_idx, vocab_index[name]] = value

            if relation_schema == "split":
                missing_relation_fields = [
                    field
                    for field in (
                        "risk_relations",
                        "missing_relations",
                        "protective_relations",
                    )
                    if field not in spec
                ]
                if missing_relation_fields:
                    raise ValueError(
                        f"{label_name} missing split relation fields: "
                        f"{missing_relation_fields}"
                    )

        return matrices

    @staticmethod
    def _weighted_sum(values, weights):
        return torch.einsum("bcd,ld->bcl", values, weights.to(values.dtype))

    def set_training_epoch(self, epoch):
        self.current_epoch = int(epoch)

    def semantic_enabled(self):
        return int(self.current_epoch) >= self.enable_reliable_semantic_epoch

    def graph_enabled(self):
        return (
            self.graph_evidence_enabled
            and int(self.current_epoch) >= self.graph_evidence_enable_epoch
        )

    def _semantic_dropout(self, values):
        if not self.training or self.semantic_dropout_p <= 0:
            return values
        keep = torch.rand_like(values) >= self.semantic_dropout_p
        return values * keep.type_as(values)

    def encode_strong_chunks(self, chunk_features, chunk_mask):
        if self.use_chunk_context:
            return self.chunk_context_encoder(chunk_features, chunk_mask)
        return self.chunk_projection(chunk_features)

    def compute_graph_evidence(
        self,
        graph_contract_evidence,
        graph_chunk_evidence,
        chunk_mask,
        dtype,
    ):
        zero_chunk = torch.zeros(
            (*chunk_mask.shape, self.num_labels),
            device=chunk_mask.device,
            dtype=dtype,
        )
        zero_contract = torch.zeros(
            (chunk_mask.shape[0], self.num_labels),
            device=chunk_mask.device,
            dtype=dtype,
        )
        zero_label = torch.zeros(self.num_labels, device=chunk_mask.device, dtype=dtype)
        if not self.graph_evidence_enabled:
            return {
                "attention_bias": zero_chunk,
                "contract_logits": zero_contract,
                "raw_contract_logits": zero_contract,
                "chunk_scores": zero_chunk,
                "beta_graph": zero_label,
                "gamma_graph": zero_label,
                "enabled": False,
            }
        if graph_contract_evidence is None or graph_chunk_evidence is None:
            raise ValueError(
                "graph_evidence_enabled=true requires graph_contract_evidence "
                "and graph_chunk_evidence"
            )
        if graph_contract_evidence.shape[:2] != (
            chunk_mask.shape[0],
            self.num_labels,
        ):
            raise ValueError(
                "graph_contract_evidence must have shape [B, num_labels, D]"
            )
        if graph_contract_evidence.shape[-1] != self.graph_evidence_dim:
            raise ValueError(
                f"graph_contract_evidence width {graph_contract_evidence.shape[-1]} "
                f"does not match graph_evidence_dim={self.graph_evidence_dim}"
            )
        if graph_chunk_evidence.shape != (
            chunk_mask.shape[0],
            chunk_mask.shape[1],
            self.num_labels,
        ):
            raise ValueError(
                "graph_chunk_evidence must have shape [B, max_chunks, num_labels]"
            )
        if not self.graph_enabled():
            return {
                "attention_bias": zero_chunk,
                "contract_logits": zero_contract,
                "raw_contract_logits": zero_contract,
                "chunk_scores": zero_chunk,
                "beta_graph": zero_label,
                "gamma_graph": zero_label,
                "enabled": False,
            }
        chunk_scores = graph_chunk_evidence.to(dtype=dtype).clamp(0.0, 1.0)
        chunk_scores = chunk_scores.masked_fill(~chunk_mask.unsqueeze(-1), 0.0)
        beta_graph = F.softplus(self.beta_graph_raw).to(dtype=dtype)
        gamma_graph = F.softplus(self.gamma_graph_raw).to(dtype=dtype)
        attention_bias = (
            self.graph_evidence_attention_scale
            * beta_graph.view(1, 1, -1)
            * chunk_scores
        )
        contract_values = graph_contract_evidence.to(dtype=dtype).clamp(0.0, 1.0)
        flat = contract_values.reshape(-1, self.graph_evidence_dim)
        raw_logits = self.graph_contract_mlp(flat).reshape(
            chunk_mask.shape[0],
            self.num_labels,
        )
        contract_logits = (
            self.graph_evidence_logit_scale
            * gamma_graph.view(1, -1)
            * raw_logits
        )
        return {
            "attention_bias": attention_bias,
            "contract_logits": contract_logits,
            "raw_contract_logits": raw_logits,
            "chunk_scores": chunk_scores,
            "beta_graph": beta_graph,
            "gamma_graph": gamma_graph,
            "enabled": True,
        }

    def compute_reliable_evidence(
        self,
        efpp_probs,
        etp_distribution,
        relation_distribution,
        chunk_mask,
    ):
        if efpp_probs is None or etp_distribution is None or relation_distribution is None:
            raise ValueError(
                "EVEFMVDV2SideEvidenceMIL requires efpp_probs, "
                "etp_distribution, and relation_distribution."
            )
        if efpp_probs.shape[:2] != chunk_mask.shape:
            raise ValueError("efpp_probs [B, C] prefix must match chunk_mask")
        if etp_distribution.shape[:2] != chunk_mask.shape:
            raise ValueError("etp_distribution [B, C] prefix must match chunk_mask")
        if relation_distribution.shape[:2] != chunk_mask.shape:
            raise ValueError("relation_distribution [B, C] prefix must match chunk_mask")
        efpp_probs = self._semantic_dropout(efpp_probs.float())
        etp_distribution = self._semantic_dropout(etp_distribution.float())
        relation_distribution = self._semantic_dropout(relation_distribution.float())

        if self.use_weighted_behavior_scoring:
            risk_pattern = self._weighted_sum(
                efpp_probs,
                self.behavior_risk_pattern_weights,
            )
            protective = self._weighted_sum(
                efpp_probs,
                self.behavior_protective_pattern_weights,
            )
            missing = self._weighted_sum(
                efpp_probs,
                self.behavior_missing_pattern_weights,
            )
            required_effect = self._weighted_sum(
                etp_distribution,
                self.behavior_effect_weights,
            )
            risk_relation = self._weighted_sum(
                relation_distribution,
                self.behavior_risk_relation_weights,
            )
            missing_relation = self._weighted_sum(
                relation_distribution,
                self.behavior_missing_relation_weights,
            )
            protective_relation = self._weighted_sum(
                relation_distribution,
                self.behavior_protective_relation_weights,
            )
            risk_total = risk_pattern + required_effect + risk_relation
            missing = missing + missing_relation
            protective = protective + protective_relation
            final = risk_total + missing - protective
        else:
            risk_pattern = self._prior_mean(efpp_probs, self.template_role_risk_patterns)
            protective = self._prior_mean(
                efpp_probs,
                self.template_role_protective_patterns,
            )
            missing = self._prior_mean(
                efpp_probs,
                self.template_role_missing_check_patterns,
            )
            required_effect = self._prior_mean(
                etp_distribution,
                self.template_required_effect_types,
            )
            relation = self._prior_mean(
                relation_distribution,
                self.template_critical_relations,
            )
            risk_total = (
                risk_pattern
                + self.effect_weight * required_effect
                + self.relation_weight * relation
            )
            final = (
                self.risk_weight * risk_total
                + self.missing_weight * missing
                - self.protective_weight * protective
            )
        valid = chunk_mask.unsqueeze(-1)
        return {
            "risk_evidence_scores": risk_total.masked_fill(~valid, 0.0),
            "protective_evidence_scores": protective.masked_fill(~valid, 0.0),
            "missing_check_evidence_scores": missing.masked_fill(~valid, 0.0),
            "final_evidence_scores": final.masked_fill(~valid, 0.0),
        }

    def compute_label_logits_from_pools(
        self, z, h, attn_weights, phi, chunk_mask, global_h
    ):
        """Default v2 head: classify the evidence-attended label representation."""
        del h, attn_weights, phi, chunk_mask, global_h
        return self.compute_label_logits(z)

    def compute_front_special_scores(self, front_special_features, chunk_mask):
        if not self.front_running_special_enabled:
            return None
        if front_special_features is None:
            raise ValueError(
                "front_running_special_enabled=true requires front_special_features"
            )
        if front_special_features.shape[:2] != chunk_mask.shape:
            raise ValueError(
                "front_special_features [B, C] prefix must match chunk_mask"
            )
        if front_special_features.shape[-1] != self.front_special_feature_dim:
            raise ValueError(
                f"front_special_features width {front_special_features.shape[-1]} "
                f"does not match front_special_feature_dim={self.front_special_feature_dim}"
            )
        values = front_special_features.float()
        risk_mask = self.front_special_risk_mask.to(values.dtype)
        protective_mask = self.front_special_protective_mask.to(values.dtype)
        risk_denom = risk_mask.sum().clamp_min(1.0)
        protective_denom = protective_mask.sum().clamp_min(1.0)
        risk_score = (values * risk_mask.view(1, 1, -1)).sum(dim=-1) / risk_denom
        protective_score = (
            values * protective_mask.view(1, 1, -1)
        ).sum(dim=-1) / protective_denom
        risk_score = risk_score.masked_fill(~chunk_mask, 0.0)
        protective_score = protective_score.masked_fill(~chunk_mask, 0.0)
        final_score = risk_score - protective_score
        return {
            "front_special_risk": risk_score,
            "front_special_protective": protective_score,
            "front_special_final": final_score.masked_fill(~chunk_mask, 0.0),
        }

    def apply_front_running_special_evidence(
        self,
        evidence,
        front_special_features,
        chunk_mask,
    ):
        if not self.front_running_special_enabled:
            return evidence, {}
        front_scores = self.compute_front_special_scores(
            front_special_features,
            chunk_mask,
        )
        front_id = int(self.front_running_label_id)
        adjusted = {
            key: value.clone()
            for key, value in evidence.items()
        }
        final = adjusted["final_evidence_scores"]
        risk = adjusted["risk_evidence_scores"]
        protective = adjusted["protective_evidence_scores"]
        missing = adjusted["missing_check_evidence_scores"]
        front_specific = front_scores["front_special_final"].to(final.dtype)
        front_risk = front_scores["front_special_risk"].to(final.dtype)
        front_protective = front_scores["front_special_protective"].to(final.dtype)
        if self.front_running_confounder_label_ids:
            confounder_phi = final[:, :, self.front_running_confounder_label_ids]
            confounder_max = confounder_phi.max(dim=-1).values
        else:
            confounder_max = torch.zeros_like(front_specific)
        suppression = F.relu(
            confounder_max - front_specific - self.front_running_confounder_margin
        )
        adjusted_front = (
            self.front_running_generic_pattern_scale * final[:, :, front_id]
            + self.front_running_special_risk_weight * front_specific
            - self.front_running_confounder_suppression_weight * suppression
        )
        final[:, :, front_id] = adjusted_front
        risk[:, :, front_id] = (
            self.front_running_generic_pattern_scale * risk[:, :, front_id]
            + self.front_running_special_risk_weight * front_risk
        )
        protective[:, :, front_id] = (
            self.front_running_generic_pattern_scale * protective[:, :, front_id]
            + self.front_running_special_protective_weight * front_protective
            + self.front_running_confounder_suppression_weight * suppression
        )
        missing[:, :, front_id] = (
            self.front_running_generic_pattern_scale * missing[:, :, front_id]
        )
        diagnostics = {
            "front_special_risk_scores": front_risk.masked_fill(~chunk_mask, 0.0),
            "front_special_protective_scores": front_protective.masked_fill(~chunk_mask, 0.0),
            "front_special_final_scores": front_specific.masked_fill(~chunk_mask, 0.0),
            "front_confounder_suppression_scores": suppression.masked_fill(~chunk_mask, 0.0),
        }
        return adjusted, diagnostics

    def gate_values(self):
        if self.recognition_head_type == "classic_escort_residual":
            beta_raw = torch.stack(
                [branch.beta_reliable_raw for branch in self.classic_label_branches]
            )
            gamma_raw = torch.stack(
                [branch.gamma_reliable_raw for branch in self.classic_label_branches]
            )
            values = {
                "beta_reliable": F.softplus(beta_raw).detach(),
                "gamma_reliable": F.softplus(gamma_raw).detach(),
            }
            return values
        values = {
            "beta_reliable": F.softplus(self.beta_reliable_raw).detach(),
            "gamma_reliable": F.softplus(self.gamma_reliable_raw).detach(),
        }
        if self.graph_evidence_enabled:
            values["beta_graph"] = F.softplus(self.beta_graph_raw).detach()
            values["gamma_graph"] = F.softplus(self.gamma_graph_raw).detach()
        return values

    def _active_label_parameters(self, dtype, device):
        if self.recognition_head_type == "classic_escort_residual":
            queries = torch.stack(
                [branch.attention_query for branch in self.classic_label_branches]
            )
            beta_raw = torch.stack(
                [branch.beta_reliable_raw for branch in self.classic_label_branches]
            )
            gamma_raw = torch.stack(
                [branch.gamma_reliable_raw for branch in self.classic_label_branches]
            )
            evidence_weight = torch.stack(
                [branch.evidence_logit_weight for branch in self.classic_label_branches]
            )
            evidence_bias = torch.stack(
                [branch.evidence_logit_bias for branch in self.classic_label_branches]
            )
        else:
            queries = self.label_attn
            beta_raw = self.beta_reliable_raw
            gamma_raw = self.gamma_reliable_raw
            evidence_weight = self.evidence_logit_weight
            evidence_bias = self.evidence_logit_bias
        beta = F.softplus(beta_raw).to(device=device, dtype=dtype)
        gamma = F.softplus(gamma_raw).to(device=device, dtype=dtype)
        semantic_mask = torch.ones_like(beta)
        if not self.semantic_enabled():
            semantic_mask.zero_()
        elif (
            self.new_branch_semantic_label_ids
            and int(self.current_epoch) < self.new_branch_semantic_enable_epoch
        ):
            semantic_mask[self.new_branch_semantic_label_ids] = 0.0
        return {
            "queries": queries.to(device=device, dtype=dtype),
            "beta_raw": beta_raw,
            "gamma_raw": gamma_raw,
            "beta": beta * semantic_mask,
            "gamma": gamma * semantic_mask,
            "evidence_weight": evidence_weight.to(device=device, dtype=dtype),
            "evidence_bias": evidence_bias.to(device=device, dtype=dtype),
        }

    def compute_front_contrastive_loss(
        self,
        label_representations,
        multi_labels,
        recognition_logits=None,
    ):
        front_z = label_representations[:, int(self.front_running_label_id), :]
        zero_loss = front_z.sum() * 0.0
        zero_count = torch.tensor(0.0, device=front_z.device)
        stats = {
            "loss": zero_loss,
            "anchor_count": zero_count,
            "positive_count": zero_count,
            "candidate_negative_count": zero_count,
            "hard_negative_count": zero_count,
            "selected_hard_negative_count": zero_count,
            "selected_prob_mean": zero_count,
            "selected_prob_max": zero_count,
        }
        if (
            not self.training
            or not self.front_contrastive_loss_enabled
            or self.front_contrastive_lambda <= 0
            or int(self.current_epoch) < self.front_contrastive_enable_epoch
        ):
            return stats

        labels = multi_labels.float()
        front_positive = labels[:, int(self.front_running_label_id)] > 0.5
        confounder_positive = (
            labels[:, self.front_contrastive_confounder_label_ids].max(dim=1).values
            > 0.5
        )
        candidate_negative = (~front_positive) & confounder_positive
        front_positive_count = front_positive.float().sum()
        candidate_negative_count = candidate_negative.float().sum()
        stats["candidate_negative_count"] = candidate_negative_count
        if bool(front_positive_count < 2) or bool(candidate_negative_count < 1):
            return stats

        selected_negative = candidate_negative
        selected_scores = None
        if self.front_contrastive_negative_mining == "front_prob_topk":
            if recognition_logits is None:
                raise ValueError(
                    "front_prob_topk mining requires recognition_logits"
                )
            front_logits = recognition_logits[:, int(self.front_running_label_id)]
            front_probs = torch.sigmoid(front_logits).detach()
            candidate_indices = candidate_negative.nonzero(as_tuple=False).flatten()
            anchor_count = int(front_positive_count.detach().cpu().item())
            candidate_count = int(candidate_negative_count.detach().cpu().item())
            target_k = max(
                self.front_contrastive_hard_negative_min_k,
                int(torch.ceil(front_positive_count * self.front_contrastive_hard_negative_ratio).detach().cpu().item()),
            )
            k = min(candidate_count, self.front_contrastive_hard_negative_max_k, target_k)
            selected_negative = torch.zeros_like(candidate_negative)
            if k > 0:
                candidate_scores = front_probs[candidate_indices]
                topk = torch.topk(candidate_scores, k=k, largest=True)
                selected_indices = candidate_indices[topk.indices]
                selected_negative[selected_indices] = True
                selected_scores = topk.values
            if anchor_count < 2 or k < 1:
                return stats
        elif recognition_logits is not None:
            front_logits = recognition_logits[:, int(self.front_running_label_id)]
            front_probs = torch.sigmoid(front_logits).detach()
            selected_scores = front_probs[selected_negative]

        selected_negative_count = selected_negative.float().sum()
        if bool(selected_negative_count < 1):
            return stats

        projected = self.front_contrastive_projector(front_z)
        projected = F.normalize(projected, dim=-1)
        logits = torch.matmul(projected, projected.transpose(0, 1))
        logits = logits / self.front_contrastive_temperature

        batch_size = logits.shape[0]
        eye = torch.eye(batch_size, device=logits.device, dtype=torch.bool)
        positive_mask = front_positive.unsqueeze(0).expand(batch_size, batch_size) & ~eye
        hard_negative_mask = selected_negative.unsqueeze(0).expand(batch_size, batch_size)
        denominator_mask = positive_mask | hard_negative_mask

        row_mask = front_positive
        row_logits = logits[row_mask]
        row_positive_mask = positive_mask[row_mask]
        row_denominator_mask = denominator_mask[row_mask]
        fill_value = torch.finfo(row_logits.dtype).min
        log_positive = torch.logsumexp(
            row_logits.masked_fill(~row_positive_mask, fill_value),
            dim=1,
        )
        log_denominator = torch.logsumexp(
            row_logits.masked_fill(~row_denominator_mask, fill_value),
            dim=1,
        )
        loss = -(log_positive - log_denominator).mean()
        selected_prob_mean = zero_count
        selected_prob_max = zero_count
        if selected_scores is not None and selected_scores.numel() > 0:
            selected_prob_mean = selected_scores.mean()
            selected_prob_max = selected_scores.max()
        stats.update(
            {
                "loss": loss,
                "anchor_count": row_mask.float().sum(),
                "positive_count": row_positive_mask.float().sum(),
                "candidate_negative_count": candidate_negative_count,
                "hard_negative_count": selected_negative_count,
                "selected_hard_negative_count": selected_negative_count,
                "selected_prob_mean": selected_prob_mean,
                "selected_prob_max": selected_prob_max,
            }
        )
        return stats

    def forward(
        self,
        chunk_features,
        chunk_mask,
        efpp_probs=None,
        etp_distribution=None,
        relation_distribution=None,
        vulnerability_evidence_probs=None,
        template_match_scores=None,
        binary_label=None,
        multi_labels=None,
        chunk_vulnerability_evidence=None,
        vulnerability_template_matches=None,
        active_vulnerability_label_mask=None,
        front_special_features=None,
        graph_contract_evidence=None,
        graph_chunk_evidence=None,
    ):
        del vulnerability_evidence_probs
        del template_match_scores
        del chunk_vulnerability_evidence
        del vulnerability_template_matches
        del active_vulnerability_label_mask

        chunk_mask = chunk_mask.bool()
        h = self.encode_strong_chunks(chunk_features, chunk_mask)
        evidence = self.compute_reliable_evidence(
            efpp_probs,
            etp_distribution,
            relation_distribution,
            chunk_mask,
        )
        evidence, front_diagnostics = self.apply_front_running_special_evidence(
            evidence,
            front_special_features,
            chunk_mask,
        )
        phi = evidence["final_evidence_scores"].to(dtype=h.dtype)
        graph_outputs = self.compute_graph_evidence(
            graph_contract_evidence,
            graph_chunk_evidence,
            chunk_mask,
            h.dtype,
        )

        v = torch.tanh(self.attn_v(h))
        u = torch.sigmoid(self.attn_u(h))
        gated = v * u
        label_parameters = self._active_label_parameters(h.dtype, h.device)
        if self.recognition_head_type == "classic_escort_residual":
            neural_attn_logits = torch.stack(
                [
                    torch.einsum("bca,a->bc", gated, query)
                    for query in label_parameters["queries"]
                ],
                dim=-1,
            )
        else:
            neural_attn_logits = torch.einsum(
                "bca,ka->bck", gated, label_parameters["queries"]
            )
        beta = label_parameters["beta"]
        gamma = label_parameters["gamma"]
        attn_logits = (
            neural_attn_logits
            + beta.view(1, 1, -1) * phi
            + graph_outputs["attention_bias"]
        )
        attn_logits = attn_logits.masked_fill(~chunk_mask.unsqueeze(-1), -1e9)
        attn_weights = torch.softmax(attn_logits, dim=1)
        z = torch.einsum("bck,bch->bkh", attn_weights, h)
        global_h = self.masked_mean(h, chunk_mask)
        neural_logits = self.compute_label_logits_from_pools(
            z, h, attn_weights, phi, chunk_mask, global_h
        )
        major_mask = self.major_enhancement_label_mask.to(dtype=h.dtype).view(1, -1)
        major_global_residual_logits = neural_logits.new_zeros(neural_logits.shape)
        if self.major_global_residual_enabled:
            global_scale = F.softplus(self.major_global_residual_scale_raw).to(dtype=h.dtype)
            major_global_residual_logits = (
                global_scale * self.major_global_residual_mlp(global_h).to(dtype=h.dtype) * major_mask
            )
        major_multipool_logits = neural_logits.new_zeros(neural_logits.shape)
        if self.major_multipool_enabled:
            top_k = min(self.major_multipool_top_k, h.shape[1])
            masked_weights = attn_weights.masked_fill(~chunk_mask.unsqueeze(-1), -1.0)
            top_values, top_indices = torch.topk(masked_weights, k=top_k, dim=1)
            h_expanded = h.unsqueeze(2).expand(-1, -1, self.num_labels, -1)
            gather_index = top_indices.unsqueeze(-1).expand(-1, -1, -1, h.shape[-1])
            top_h = torch.gather(h_expanded, 1, gather_index)
            valid_top = (top_values >= 0.0).unsqueeze(-1).to(dtype=h.dtype)
            top_h = (
                (top_h * valid_top).sum(dim=1)
                / valid_top.sum(dim=1).clamp_min(1.0)
            )
            global_for_labels = global_h.unsqueeze(1).expand(-1, self.num_labels, -1)
            multipool_repr = torch.cat([z, top_h, global_for_labels], dim=-1)
            raw_multipool_logits = (
                multipool_repr
                * self.major_multipool_label_out.unsqueeze(0).to(dtype=h.dtype)
            ).sum(dim=-1) + self.major_multipool_label_bias.unsqueeze(0).to(dtype=h.dtype)
            multipool_scale = F.softplus(self.major_multipool_scale_raw).to(dtype=h.dtype)
            major_multipool_logits = multipool_scale * raw_multipool_logits * major_mask

        attended_phi = (attn_weights * phi).sum(dim=1)
        masked_phi = phi.masked_fill(~chunk_mask.unsqueeze(-1), -1e9)
        k = min(self.top_k, phi.shape[1])
        top_phi = torch.topk(masked_phi, k=k, dim=1).values
        valid_top = top_phi > -1e8
        topk_phi = (
            top_phi.masked_fill(~valid_top, 0.0).sum(dim=1)
            / valid_top.sum(dim=1).clamp_min(1).to(dtype=h.dtype)
        )
        evidence_features = torch.stack([attended_phi, topk_phi], dim=-1)
        evidence_logits = (
            evidence_features * label_parameters["evidence_weight"].unsqueeze(0)
        ).sum(dim=-1) + label_parameters["evidence_bias"].unsqueeze(0)
        recognition_logits = (
            neural_logits
            + gamma.view(1, -1) * evidence_logits
            + graph_outputs["contract_logits"]
            + major_global_residual_logits
            + major_multipool_logits
        )

        detection_logits = self.compute_detection_logits(global_h, recognition_logits)
        chunk_scores = torch.sigmoid(
            attn_logits.masked_fill(~chunk_mask.unsqueeze(-1), -30.0)
        ).masked_fill(~chunk_mask.unsqueeze(-1), 0.0)

        loss = None
        detection_loss = None
        recognition_loss = None
        gate_regularization_loss = None
        front_hard_negative_loss = None
        front_hard_negative_active_count = torch.tensor(0.0, device=h.device)
        front_contrastive_loss = None
        front_contrastive_anchor_count = torch.tensor(0.0, device=h.device)
        front_contrastive_positive_count = torch.tensor(0.0, device=h.device)
        front_contrastive_candidate_negative_count = torch.tensor(0.0, device=h.device)
        front_contrastive_hard_negative_count = torch.tensor(0.0, device=h.device)
        front_contrastive_selected_hard_negative_count = torch.tensor(0.0, device=h.device)
        front_contrastive_selected_prob_mean = torch.tensor(0.0, device=h.device)
        front_contrastive_selected_prob_max = torch.tensor(0.0, device=h.device)
        rare_negative_subsampling_stats = self._empty_rare_negative_subsampling_stats(
            h.device
        )
        if binary_label is not None and multi_labels is not None:
            if self.detection_loss_weight > 0 and self.detection_classifier is not None:
                detection_loss = self.detection_loss_fn(
                    detection_logits,
                    binary_label.float(),
                )
            recognition_loss, rare_negative_subsampling_stats = self.compute_recognition_loss(
                recognition_logits,
                multi_labels,
                return_stats=True,
            )
            active_beta = F.softplus(label_parameters["beta_raw"])
            active_gamma = F.softplus(label_parameters["gamma_raw"])
            gate_regularization_loss = (
                active_beta.pow(2).mean() + active_gamma.pow(2).mean()
            )
            if self.graph_evidence_enabled:
                active_beta_graph = F.softplus(self.beta_graph_raw)
                active_gamma_graph = F.softplus(self.gamma_graph_raw)
                gate_regularization_loss = gate_regularization_loss + (
                    active_beta_graph.pow(2).mean()
                    + active_gamma_graph.pow(2).mean()
                )
            front_hard_negative_loss = recognition_logits.sum() * 0.0
            if (
                self.training
                and self.front_hard_negative_loss_enabled
                and self.front_hard_negative_lambda > 0
                and self.front_running_label_id is not None
                and self.front_hard_negative_confounder_label_ids
            ):
                front_id = int(self.front_running_label_id)
                front_logits = recognition_logits[:, front_id]
                front_probs = torch.sigmoid(front_logits)
                confounder_labels = multi_labels[
                    :,
                    self.front_hard_negative_confounder_label_ids,
                ].float()
                confounder_positive = confounder_labels.max(dim=1).values > 0.5
                front_negative = multi_labels[:, front_id].float() < 0.5
                high_front_prob = front_probs >= self.front_hard_negative_prob_threshold
                hard_mask = front_negative & confounder_positive & high_front_prob
                front_hard_negative_active_count = hard_mask.float().sum()
                if bool(hard_mask.any()):
                    front_hard_negative_loss = F.softplus(
                        front_logits[hard_mask] - self.front_hard_negative_margin_logit
                    ).mean()
            front_contrastive_loss = recognition_logits.sum() * 0.0
            if (
                self.front_contrastive_loss_enabled
                and self.front_contrastive_lambda > 0
                and self.front_running_label_id is not None
            ):
                contrastive_stats = self.compute_front_contrastive_loss(
                    z,
                    multi_labels,
                    recognition_logits=recognition_logits,
                )
                front_contrastive_loss = contrastive_stats["loss"]
                front_contrastive_anchor_count = contrastive_stats["anchor_count"]
                front_contrastive_positive_count = contrastive_stats["positive_count"]
                front_contrastive_candidate_negative_count = contrastive_stats[
                    "candidate_negative_count"
                ]
                front_contrastive_hard_negative_count = contrastive_stats[
                    "hard_negative_count"
                ]
                front_contrastive_selected_hard_negative_count = contrastive_stats[
                    "selected_hard_negative_count"
                ]
                front_contrastive_selected_prob_mean = contrastive_stats[
                    "selected_prob_mean"
                ]
                front_contrastive_selected_prob_max = contrastive_stats[
                    "selected_prob_max"
                ]
            loss = (
                self.compute_weighted_task_loss(detection_loss, recognition_loss)
                + self.lambda_gate * gate_regularization_loss
                + self.front_hard_negative_lambda * front_hard_negative_loss
                + self.front_contrastive_lambda * front_contrastive_loss
            ).reshape(1)

        outputs = {
            "loss": loss,
            "detection_loss": detection_loss,
            "recognition_loss": recognition_loss,
            "gate_regularization_loss": gate_regularization_loss,
            "front_hard_negative_loss": front_hard_negative_loss,
            "front_hard_negative_active_count": front_hard_negative_active_count,
            "front_contrastive_loss": front_contrastive_loss,
            "front_contrastive_anchor_count": front_contrastive_anchor_count,
            "front_contrastive_positive_count": front_contrastive_positive_count,
            "front_contrastive_candidate_negative_count": front_contrastive_candidate_negative_count,
            "front_contrastive_hard_negative_count": front_contrastive_hard_negative_count,
            "front_contrastive_selected_hard_negative_count": front_contrastive_selected_hard_negative_count,
            "front_contrastive_selected_prob_mean": front_contrastive_selected_prob_mean,
            "front_contrastive_selected_prob_max": front_contrastive_selected_prob_max,
            "rare_negative_subsampling_stats": rare_negative_subsampling_stats,
            "detection_logits": detection_logits,
            "recognition_logits": recognition_logits,
            "chunk_logits": attn_logits,
            "chunk_scores": chunk_scores,
            "attn_weights": attn_weights,
            "neural_attention_logits": neural_attn_logits,
            "neural_logits": neural_logits,
            "evidence_logits": evidence_logits,
            "major_global_residual_logits": major_global_residual_logits,
            "major_multipool_logits": major_multipool_logits,
            "major_enhancement_label_mask": self.major_enhancement_label_mask,
            "graph_contract_logits": graph_outputs["contract_logits"],
            "graph_raw_contract_logits": graph_outputs["raw_contract_logits"],
            "graph_chunk_scores": graph_outputs["chunk_scores"],
            "beta_graph": graph_outputs["beta_graph"],
            "gamma_graph": graph_outputs["gamma_graph"],
            "graph_enabled": torch.tensor(
                bool(graph_outputs["enabled"]),
                device=h.device,
            ),
            "beta_reliable": beta,
            "gamma_reliable": gamma,
            "semantic_enabled": torch.tensor(
                bool(self.semantic_enabled()),
                device=h.device,
            ),
            "risk_evidence_scores": evidence["risk_evidence_scores"],
            "protective_evidence_scores": evidence["protective_evidence_scores"],
            "missing_check_evidence_scores": evidence[
                "missing_check_evidence_scores"
            ],
            "final_evidence_scores": evidence["final_evidence_scores"],
        }
        outputs.update(front_diagnostics)
        return outputs


class EVEFMVDV2MultiScaleMIL(EVEFMVDV2SideEvidenceMIL):
    """Main-6 label-wise MIL with evidence, cross-attention, top-k, and global pools."""

    def __init__(self, config):
        super().__init__(config)
        if self.recognition_head_type != "label_branch_mlp":
            raise ValueError(
                "EVEFMVDV2MultiScaleMIL requires recognition_head_type=label_branch_mlp"
            )
        heads = int(config.get("multiscale_cross_attention_heads", 8))
        if self.hidden_dim % heads != 0:
            raise ValueError("multiscale_cross_attention_heads must divide hidden_dim")
        self.multiscale_top_k = int(config.get("multiscale_top_k", self.top_k))
        if self.multiscale_top_k <= 0:
            raise ValueError("multiscale_top_k must be positive")
        self.multiscale_label_queries = nn.Parameter(
            torch.empty(self.num_labels, self.hidden_dim)
        )
        self.multiscale_cross_attention = nn.MultiheadAttention(
            self.hidden_dim,
            heads,
            dropout=float(config.get("multiscale_cross_attention_dropout", 0.1)),
            batch_first=True,
        )
        self.multiscale_cross_norm = nn.LayerNorm(self.hidden_dim)
        branch_dropout = float(config.get("label_branch_dropout", config.get("dropout", 0.1)))
        self.multiscale_label_branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.hidden_dim * 4),
                    nn.Linear(self.hidden_dim * 4, self.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(branch_dropout),
                    nn.Linear(self.hidden_dim, 1),
                )
                for _ in range(self.num_labels)
            ]
        )
        nn.init.normal_(self.multiscale_label_queries, mean=0.0, std=0.02)

    def compute_label_logits_from_pools(
        self, z, h, attn_weights, phi, chunk_mask, global_h
    ):
        batch_size = h.shape[0]
        queries = self.multiscale_label_queries.unsqueeze(0).expand(
            batch_size, -1, -1
        ).to(dtype=h.dtype)
        cross, _ = self.multiscale_cross_attention(
            queries,
            h,
            h,
            key_padding_mask=(~chunk_mask).contiguous(),
            need_weights=False,
        )
        cross = self.multiscale_cross_norm(cross + queries)

        top_k = min(self.multiscale_top_k, h.shape[1])
        masked_phi = phi.masked_fill(~chunk_mask.unsqueeze(-1), -1e9)
        _, top_indices = torch.topk(masked_phi, k=top_k, dim=1)
        expanded_h = h.unsqueeze(2).expand(-1, -1, self.num_labels, -1)
        gather_index = top_indices.unsqueeze(-1).expand(-1, -1, -1, self.hidden_dim)
        top_h = torch.gather(expanded_h, 1, gather_index).mean(dim=1)
        global_for_labels = global_h.unsqueeze(1).expand(-1, self.num_labels, -1)
        combined = torch.cat([z, cross, top_h, global_for_labels], dim=-1)
        logits = [
            branch(combined[:, label_idx, :]).squeeze(-1)
            for label_idx, branch in enumerate(self.multiscale_label_branches)
        ]
        return torch.stack(logits, dim=1)
