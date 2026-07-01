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
        self.recognition_loss_type = config.get("recognition_loss_type", "bce")
        if self.recognition_loss_type not in {"bce", "asl"}:
            raise ValueError(
                f"Unsupported recognition_loss_type: {self.recognition_loss_type}"
            )
        self.asl_gamma_neg = float(config.get("asl_gamma_neg", 4.0))
        self.asl_gamma_pos = float(config.get("asl_gamma_pos", 0.0))
        self.asl_clip = float(config.get("asl_clip", 0.05))
        self.asl_eps = float(config.get("asl_eps", 1e-8))

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

    def compute_recognition_loss(self, recognition_logits, multi_labels):
        if self.recognition_loss_type == "asl":
            return asymmetric_multilabel_loss(
                recognition_logits,
                multi_labels.float(),
                gamma_neg=self.asl_gamma_neg,
                gamma_pos=self.asl_gamma_pos,
                clip=self.asl_clip,
                eps=self.asl_eps,
            )
        if self.recognition_pos_weight is None:
            return F.binary_cross_entropy_with_logits(
                recognition_logits,
                multi_labels.float(),
            )
        return F.binary_cross_entropy_with_logits(
            recognition_logits,
            multi_labels.float(),
            pos_weight=self.recognition_pos_weight,
        )

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
            recognition_loss = self.compute_recognition_loss(
                recognition_logits,
                multi_labels,
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
        detection_logits = self.detection_classifier(global_h).squeeze(-1)

        loss = None
        detection_loss = None
        recognition_loss = None
        evidence_align_loss = None
        template_consistency_loss = None
        if binary_label is not None and multi_labels is not None:
            detection_loss = self.detection_loss_fn(detection_logits, binary_label.float())
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
                ((detection_loss + recognition_loss) / 2)
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
        self.current_epoch = 10**9

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

    def _semantic_dropout(self, values):
        if not self.training or self.semantic_dropout_p <= 0:
            return values
        keep = torch.rand_like(values) >= self.semantic_dropout_p
        return values * keep.type_as(values)

    def encode_strong_chunks(self, chunk_features, chunk_mask):
        if self.use_chunk_context:
            return self.chunk_context_encoder(chunk_features, chunk_mask)
        return self.chunk_projection(chunk_features)

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

    def gate_values(self):
        return {
            "beta_reliable": F.softplus(self.beta_reliable_raw).detach(),
            "gamma_reliable": F.softplus(self.gamma_reliable_raw).detach(),
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
        phi = evidence["final_evidence_scores"].to(dtype=h.dtype)

        v = torch.tanh(self.attn_v(h))
        u = torch.sigmoid(self.attn_u(h))
        gated = v * u
        neural_attn_logits = torch.einsum("bca,ka->bck", gated, self.label_attn)
        beta = F.softplus(self.beta_reliable_raw).to(dtype=h.dtype)
        gamma = F.softplus(self.gamma_reliable_raw).to(dtype=h.dtype)
        if not self.semantic_enabled():
            beta = torch.zeros_like(beta)
            gamma = torch.zeros_like(gamma)
        attn_logits = neural_attn_logits + beta.view(1, 1, -1) * phi
        attn_logits = attn_logits.masked_fill(~chunk_mask.unsqueeze(-1), -1e9)
        attn_weights = torch.softmax(attn_logits, dim=1)
        z = torch.einsum("bck,bch->bkh", attn_weights, h)
        neural_logits = (
            z * self.label_out.unsqueeze(0)
        ).sum(dim=-1) + self.label_bias

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
            evidence_features * self.evidence_logit_weight.unsqueeze(0).to(dtype=h.dtype)
        ).sum(dim=-1) + self.evidence_logit_bias.unsqueeze(0).to(dtype=h.dtype)
        recognition_logits = neural_logits + gamma.view(1, -1) * evidence_logits

        global_h = self.masked_mean(h, chunk_mask)
        detection_logits = self.detection_classifier(global_h).squeeze(-1)
        chunk_scores = torch.sigmoid(
            attn_logits.masked_fill(~chunk_mask.unsqueeze(-1), -30.0)
        ).masked_fill(~chunk_mask.unsqueeze(-1), 0.0)

        loss = None
        detection_loss = None
        recognition_loss = None
        gate_regularization_loss = None
        if binary_label is not None and multi_labels is not None:
            detection_loss = self.detection_loss_fn(
                detection_logits,
                binary_label.float(),
            )
            recognition_loss = self.compute_recognition_loss(
                recognition_logits,
                multi_labels,
            )
            active_beta = F.softplus(self.beta_reliable_raw)
            active_gamma = F.softplus(self.gamma_reliable_raw)
            gate_regularization_loss = (
                active_beta.pow(2).mean() + active_gamma.pow(2).mean()
            )
            loss = (
                (detection_loss + recognition_loss) / 2
                + self.lambda_gate * gate_regularization_loss
            ).reshape(1)

        return {
            "loss": loss,
            "detection_loss": detection_loss,
            "recognition_loss": recognition_loss,
            "gate_regularization_loss": gate_regularization_loss,
            "detection_logits": detection_logits,
            "recognition_logits": recognition_logits,
            "chunk_logits": attn_logits,
            "chunk_scores": chunk_scores,
            "attn_weights": attn_weights,
            "neural_attention_logits": neural_attn_logits,
            "neural_logits": neural_logits,
            "evidence_logits": evidence_logits,
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
