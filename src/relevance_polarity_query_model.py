"""Aligned relevance-polarity queries over the frozen P11 opcode backbone design."""

import math

import torch
from torch import nn

from light_label_model import LabelGuidedOpcodeNet, validate_model_config
from polarity_query_model import SharedMultiHeadCrossAttention


VARIANTS = ("B1", "B2")


class RelevancePolarityQueryNet(LabelGuidedOpcodeNet):
    """Locate vulnerability-relevant tokens, then discriminate polarity."""

    def __init__(self, variant, vocab_size, pad_id, embedding_dim=128, hidden=384,
                 labels=6, heads=4, bidirectional=True, gru_layers=1):
        if variant not in VARIANTS:
            raise ValueError(f"unknown relevance-polarity variant: {variant}")
        super().__init__("b0_mean", vocab_size, pad_id, embedding_dim, hidden, labels,
                         bidirectional, gru_layers=gru_layers)
        del self.classifier
        self.variant = variant
        self.polarity_queries = nn.Parameter(torch.empty(labels, 2, self.output_dim))
        nn.init.normal_(self.polarity_queries, std=0.02)
        self.cross_attention = SharedMultiHeadCrossAttention(self.output_dim, heads)
        self.label_scorer = nn.Parameter(torch.empty(labels, self.output_dim))
        nn.init.xavier_uniform_(self.label_scorer)
        self.branch_bias = nn.Parameter(torch.zeros(labels, 2))
        # Register the B2-only parameters after all common modules so B1 and
        # B2 receive bit-identical initialization for every shared parameter.
        if variant == "B2":
            self.relevance_queries = nn.Parameter(torch.empty(labels, self.output_dim))
            nn.init.normal_(self.relevance_queries, std=0.02)
        else:
            self.register_parameter("relevance_queries", None)

    def _project_heads(self, values):
        heads = self.cross_attention.num_heads
        head_dim = self.cross_attention.head_dim
        return values.view(*values.shape[:-1], heads, head_dim)

    def _locator_queries(self, relevance_permutation=None):
        locator = self.relevance_queries if self.variant == "B2" else self.polarity_queries[:, 0]
        if relevance_permutation is not None:
            locator = locator[relevance_permutation]
        return locator

    def score(self, representations, score_mode="full"):
        energies = torch.einsum("blpd,ld->blp", representations, self.label_scorer) + self.branch_bias
        if score_mode == "full":
            logits = energies[..., 0] - energies[..., 1]
        elif score_mode == "positive_only":
            logits = energies[..., 0]
        elif score_mode == "negative_only":
            logits = -energies[..., 1]
        else:
            raise ValueError(f"unknown score mode: {score_mode}")
        return logits, energies

    def forward(self, input_ids, lengths, mask, diagnostics=False,
                relevance_permutation=None, swap_polarities=False,
                score_mode="full"):
        if mask.dtype != torch.bool or not bool(mask.any(1).all()):
            raise ValueError("Each sample needs a nonempty boolean token mask")
        hidden = self.encode(input_ids, lengths)
        batch, tokens, _ = hidden.shape
        heads = self.cross_attention.num_heads
        head_dim = self.cross_attention.head_dim

        locator = self._locator_queries(relevance_permutation)
        q_rel = self._project_heads(self.cross_attention.q_proj(locator))
        keys = self.cross_attention.k_proj(hidden).view(batch, tokens, heads, head_dim).permute(0, 2, 1, 3)
        values = self.cross_attention.v_proj(hidden).view(batch, tokens, heads, head_dim).permute(0, 2, 1, 3)
        relevance_scores = torch.einsum("lhd,bhtd->blht", q_rel, keys) / math.sqrt(head_dim)
        relevance_scores = relevance_scores.masked_fill(~mask[:, None, None, :],
                                                        torch.finfo(relevance_scores.dtype).min)
        alpha_rel = torch.softmax(relevance_scores, dim=-1)
        z_rel = torch.einsum("blht,bhtd->blhd", alpha_rel, values)

        polarity_queries = self.polarity_queries.flip(1) if swap_polarities else self.polarity_queries
        projected_polarity = self._project_heads(self.cross_attention.q_proj(polarity_queries))
        gates = 1.0 + torch.tanh(projected_polarity)
        gated_heads = z_rel[:, :, None, :, :] * gates[None, :, :, :, :]
        joined = gated_heads.reshape(batch, self.num_labels, 2, self.output_dim)
        representations = self.cross_attention.out_proj(joined)
        logits, energies = self.score(representations, score_mode)
        result = {"logits": logits, "energies": energies,
                  "representations": representations}
        if diagnostics:
            result.update(alpha_rel=alpha_rel, z_rel=z_rel, gates=gates,
                          hidden_shape=tuple(hidden.shape))
        return result


def build_relevance_polarity_model(variant, config, vocab_size, pad_id):
    model = RelevancePolarityQueryNet(
        variant, vocab_size, pad_id, config["embedding_dim"],
        config["gru_hidden_size"], config["num_labels"],
        config["attention_heads"], config["bidirectional"], config["gru_layers"])
    validate_model_config(model, config)
    if model.cross_attention.num_heads != config["attention_heads"]:
        raise ValueError("attention heads mismatch")
    expected = (config["num_labels"], 2, 2 * config["gru_hidden_size"])
    if tuple(model.polarity_queries.shape) != expected:
        raise ValueError("polarity query shape mismatch")
    return model
