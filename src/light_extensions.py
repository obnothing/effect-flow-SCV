"""Mutually exclusive topology-guided extensions of the light B2 model."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from light_label_model import LabelGuidedOpcodeNet


def inverse_softplus(value):
    value = float(value)
    return math.log(math.expm1(value))


class LightExtensionNet(LabelGuidedOpcodeNet):
    def __init__(self, variant, *args, local_radius=8, erase_mass=0.30, propagation_k=4,
                 segment_kappa=1.0, segment_gap=2, segment_min_len=2, segment_max=8,
                 lambda_consistency=0.01, confounder_clusters=16,
                 propagation_chunk_size=512, **kwargs):
        super().__init__("b2_label_attention", *args, **kwargs)
        self.extension = str(variant)
        self.local_radius = int(local_radius)
        self.erase_mass = float(erase_mass)
        self.propagation_k = int(propagation_k)
        self.propagation_chunk_size = int(propagation_chunk_size)
        self.segment_kappa = float(segment_kappa)
        self.segment_gap = int(segment_gap)
        self.segment_min_len = int(segment_min_len)
        self.segment_max = int(segment_max)
        if self.extension == "e1_vcfm":
            self.complement_beta = nn.Parameter(torch.tensor(0.1))
        elif self.extension == "e2_vrop":
            self.propagation_gamma = nn.Parameter(torch.tensor(0.1))
            self.propagation_norm = nn.LayerNorm(self.output_dim)
        elif self.extension == "e3_vasm":
            self.segment_attention = nn.MultiheadAttention(self.output_dim, 4, batch_first=True)
            self.segment_projection = nn.Linear(self.output_dim, self.output_dim, bias=False)
            self.segment_gamma = nn.Parameter(torch.tensor(0.1))
        elif self.extension == "e4_pgvr":
            self.raw_sigma = nn.Parameter(torch.full((self.num_labels,), inverse_softplus(8.0)))
            self.raw_eta = nn.Parameter(torch.full((self.num_labels,), inverse_softplus(0.1)))
            self.consistency_lambda = float(lambda_consistency)
        elif self.extension == "e5_tdvp":
            self.register_buffer("confounder_dictionary", torch.zeros(int(confounder_clusters), self.output_dim))
            self.conf_q = nn.Linear(self.output_dim, 128, bias=False)
            self.conf_k = nn.Linear(self.output_dim, 128, bias=False)
            self.conf_fuse = nn.Linear(self.output_dim * 2, self.output_dim)
        else:
            raise ValueError(f"unknown extension: {variant}")

    def set_confounder_dictionary(self, dictionary):
        value = dictionary.detach().float()
        self._buffers["confounder_dictionary"] = value

    def _base(self, hidden, mask, query_permutation=None):
        projected = self.attention_projection(hidden)
        queries = self.label_queries
        if query_permutation is not None:
            queries = queries.index_select(0, torch.as_tensor(query_permutation, device=queries.device))
        scores = torch.einsum("bth,lh->blt", projected, queries) / math.sqrt(self.output_dim)
        attention = self.masked_softmax(scores, mask.unsqueeze(1))
        representation = torch.einsum("blt,bth->blh", attention, hidden)
        return scores, attention, representation

    def _vcfm(self, hidden, scores, attention, representation, mask):
        order = attention.argsort(dim=-1, descending=True)
        ranked = attention.gather(-1, order)
        ranked_mask = mask.unsqueeze(1).expand_as(attention).gather(-1, order)
        previous_mass = ranked.cumsum(-1) - ranked
        erase_ranked = ranked_mask & (previous_mass < self.erase_mass)
        erase = torch.zeros_like(erase_ranked)
        erase.scatter_(-1, order, erase_ranked)
        remaining = mask.unsqueeze(1) & ~erase
        second = self.masked_softmax(scores, remaining)
        complement = torch.einsum("blt,bth->blh", second, hidden)
        return representation + self.complement_beta * complement, {"complement_attention": second, "complement_beta": self.complement_beta.detach()}

    def _vrop(self, hidden, attention, representation, mask):
        k = min(self.propagation_k, hidden.shape[1])
        ranked_attention = attention.detach().masked_fill(~mask.unsqueeze(1), torch.finfo(attention.dtype).min)
        indices = ranked_attention.topk(k, dim=-1).indices
        h_norm = F.normalize(hidden, dim=-1)
        refined_representations = []
        refined_attentions = []
        chunk_size = max(1, self.propagation_chunk_size)

        def refine_chunk(left, right, anchors):
            similarity = torch.einsum("bkd,btd->btk", F.normalize(anchors, dim=-1), h_norm[:, left:right])
            propagated = torch.einsum("btk,bkd->btd", torch.softmax(similarity / 0.2, dim=-1), anchors)
            return hidden[:, left:right] + self.propagation_gamma * self.propagation_norm(propagated)

        # Keep the [B,T,H] propagation workspace label-local. Materializing
        # [B,L,T,K,H] is unnecessary and exceeds small-GPU memory on long inputs.
        for label_id in range(self.num_labels):
            anchors = hidden.gather(1, indices[:, label_id].unsqueeze(-1).expand(-1, -1, self.output_dim))
            def run_label(query, label_anchors):
                score_parts = []
                for left in range(0, hidden.shape[1], chunk_size):
                    right = min(left + chunk_size, hidden.shape[1])
                    refined_chunk = refine_chunk(left, right, label_anchors)
                    score_parts.append(torch.einsum("bth,h->bt", refined_chunk, query) / math.sqrt(self.output_dim))
                refined_scores = torch.cat(score_parts, dim=1)
                label_attention = self.masked_softmax(refined_scores, mask)
                representation_parts = []
                for left in range(0, hidden.shape[1], chunk_size):
                    right = min(left + chunk_size, hidden.shape[1])
                    refined_chunk = refine_chunk(left, right, label_anchors)
                    representation_parts.append(torch.einsum("bt,bth->bh", label_attention[:, left:right], refined_chunk))
                return torch.stack(representation_parts, dim=0).sum(0), label_attention

            if torch.is_grad_enabled():
                label_representation, label_attention = checkpoint(
                    run_label, self.label_queries[label_id], anchors, use_reentrant=False
                )
            else:
                label_representation, label_attention = run_label(self.label_queries[label_id], anchors)
            refined_representations.append(label_representation)
            refined_attentions.append(label_attention)
        refined_representation = torch.stack(refined_representations, dim=1)
        refined_attention = torch.stack(refined_attentions, dim=1)
        return refined_representation, {"representative_indices": indices, "propagation_gamma": self.propagation_gamma.detach(), "refined_attention": refined_attention}

    def _segments(self, hidden, attention, mask):
        batch, tokens, dim = hidden.shape
        result = hidden.new_zeros(batch, self.num_labels, dim)
        counts = []
        lengths = []
        coverage = []
        for row in range(batch):
            row_counts = []
            row_lengths = []
            row_coverage = []
            for label in range(self.num_labels):
                valid = int(mask[row].sum())
                threshold = attention[row, label, :valid].mean() + self.segment_kappa * attention[row, label, :valid].std(unbiased=False)
                selected = (attention[row, label, :valid] >= threshold).tolist()
                runs = []
                start = None
                for index, active in enumerate(selected + [False]):
                    if active and start is None: start = index
                    if not active and start is not None:
                        runs.append((start, index))
                        start = None
                segments = []
                for left, right in runs:
                    if segments and left - segments[-1][1] <= self.segment_gap:
                        segments[-1] = (segments[-1][0], right)
                    else:
                        segments.append((left, right))
                segments = [(left, right) for left, right in segments if right - left >= self.segment_min_len]
                segment_masses = [float(attention[row, label, left:right].sum().detach()) for left, right in segments]
                if len(segments) > self.segment_max:
                    keep = sorted(range(len(segment_masses)), key=segment_masses.__getitem__, reverse=True)[:self.segment_max]
                    segments = [segments[index] for index in sorted(keep)]
                row_counts.append(len(segments))
                row_lengths.append([right - left for left, right in segments])
                row_coverage.append(float(sum(right - left for left, right in segments) / max(valid, 1)))
                if not segments:
                    continue
                values = torch.stack([(attention[row, label, left:right].unsqueeze(-1) * hidden[row, left:right]).sum(0) / attention[row, label, left:right].sum().clamp_min(1e-12) for left, right in segments])
                interacted, _ = self.segment_attention(values.unsqueeze(0), values.unsqueeze(0), values.unsqueeze(0), need_weights=False)
                weights = torch.softmax(self.segment_projection(interacted.squeeze(0)) @ self.label_queries[label], dim=0)
                result[row, label] = (weights.unsqueeze(-1) * interacted.squeeze(0)).sum(0)
            counts.append(row_counts)
            lengths.append(row_lengths)
            coverage.append(row_coverage)
        return result, {"segment_counts": counts, "segment_lengths": lengths, "segment_coverage": coverage,
                        "segment_gamma": self.segment_gamma.detach()}

    def _pgvr(self, hidden, attention, representation, mask):
        positions = torch.arange(hidden.shape[1], device=hidden.device).view(1, 1, -1)
        peak = attention.argmax(dim=-1).unsqueeze(-1)
        sigma = F.softplus(self.raw_sigma).view(1, self.num_labels, 1)
        prior = torch.exp(-((positions - peak).float() ** 2) / (2 * sigma.pow(2)))
        prior = prior * mask.unsqueeze(1).to(prior.dtype)
        prior = prior / prior.sum(-1, keepdim=True).clamp_min(1e-12)
        eta = F.softplus(self.raw_eta).view(1, self.num_labels, 1)
        fused_attention = attention + eta * prior
        fused_attention = fused_attention / fused_attention.sum(-1, keepdim=True).clamp_min(1e-12)
        refined = torch.einsum("blt,bth->blh", fused_attention, hidden)
        valid = mask.unsqueeze(1).to(attention.dtype)
        consistency = ((attention - fused_attention.detach()).pow(2) * valid).sum() / (valid.sum() * attention.shape[1]).clamp_min(1.0)
        return refined, {"prior_attention": prior, "fused_attention": fused_attention, "sigma": sigma.detach().view(-1), "eta": eta.detach().view(-1), "consistency_loss": consistency}

    def _tdvp(self, representation):
        dictionary = self.confounder_dictionary.to(representation.device, representation.dtype)
        q = self.conf_q(representation); k = self.conf_k(dictionary)
        scores = torch.einsum("blh,kh->blk", q, k) / math.sqrt(k.shape[-1])
        weights = torch.softmax(scores, dim=-1)
        context = torch.einsum("blk,kh->blh", weights, dictionary)
        return self.conf_fuse(torch.cat([representation, context], dim=-1)), {"context_attention": weights, "context": context, "context_missing": False}

    def forward(self, input_ids, lengths, mask, query_permutation=None):
        hidden = self.encode(input_ids, lengths)
        scores, attention, representation = self._base(hidden, mask, query_permutation)
        base_representation = representation
        diagnostics = {"attention": attention, "representations": representation}
        if self.extension == "e1_vcfm":
            representation, extra = self._vcfm(hidden, scores, attention, representation, mask)
        elif self.extension == "e2_vrop":
            representation, extra = self._vrop(hidden, attention, representation, mask)
        elif self.extension == "e3_vasm":
            segment, extra = self._segments(hidden, attention, mask)
            representation = representation + self.segment_gamma * segment
        elif self.extension == "e4_pgvr":
            representation, extra = self._pgvr(hidden, attention, representation, mask)
        else:
            representation, extra = self._tdvp(representation)
        logits = torch.einsum("blh,lh->bl", representation, self.label_scorer) + self.label_bias
        diagnostics.update(extra)
        diagnostics["base_attention"] = attention
        diagnostics["base_representations"] = base_representation
        diagnostics["representations"] = representation
        diagnostics["logits"] = logits
        return diagnostics
