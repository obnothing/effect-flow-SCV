"""B2-compatible models for the isolated TDVP controls and candidate."""

import math

import torch
import torch.nn as nn

from light_label_model import LabelGuidedOpcodeNet


class E5TDVPModel(LabelGuidedOpcodeNet):
    """B2 classifier with a frozen, label-agnostic context dictionary.

    ``mode`` is one of ``d1_param_control``, ``d2_global_mean``,
    ``d3_shuffled_dictionary`` or ``d4_tdvp``. All four modes retain the same
    context projection and fusion parameters; only the context source differs.
    """

    def __init__(self, mode, *args, dictionary_size=16, joint_dim=128, **kwargs):
        super().__init__("b2_label_attention", *args, **kwargs)
        self.mode = str(mode)
        self.dictionary_size = int(dictionary_size)
        self.joint_dim = int(joint_dim)
        self.register_buffer("context_dictionary", torch.zeros(self.dictionary_size, self.output_dim))
        self.register_buffer("global_context", torch.zeros(self.output_dim))
        self.conf_q = nn.Linear(self.output_dim, self.joint_dim, bias=False)
        self.conf_k = nn.Linear(self.output_dim, self.joint_dim, bias=False)
        self.conf_fuse = nn.Linear(self.output_dim * 2, self.output_dim)
        if self.mode not in {"d1_param_control", "d2_global_mean", "d3_shuffled_dictionary", "d4_tdvp"}:
            raise ValueError(f"unknown TDVP mode: {mode}")

    @torch.no_grad()
    def set_dictionary(self, dictionary):
        value = dictionary.detach().float()
        if value.shape != self.context_dictionary.shape:
            raise ValueError(f"expected dictionary {tuple(self.context_dictionary.shape)}, got {tuple(value.shape)}")
        self.context_dictionary.copy_(value)

    @torch.no_grad()
    def set_global_context(self, value):
        value = value.detach().float().view(-1)
        if value.shape != self.global_context.shape:
            raise ValueError(f"expected global context {tuple(self.global_context.shape)}, got {tuple(value.shape)}")
        self.global_context.copy_(value)

    def _base(self, hidden, mask):
        projected = self.attention_projection(hidden)
        scores = torch.einsum("bth,lh->blt", projected, self.label_queries) / math.sqrt(self.output_dim)
        attention = self.masked_softmax(scores, mask.unsqueeze(1))
        representation = torch.einsum("blt,bth->blh", attention, hidden)
        return attention, representation

    def _context(self, representation):
        batch = representation.shape[0]
        if self.mode == "d1_param_control":
            context = torch.zeros_like(representation)
            weights = representation.new_zeros(batch, self.num_labels, self.dictionary_size)
            return context, weights
        if self.mode == "d2_global_mean":
            context = self.global_context.to(representation.device, representation.dtype).view(1, 1, -1)
            context = context.expand(batch, self.num_labels, -1)
            weights = representation.new_full((batch, self.num_labels, self.dictionary_size), 1.0 / self.dictionary_size)
            return context, weights
        dictionary = self.context_dictionary.to(representation.device, representation.dtype)
        query = self.conf_q(representation)
        keys = self.conf_k(dictionary)
        scores = torch.einsum("blh,kh->blk", query, keys) / math.sqrt(self.joint_dim)
        weights = torch.softmax(scores, dim=-1)
        context = torch.einsum("blk,kh->blh", weights, dictionary)
        return context, weights

    def forward(self, input_ids, lengths, mask, context_zero=False,
                context_batch_permutation=None, centroid_permutation=None):
        hidden = self.encode(input_ids, lengths)
        attention, representation = self._base(hidden, mask)
        context, context_weights = self._context(representation)
        if centroid_permutation is not None and self.mode in {"d3_shuffled_dictionary", "d4_tdvp"}:
            dictionary = self.context_dictionary.to(representation.device, representation.dtype)
            dictionary = dictionary.index_select(0, torch.as_tensor(centroid_permutation, device=representation.device))
            query = self.conf_q(representation)
            keys = self.conf_k(dictionary)
            scores = torch.einsum("blh,kh->blk", query, keys) / math.sqrt(self.joint_dim)
            context_weights = torch.softmax(scores, dim=-1)
            context = torch.einsum("blk,kh->blh", context_weights, dictionary)
        if context_batch_permutation is not None:
            permutation = torch.as_tensor(context_batch_permutation, device=representation.device, dtype=torch.long)
            context = context.index_select(0, permutation)
        if context_zero:
            context = torch.zeros_like(context)
        adjusted = self.conf_fuse(torch.cat([representation, context], dim=-1))
        logits = torch.einsum("blh,lh->bl", adjusted, self.label_scorer) + self.label_bias
        return {
            "logits": logits,
            "attention": attention,
            "base_attention": attention,
            "representations": adjusted,
            "base_representations": representation,
            "context": context,
            "context_attention": context_weights,
        }

