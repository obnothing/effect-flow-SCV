"""EVM-BERT with read-only vulnerability probes and label-wise MIL."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertForMaskedLM


def masked_logmeanexp(values, mask, tau=1.0, dim=1):
    if tau <= 0:
        raise ValueError("tau must be positive")
    mask = mask.bool()
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    count = mask.sum(dim=dim).clamp_min(1).to(values.dtype)
    scaled = (values / tau).masked_fill(~mask, torch.finfo(values.dtype).min)
    return tau * (torch.logsumexp(scaled, dim=dim) - count.log())


class VulProbeModel(nn.Module):
    def __init__(self, backbone_path, variant="b2_label_probe", num_labels=6,
                 tau=1.0, num_heads=8, scorer="shared", encoder_chunk_batch=4,
                 interaction_dim=128):
        super().__init__()
        self.variant = str(variant)
        self.num_labels = int(num_labels)
        self.tau = float(tau)
        self.encoder_chunk_batch = int(encoder_chunk_batch)
        mlm = BertForMaskedLM.from_pretrained(backbone_path, local_files_only=True)
        self.encoder = mlm.bert
        self.hidden_dim = int(self.encoder.config.hidden_size)
        if self.hidden_dim % int(num_heads):
            raise ValueError("hidden size must be divisible by num_heads")
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_dim // self.num_heads
        self.scorer_type = str(scorer)
        self.training_stage = "frozen"
        if self.variant == "b0_shared_representation":
            self.baseline_head = nn.Linear(self.hidden_dim, self.num_labels)
        else:
            query_count = 1 if self.variant == "b1_shared_probe" else self.num_labels
            self.probes = nn.Parameter(torch.empty(query_count, self.hidden_dim))
            self.query_projection = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
            self.key_projection = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
            self.value_projection = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
            if self.variant == "b3_probe_interaction":
                self.interaction_down = nn.Linear(self.hidden_dim, int(interaction_dim), bias=False)
                self.interaction = nn.MultiheadAttention(int(interaction_dim), 4, batch_first=True)
                self.interaction_up = nn.Linear(int(interaction_dim), self.hidden_dim, bias=False)
            if self.scorer_type == "shared":
                self.shared_scorer = nn.Linear(self.hidden_dim, 1, bias=False)
                self.label_bias = nn.Parameter(torch.zeros(self.num_labels))
            elif self.scorer_type == "label_specific":
                self.label_scorer = nn.Parameter(torch.empty(self.num_labels, self.hidden_dim))
                self.label_bias = nn.Parameter(torch.zeros(self.num_labels))
                nn.init.xavier_uniform_(self.label_scorer)
            else:
                raise ValueError("scorer must be shared or label_specific")
            nn.init.normal_(self.probes, mean=0.0, std=0.02)

    def set_training_stage(self, stage, unfreeze_last_layers=2):
        self.training_stage = str(stage)
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        if stage == "joint":
            layers = self.encoder.encoder.layer
            for layer in layers[-int(unfreeze_last_layers):]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True
            for parameter in self.encoder.pooler.parameters() if self.encoder.pooler is not None else []:
                parameter.requires_grad = False
        elif stage != "frozen":
            raise ValueError("stage must be frozen or joint")

    def train(self, mode=True):
        super().train(mode)
        if mode and self.training_stage == "frozen":
            self.encoder.eval()
        return self

    def _encode(self, flat_ids, flat_attention):
        outputs = []
        for left in range(0, flat_ids.shape[0], self.encoder_chunk_batch):
            right = left + self.encoder_chunk_batch
            outputs.append(self.encoder(
                input_ids=flat_ids[left:right],
                attention_mask=flat_attention[left:right],
                token_type_ids=torch.zeros_like(flat_ids[left:right]),
                return_dict=True,
            ).last_hidden_state)
        return torch.cat(outputs, dim=0)

    def _queries(self):
        probes = self.probes
        if self.variant == "b3_probe_interaction":
            low = self.interaction_down(probes).unsqueeze(0)
            interacted, _ = self.interaction(low, low, low, need_weights=False)
            probes = probes + self.interaction_up(interacted.squeeze(0))
        if probes.shape[0] == 1:
            probes = probes.expand(self.num_labels, -1)
        return self.query_projection(probes)

    def _probe(self, hidden, token_mask):
        chunks, tokens, _ = hidden.shape
        keys = self.key_projection(hidden).view(chunks, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.value_projection(hidden).view(chunks, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = self._queries().view(self.num_labels, self.num_heads, self.head_dim).permute(1, 0, 2)
        scores = torch.einsum("hld,nhtd->nhlt", queries, keys) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~token_mask[:, None, None, :], torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        representation = torch.einsum("nhlt,nhtd->nhld", attention, values)
        representation = representation.permute(0, 2, 1, 3).reshape(chunks, self.num_labels, self.hidden_dim)
        return representation, attention.mean(dim=1)

    def _score(self, representation):
        if self.scorer_type == "shared":
            return self.shared_scorer(representation).squeeze(-1) + self.label_bias
        return torch.einsum("nlh,lh->nl", representation, self.label_scorer) + self.label_bias

    def forward(self, input_ids, attention_mask, content_mask, chunk_mask,
                probe_permutation=None):
        batch, chunks, tokens = input_ids.shape
        valid = chunk_mask.reshape(-1)
        flat_ids = input_ids.reshape(-1, tokens)[valid]
        flat_attention = attention_mask.reshape(-1, tokens)[valid]
        flat_content = content_mask.reshape(-1, tokens)[valid].clone()
        empty = ~flat_content.any(dim=1)
        if empty.any():
            flat_content[empty, 0] = True
        hidden = self._encode(flat_ids, flat_attention)
        if self.variant == "b0_shared_representation":
            mask = flat_content.unsqueeze(-1).to(hidden.dtype)
            representation = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            valid_logits = self.baseline_head(representation)
            token_attention = flat_content.float() / flat_content.sum(dim=1, keepdim=True).clamp_min(1)
            token_attention = token_attention.unsqueeze(1).expand(-1, self.num_labels, -1)
            valid_representation = representation.unsqueeze(1).expand(-1, self.num_labels, -1)
        else:
            valid_representation, token_attention = self._probe(hidden, flat_content)
            if probe_permutation is not None:
                permutation = torch.as_tensor(probe_permutation, device=hidden.device, dtype=torch.long)
                valid_representation = valid_representation.index_select(1, permutation)
                token_attention = token_attention.index_select(1, permutation)
            valid_logits = self._score(valid_representation)
        chunk_logits = valid_logits.new_full((batch * chunks, self.num_labels), -1e4)
        chunk_logits[valid] = valid_logits
        chunk_logits = chunk_logits.view(batch, chunks, self.num_labels)
        contract_logits = masked_logmeanexp(chunk_logits, chunk_mask, self.tau, dim=1)
        scaled = (chunk_logits / self.tau).masked_fill(~chunk_mask.unsqueeze(-1), torch.finfo(chunk_logits.dtype).min)
        chunk_weights = torch.softmax(scaled, dim=1)
        return {
            "logits": contract_logits,
            "chunk_logits": chunk_logits,
            "chunk_weights": chunk_weights,
            "valid_token_attention": token_attention,
            "valid_probe_representation": valid_representation,
            "valid_chunk_flat_mask": valid,
        }
