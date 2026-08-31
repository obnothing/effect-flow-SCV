"""Stack-aware token encoder followed by the existing MLM8 MIL head."""

from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path

from evm_chunk_mil_model import MLM8ViewMultiSlotMIL
from stack_aware_bert import StackAwareBertForMaskedLM


def pool_eight(hidden, mask):
    mask = mask.bool()
    valid = mask.clone()
    valid[:, 0] = False
    lengths = mask.sum(dim=1).long()
    sep = (lengths - 1).clamp_min(0)
    valid.scatter_(1, sep.unsqueeze(1), False)
    count = valid.sum(dim=1)
    weights = valid.unsqueeze(-1).to(hidden.dtype)
    mean = (hidden * weights).sum(dim=1) / count.clamp_min(1).unsqueeze(-1).to(hidden.dtype)
    maximum = hidden.masked_fill(~valid.unsqueeze(-1), torch.finfo(hidden.dtype).min).max(dim=1).values
    views = [hidden[:, 0], mean, maximum]
    positions = torch.arange(hidden.shape[1], device=hidden.device).view(1, -1)
    for segment in range(5):
        starts = torch.div(count * segment, 5, rounding_mode="floor") + 1
        ends = torch.div(count * (segment + 1), 5, rounding_mode="floor") + 1
        segment_mask = (positions >= starts.unsqueeze(1)) & (positions < ends.unsqueeze(1)) & valid
        segment_count = segment_mask.sum(dim=1)
        segment_mean = (hidden * segment_mask.unsqueeze(-1).to(hidden.dtype)).sum(dim=1) / segment_count.clamp_min(1).unsqueeze(-1).to(hidden.dtype)
        segment_mean[segment_count == 0] = mean[segment_count == 0]
        views.append(segment_mean)
    fallback = count == 0
    if fallback.any():
        for view in views:
            view[fallback] = hidden[fallback, 0]
    return torch.stack(views, dim=1)


class StackAwareMLM8MIL(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = StackAwareBertForMaskedLM.from_pretrained(config["hf_model_path"], local_files_only=True)
        pretrained_stack = config.get("stack_aware_mlm_checkpoint")
        if pretrained_stack:
            checkpoint_path = Path(pretrained_stack)
            if not checkpoint_path.is_absolute():
                checkpoint_path = Path(__file__).resolve().parents[1] / checkpoint_path
            try:
                payload = torch.load(checkpoint_path, map_location="cpu")
                # Newer checkpoints add per-head relation gates.  Loading an
                # older route-local MLM checkpoint remains valid because the
                # new gates keep their configured initialization.
                self.encoder.load_state_dict(payload["model_state_dict"], strict=False)
            except FileNotFoundError:
                # Downstream-only MVP can start from the frozen continued MLM
                # while the optional train-only stack-aware MLM is unavailable.
                pass
        self.chunk_microbatch = int(config.get("encoder_chunk_microbatch", 8))
        if bool(config.get("encoder_frozen", True)):
            for parameter in self.encoder.base.parameters():
                parameter.requires_grad = False
        mil_config = dict(config)
        mil_config["feature_dim"] = int(self.encoder.config.hidden_size)
        mil_config["num_views"] = 8
        self.mil = MLM8ViewMultiSlotMIL(mil_config)

    def freeze_encoder(self):
        """Freeze the complete Stack-Aware BERT for offline feature extraction."""
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        self.encoder.eval()

    def _encode(self, batch):
        ids = batch["input_ids"]
        mask = batch["attention_mask"]
        states = batch["stack_state"]
        boundary = batch["boundary_state"]
        total = ids.shape[0]
        pooled = []
        for left in range(0, total, self.chunk_microbatch):
            right = min(total, left + self.chunk_microbatch)
            edge_left = int(batch["edge_offsets"][left])
            edge_right = int(batch["edge_offsets"][right])
            offsets = batch["edge_offsets"][left:right + 1] - edge_left
            outputs = self.encoder(
                input_ids=ids[left:right], attention_mask=mask[left:right],
                stack_state=states[left:right], boundary_state=boundary[left:right],
                edge_offsets=offsets,
                edge_src=batch["edge_src"][edge_left:edge_right],
                edge_dst=batch["edge_dst"][edge_left:edge_right],
                edge_type=batch["edge_type"][edge_left:edge_right],
                edge_slot=batch["edge_slot"][edge_left:edge_right],
                edge_distance=batch["edge_distance"][edge_left:edge_right],
                edge_confidence=batch["edge_confidence"][edge_left:edge_right],
            )
            pooled.append(pool_eight(outputs.last_hidden_state, mask[left:right]))
        return torch.cat(pooled, dim=0)

    def forward(self, batch, return_attention=False):
        chunk_features = self._encode(batch)
        sample_offsets = batch["sample_chunk_offsets"].tolist()
        batch_size = len(sample_offsets) - 1
        max_chunks = max(sample_offsets[index + 1] - sample_offsets[index] for index in range(batch_size))
        padded = chunk_features.new_zeros((batch_size, max_chunks, 8, chunk_features.shape[-1]))
        chunk_mask = torch.zeros((batch_size, max_chunks), dtype=torch.bool, device=chunk_features.device)
        for index in range(batch_size):
            left, right = sample_offsets[index], sample_offsets[index + 1]
            count = right - left
            padded[index, :count] = chunk_features[left:right]
            chunk_mask[index, :count] = True
        output = self.mil(padded, chunk_mask, multi_labels=None, return_attention=return_attention)
        output["chunk_features"] = padded
        output["chunk_mask"] = chunk_mask
        return output
