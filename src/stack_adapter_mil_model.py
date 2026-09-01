"""Stack-Adapter encoder with the existing eight-view MIL head."""

from __future__ import annotations

import torch.nn as nn

from evm_chunk_mil_model import MLM8ViewMultiSlotMIL
from stack_aware_mil_model import pool_eight
from stack_adapter_bert import StackAdapterBertForMaskedLM


class StackAdapterMLM8MIL(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = StackAdapterBertForMaskedLM.from_pretrained(
            config["hf_model_path"],
            local_files_only=True,
            stack_hidden_dim=int(config.get("stack_adapter_hidden_dim", 256)),
            stack_layers=int(config.get("stack_adapter_layers", 1)),
            lora_rank=int(config.get("lora_rank", 8)),
            lora_alpha=float(config.get("lora_alpha", 16.0)),
            lora_dropout=float(config.get("lora_dropout", 0.05)),
            fusion_init=float(config.get("fusion_init", 0.05)),
        )
        self.encoder.freeze_base()
        self.chunk_microbatch = int(config.get("encoder_chunk_microbatch", 1))
        mil_config = dict(config)
        mil_config["feature_dim"] = int(self.encoder.config.hidden_size)
        mil_config["num_views"] = 8
        self.mil = MLM8ViewMultiSlotMIL(mil_config)

    def _encode(self, batch):
        ids, mask = batch["input_ids"], batch["attention_mask"]
        total, pooled = ids.shape[0], []
        for left in range(0, total, self.chunk_microbatch):
            right = min(total, left + self.chunk_microbatch)
            edge_left = int(batch["edge_offsets"][left])
            edge_right = int(batch["edge_offsets"][right])
            offsets = batch["edge_offsets"][left:right + 1] - edge_left
            output = self.encoder(
                input_ids=ids[left:right], attention_mask=mask[left:right],
                stack_state=batch["stack_state"][left:right],
                boundary_state=batch["boundary_state"][left:right],
                edge_offsets=offsets,
                edge_src=batch["edge_src"][edge_left:edge_right],
                edge_dst=batch["edge_dst"][edge_left:edge_right],
                edge_type=batch["edge_type"][edge_left:edge_right],
                edge_slot=batch["edge_slot"][edge_left:edge_right],
                edge_distance=batch["edge_distance"][edge_left:edge_right],
                edge_confidence=batch["edge_confidence"][edge_left:edge_right],
            )
            pooled.append(pool_eight(output.last_hidden_state, mask[left:right]))
        return torch.cat(pooled, dim=0)

    def forward(self, batch, return_attention=False):
        features = self._encode(batch)
        offsets = batch["sample_chunk_offsets"].tolist()
        batch_size = len(offsets) - 1
        max_chunks = max(offsets[index + 1] - offsets[index] for index in range(batch_size))
        padded = features.new_zeros((batch_size, max_chunks, 8, features.shape[-1]))
        chunk_mask = torch.zeros((batch_size, max_chunks), dtype=torch.bool, device=features.device)
        for index in range(batch_size):
            left, right = offsets[index], offsets[index + 1]
            padded[index, :right - left] = features[left:right]
            chunk_mask[index, :right - left] = True
        return self.mil(padded, chunk_mask, multi_labels=None, return_attention=return_attention)
