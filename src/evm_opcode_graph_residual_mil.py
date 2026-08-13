"""Label-conditioned graph residual MIL for the Opcode-CSDG route."""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from evm_chunk_mil_model import MLM8ViewMultiSlotMIL


class RelationGraphLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_edge_types: int, dropout: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.source = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.destination = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.relation = nn.Embedding(num_edge_types, hidden_dim)
        self.score = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.message = nn.Linear(hidden_dim, hidden_dim)
        self.self_update = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _edge_softmax(scores, destinations, node_count):
        if scores.numel() == 0:
            return scores
        # Keep the reduction in fp32 under autocast. The resulting weights are
        # cast back so message aggregation has one consistent dtype.
        scores_float = scores.float()
        maximum = torch.full((node_count,), -torch.inf, device=scores.device, dtype=torch.float32)
        maximum.scatter_reduce_(0, destinations, scores_float, reduce="amax", include_self=True)
        weights = torch.exp(scores_float - maximum[destinations])
        denominator = torch.zeros(node_count, device=scores.device, dtype=torch.float32)
        denominator.index_add_(0, destinations, weights)
        return (weights / denominator[destinations].clamp_min(1e-8)).to(scores.dtype)

    def forward(self, node_features, edge_index, edge_type, node_mask, use_edges=True):
        if node_features.ndim != 2:
            raise ValueError("node_features must be [V,H]")
        valid = node_mask.bool()
        x = node_features.masked_fill(~valid.unsqueeze(1), 0.0)
        if use_edges and edge_index.numel():
            keep = valid[edge_index[0]] & valid[edge_index[1]]
            source_index = edge_index[0][keep]
            destination_index = edge_index[1][keep]
            relation = self.relation(edge_type[keep])
            source = self.source(x[source_index])
            destination = self.destination(x[destination_index])
            scores = self.score(torch.cat([source, destination, relation], dim=-1)).squeeze(-1)
            weights = self._edge_softmax(scores, destination_index, x.shape[0])
            messages = self.message(source + relation) * weights.unsqueeze(1)
            aggregate = torch.zeros_like(x)
            # Autocast may produce half-precision messages while x remains
            # float32; index_add_ requires both tensors to have one dtype.
            aggregate.index_add_(0, destination_index, messages.to(aggregate.dtype))
        else:
            aggregate = torch.zeros_like(x)
        updated = self.norm(x + self.dropout(F.gelu(self.self_update(x) + aggregate)))
        return updated.masked_fill(~valid.unsqueeze(1), 0.0)


class OpcodeGraphResidualMIL(nn.Module):
    """Frozen MLM8 sequence predictor plus a zero-initialized graph correction."""

    def __init__(self, config):
        super().__init__()
        self.num_labels = int(config.get("num_labels", 6))
        self.label_names = list(config.get("label_names", []))
        if len(self.label_names) != self.num_labels:
            raise ValueError("OpcodeGraphResidualMIL requires six label names")
        self.sequence_branch = MLM8ViewMultiSlotMIL(config)
        checkpoint_path = config.get("sequence_checkpoint")
        if checkpoint_path:
            self._load_sequence_checkpoint(checkpoint_path)
        for parameter in self.sequence_branch.parameters():
            parameter.requires_grad = False
        self.sequence_branch.eval()

        graph_dim = int(config.get("graph_hidden_dim", 256))
        self.graph_projection = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, graph_dim),
            nn.GELU(),
        )
        self.graph_layers = nn.ModuleList([
            RelationGraphLayer(graph_dim, 6, float(config.get("graph_dropout", 0.1)))
            for _ in range(2)
        ])
        self.graph_queries = nn.Parameter(torch.empty(self.num_labels, int(config.get("graph_query_slots", 3)), graph_dim))
        self.graph_key = nn.Linear(graph_dim, graph_dim, bias=False)
        self.graph_value = nn.Linear(graph_dim, graph_dim, bias=False)
        self.graph_slot_score = nn.Linear(graph_dim, 1)
        self.graph_slot_merge = nn.Linear(graph_dim, 1)
        self.graph_heads = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(graph_dim), nn.Linear(graph_dim, graph_dim // 2), nn.GELU(), nn.Linear(graph_dim // 2, 1))
            for _ in range(self.num_labels)
        ])
        self.empty_graph = nn.Parameter(torch.zeros(graph_dim))
        self.alpha = nn.Parameter(torch.zeros(self.num_labels))
        self.sparse_topk = int(config.get("graph_sparse_topk", 0))
        self.gumbel_temperature = float(config.get("graph_gumbel_temperature", 0.5))
        self.use_graph_edges = bool(config.get("use_graph_edges", True))
        self.graph_edge_type_mask = set(int(value) for value in config.get("graph_edge_type_mask", [0, 1, 2, 3, 4, 5]))
        nn.init.normal_(self.graph_queries, mean=0.0, std=0.02)

    def _load_sequence_checkpoint(self, checkpoint_path):
        payload = torch.load(checkpoint_path, map_location="cpu")
        state = payload.get("model_state_dict", payload)
        state = {
            (key[7:] if key.startswith("module.") else key): value
            for key, value in state.items()
        }
        missing, unexpected = self.sequence_branch.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise ValueError(f"Sequence checkpoint mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")

    def train(self, mode=True):
        super().train(mode)
        self.sequence_branch.eval()
        return self

    def _graph_one(self, node_features, node_mask, edge_index, edge_type, return_attention):
        x = self.graph_projection(node_features)
        if x.shape[0] == 0 or not bool(node_mask.bool().any()):
            label_repr = self.empty_graph.view(1, -1).expand(self.num_labels, -1)
            graph_logits = torch.cat(
                [head(label_repr[label_id]) for label_id, head in enumerate(self.graph_heads)],
                dim=0,
            )
            if return_attention:
                return graph_logits, {
                    "node_attention": x.new_zeros((self.num_labels, self.graph_queries.shape[1], x.shape[0])),
                    "slot_weights": x.new_full((self.num_labels, self.graph_queries.shape[1]), 1.0 / self.graph_queries.shape[1]),
                    "selected_nodes": None,
                    "node_features": x,
                }
            return graph_logits, None
        for layer in self.graph_layers:
            x = layer(x, edge_index, edge_type, node_mask, use_edges=self.use_graph_edges)
        valid = node_mask.bool()
        keys = self.graph_key(x)
        values = self.graph_value(x)
        slots = self.graph_queries + keys.new_zeros((self.num_labels, self.graph_queries.shape[1], keys.shape[-1]))
        slot_logits = torch.einsum("lsd,vd->lsv", slots, keys) / (keys.shape[-1] ** 0.5)
        slot_logits = slot_logits.masked_fill(
            ~valid.view(1, 1, -1), torch.finfo(slot_logits.dtype).min
        )
        selected = None
        if self.sparse_topk > 0 and int(valid.sum()) > self.sparse_topk:
            label_scores = slot_logits.max(dim=1).values
            if self.training:
                label_scores = label_scores + torch.rand_like(label_scores).clamp_min(1e-6).log().neg().log().neg() * self.gumbel_temperature
            top_indices = label_scores.topk(self.sparse_topk, dim=-1).indices
            selected = torch.zeros_like(label_scores, dtype=torch.bool)
            selected.scatter_(1, top_indices, True)
            slot_logits = slot_logits.masked_fill(
                ~selected.unsqueeze(1), torch.finfo(slot_logits.dtype).min
            )
        attention = torch.softmax(slot_logits, dim=-1)
        slot_repr = torch.einsum("lsv,vd->lsd", attention, values)
        merge = torch.softmax(self.graph_slot_merge(slot_repr).squeeze(-1), dim=-1)
        label_repr = torch.einsum("ls,lsd->ld", merge, slot_repr)
        graph_logits = torch.cat([head(label_repr[label_id]) for label_id, head in enumerate(self.graph_heads)], dim=0)
        if return_attention:
            return graph_logits, {"node_attention": attention, "slot_weights": merge, "selected_nodes": selected, "node_features": x}
        return graph_logits, None

    def forward(self, sequence_features, sequence_mask, node_features, node_mask, edge_index, edge_type, return_attention=False):
        with torch.no_grad():
            sequence_output = self.sequence_branch(sequence_features, sequence_mask, return_attention=return_attention)
        sequence_logits = sequence_output["recognition_logits"]
        graph_logits = []
        graph_attention = []
        for index in range(len(node_features)):
            current_edge_index = edge_index[index]
            current_edge_type = edge_type[index]
            if current_edge_type.numel() and self.graph_edge_type_mask != set(range(6)):
                keep = torch.zeros_like(current_edge_type, dtype=torch.bool)
                for edge_type_id in self.graph_edge_type_mask:
                    keep |= current_edge_type == edge_type_id
                current_edge_index = current_edge_index[:, keep]
                current_edge_type = current_edge_type[keep]
            current_logits, current_attention = self._graph_one(
                node_features[index], node_mask[index], current_edge_index, current_edge_type, return_attention
            )
            graph_logits.append(current_logits)
            if return_attention:
                graph_attention.append(current_attention)
        graph_logits = torch.stack(graph_logits, dim=0)
        final_logits = sequence_logits + torch.tanh(self.alpha).view(1, -1) * graph_logits
        result = {
            "recognition_logits": final_logits,
            "sequence_logits": sequence_logits,
            "graph_logits": graph_logits,
            "residual_gate": torch.tanh(self.alpha),
        }
        if return_attention:
            result["sequence_attention"] = sequence_output
            result["graph_attention"] = graph_attention
        return result
