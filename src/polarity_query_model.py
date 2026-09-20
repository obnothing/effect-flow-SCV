"""Contract-label supervised polarity queries over R0 token representations."""

import hashlib

import torch
from torch import nn
from torch.nn import functional as F

from light_label_model import LabelGuidedOpcodeNet, validate_model_config

VARIANTS = ("P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10", "P11", "P12", "P13")


class SharedMultiHeadCrossAttention(nn.Module):
    """Memory-bounded multi-head query-to-token cross-attention.

    The explicit score tensor is [B, heads, query_tokens, key_tokens]; it
    never constructs a key-token by key-token attention matrix.
    """

    def __init__(self, embed_dim, num_heads, key_dim=None):
        super().__init__()
        if int(embed_dim) % int(num_heads):
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = int(embed_dim)
        self.key_dim = int(key_dim if key_dim is not None else embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.k_proj = nn.Linear(self.key_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.key_dim, self.embed_dim, bias=False)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=False,
                average_attn_weights=False):
        batch, query_tokens, _ = query.shape
        key_tokens = key.shape[1]
        q = self.q_proj(query).view(batch, query_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch, key_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch, key_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        attended = torch.matmul(weights, v).transpose(1, 2).contiguous().view(batch, query_tokens, self.embed_dim)
        output = self.out_proj(attended)
        if not need_weights:
            return output, None
        return output, weights if not average_attn_weights else weights.mean(1)


class PolarityQueryNet(LabelGuidedOpcodeNet):
    def __init__(self, mode, vocab_size, pad_id, embedding_dim=128, hidden=384, labels=6, heads=4,
                 bidirectional=True, gru_layers=1, representation_dropout=0.0, query_dim=None):
        if mode not in VARIANTS[1:]:
            raise ValueError(mode)
        # Reuse precisely the original opcode embedding and packed BiGRU.
        super().__init__("b0_mean", vocab_size, pad_id, embedding_dim, hidden, labels,
                         bidirectional, gru_layers=gru_layers,
                         representation_dropout=representation_dropout)
        del self.classifier
        self.mode = mode
        self.polarities = 1 if mode == "P1" else 2
        self.query_dim = int(query_dim if query_dim is not None else self.output_dim)
        self.queries = nn.Parameter(torch.empty(labels, self.polarities, self.query_dim))
        nn.init.normal_(self.queries, std=0.02)
        self.cross_attention = SharedMultiHeadCrossAttention(self.query_dim, heads, key_dim=self.output_dim)
        self.label_scorer = nn.Parameter(torch.empty(labels, self.query_dim))
        nn.init.xavier_uniform_(self.label_scorer)
        self.branch_bias = nn.Parameter(torch.zeros(labels, self.polarities))

    def score(self, representations):
        scored = self.representation_dropout(representations)
        energies = torch.einsum("blpd,ld->blp", scored, self.label_scorer) + self.branch_bias
        if self.mode == "P1":
            logits = energies[..., 0]
        elif self.mode == "P2":
            # Strict capacity control: average the two branch energies.
            logits = energies.mean(2)
        else:
            logits = energies[..., 0] - energies[..., 1]
        return logits, energies

    def encode_tokens(self, input_ids, lengths, mask):
        """Encoder hook used by architecture studies; P11 remains unchanged."""
        return self.encode(input_ids, lengths)

    def forward(self, input_ids, lengths, mask, diagnostics=False):
        if mask.dtype != torch.bool or not bool(mask.any(1).all()):
            raise ValueError("Each sample needs a nonempty boolean token mask")
        hidden = self.encode_tokens(input_ids, lengths, mask)
        queries = self.queries.flatten(0, 1).unsqueeze(0).expand(input_ids.shape[0], -1, -1)
        evidence, attention = self.cross_attention(queries, hidden, hidden,
            key_padding_mask=~mask, need_weights=diagnostics, average_attn_weights=False)
        representations = evidence.reshape(input_ids.shape[0], self.num_labels, self.polarities, self.query_dim)
        logits, energies = self.score(representations)
        result = {"logits": logits, "energies": energies, "representations": representations}
        if diagnostics:
            result["attention"] = attention.mean(1).reshape(input_ids.shape[0], self.num_labels, self.polarities, -1)
        return result


def loss_terms(output, targets, pos_weight, auxiliary_weight,
               positive_multiplier=1.0, negative_multiplier=1.0,
               positive_label_multiplier=1.0, negative_label_multiplier=1.0,
               dos_soft_targets=None):
    classification = F.binary_cross_entropy_with_logits(output["logits"], targets, pos_weight=pos_weight)
    polarity = classification.new_zeros(())
    if auxiliary_weight:
        energies = output["energies"]
        positive_targets = targets
        negative_targets = 1 - targets
        if dos_soft_targets is not None:
            positive_targets = targets.clone()
            negative_targets = (1 - targets).clone()
            dos = targets[..., 4] > 0.5
            positive_high = torch.as_tensor(dos_soft_targets["positive_high"], device=targets.device, dtype=targets.dtype)
            positive_low = torch.as_tensor(dos_soft_targets["positive_low"], device=targets.device, dtype=targets.dtype)
            negative_low = torch.as_tensor(dos_soft_targets["negative_low"], device=targets.device, dtype=targets.dtype)
            negative_high = torch.as_tensor(dos_soft_targets["negative_high"], device=targets.device, dtype=targets.dtype)
            positive_targets[..., 4] = torch.where(dos, positive_high, positive_low)
            negative_targets[..., 4] = torch.where(dos, negative_low, negative_high)
        positive = F.binary_cross_entropy_with_logits(energies[..., 0], positive_targets, reduction="none")
        negative = F.binary_cross_entropy_with_logits(energies[..., 1], negative_targets, reduction="none")
        positive = positive * torch.as_tensor(positive_label_multiplier, device=positive.device,
                                               dtype=positive.dtype).view(1, -1)
        negative = negative * torch.as_tensor(negative_label_multiplier, device=negative.device,
                                               dtype=negative.dtype).view(1, -1)
        positive = positive.mean()
        negative = negative.mean()
        polarity = 0.5 * (positive_multiplier * positive + negative_multiplier * negative)
    return classification + auxiliary_weight * polarity, classification, polarity


def tensor_hash(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        value = tensor.detach().cpu().contiguous()
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def encoder_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            if k.startswith(("embedding.", "encoder.", "sequence_encoder."))}


def build_model(mode, config, vocab_size, pad_id):
    encoder_type = config.get("encoder_type", "bigru")
    if encoder_type != "bigru":
        from encoders.pdvq_encoders import EncoderStudyPDVQNet
        # initialize() builds a P0 template only to obtain an identical encoder
        # state. The template therefore uses the same study encoder and its
        # head is discarded by encoder_state().
        study_mode = "P11" if mode == "P0" else mode
        return EncoderStudyPDVQNet(study_mode, config, vocab_size, pad_id)
    if mode == "P0":
        model = LabelGuidedOpcodeNet("b2_label_attention", vocab_size, pad_id,
            config["embedding_dim"], config["gru_hidden_size"], config["num_labels"],
            config["bidirectional"], gru_layers=config["gru_layers"])
    else:
        model = PolarityQueryNet(mode, vocab_size, pad_id, config["embedding_dim"],
            config["gru_hidden_size"], config["num_labels"], config["attention_heads"],
            config["bidirectional"], config["gru_layers"], config.get("representation_dropout", 0.0),
            config.get("query_dim", 2 * config["gru_hidden_size"]))
    validate_model_config(model, config)
    if mode != "P0":
        if model.cross_attention.num_heads != config["attention_heads"]:
            raise ValueError("Attention heads mismatch")
        if model.queries.shape != (config["num_labels"], 1 if mode == "P1" else 2,
                                   config.get("query_dim", 2 * config["gru_hidden_size"])):
            raise ValueError("Query shape mismatch")
    return model
