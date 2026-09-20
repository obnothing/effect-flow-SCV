"""Memory-bounded sequence encoders for the PDVQ encoder study."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from polarity_query_model import PolarityQueryNet


class FactorizedWindowPosition(nn.Module):
    """Learnable absolute position as block position plus in-window offset."""

    def __init__(self, max_len, window_size, dim):
        super().__init__()
        self.window_size = int(window_size)
        blocks = math.ceil(int(max_len) / self.window_size)
        self.block = nn.Parameter(torch.empty(blocks, dim))
        self.offset = nn.Parameter(torch.empty(self.window_size, dim))
        nn.init.normal_(self.block, std=0.02)
        nn.init.normal_(self.offset, std=0.02)

    def forward(self, length):
        positions = torch.arange(length, device=self.block.device)
        return self.block.index_select(0, positions // self.window_size) + self.offset.index_select(
            0, positions % self.window_size
        )


class WindowTransformerLayer(nn.Module):
    """Pre-LN non-overlapping window attention with O(TW) score complexity."""

    def __init__(self, dim, heads, ffn_dim, window_size):
        super().__init__()
        if dim % heads:
            raise ValueError("local transformer dim must be divisible by heads")
        self.dim = int(dim)
        self.heads = int(heads)
        self.head_dim = self.dim // self.heads
        self.window_size = int(window_size)
        self.norm1 = nn.LayerNorm(self.dim)
        self.qkv = nn.Linear(self.dim, 3 * self.dim, bias=True)
        self.out = nn.Linear(self.dim, self.dim)
        self.norm2 = nn.LayerNorm(self.dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.dim, int(ffn_dim)),
            nn.GELU(),
            nn.Linear(int(ffn_dim), self.dim),
        )

    def forward(self, values, mask):
        batch, length, dim = values.shape
        window = self.window_size
        padded_length = math.ceil(length / window) * window
        pad = padded_length - length
        if pad:
            values = F.pad(values, (0, 0, 0, pad))
            mask = F.pad(mask, (0, pad), value=False)
        windows = values.reshape(batch * (padded_length // window), window, dim)
        window_mask = mask.reshape(batch * (padded_length // window), window)
        nonempty = window_mask.any(1)
        output = torch.zeros_like(windows)
        if bool(nonempty.any()):
            selected = windows[nonempty]
            selected_mask = window_mask[nonempty]
            normalized = self.norm1(selected)
            qkv = self.qkv(normalized).reshape(selected.shape[0], window, 3, self.heads, self.head_dim)
            q, k, v = qkv.unbind(2)
            q, k, v = (item.transpose(1, 2) for item in (q, k, v))
            attended = F.scaled_dot_product_attention(
                q, k, v, attn_mask=selected_mask[:, None, None, :], dropout_p=0.0
            )
            attended = attended.transpose(1, 2).reshape(selected.shape[0], window, dim)
            selected = selected + self.out(attended)
            selected = selected + self.ffn(self.norm2(selected))
            selected = selected * selected_mask.unsqueeze(-1).to(selected.dtype)
            output[nonempty] = selected
        return output.reshape(batch, padded_length, dim)[:, :length]


class LocalWindowEncoder(nn.Module):
    def __init__(self, max_len, input_dim, model_dim, output_dim, layers, heads, ffn_dim, window_size):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, model_dim)
        self.position = FactorizedWindowPosition(max_len, window_size, model_dim)
        self.layers = nn.ModuleList(
            WindowTransformerLayer(model_dim, heads, ffn_dim, window_size) for _ in range(int(layers))
        )
        self.final_norm = nn.LayerNorm(model_dim)
        self.output_projection = nn.Linear(model_dim, output_dim)

    def forward(self, values, mask):
        hidden = self.input_projection(values)
        hidden = hidden + self.position(hidden.shape[1]).unsqueeze(0).to(hidden.dtype)
        hidden = hidden * mask.unsqueeze(-1).to(hidden.dtype)
        for layer in self.layers:
            hidden = layer(hidden, mask)
        hidden = self.output_projection(self.final_norm(hidden))
        return hidden * mask.unsqueeze(-1).to(hidden.dtype)


class ResidualLocalRefiner(nn.Module):
    def __init__(self, max_len, input_dim, model_dim, heads, ffn_dim, window_size, gamma_init):
        super().__init__()
        self.pre_norm = nn.LayerNorm(input_dim)
        self.local = LocalWindowEncoder(max_len, input_dim, model_dim, input_dim, 1, heads, ffn_dim, window_size)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, hidden, mask):
        residual = self.local(self.pre_norm(hidden), mask)
        return (hidden + self.gamma * residual) * mask.unsqueeze(-1).to(hidden.dtype)


class BlockGlobalRefiner(nn.Module):
    def __init__(self, max_len, input_dim, model_dim, layers, heads, ffn_dim, block_size, gamma_init):
        super().__init__()
        self.block_size = int(block_size)
        max_blocks = math.ceil(int(max_len) / self.block_size)
        self.input_projection = nn.Linear(input_dim, model_dim)
        self.position = nn.Parameter(torch.empty(max_blocks, model_dim))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=ffn_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=int(layers), norm=nn.LayerNorm(model_dim))
        self.output_projection = nn.Linear(model_dim, input_dim)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, hidden, mask):
        batch, length, dim = hidden.shape
        block = self.block_size
        padded_length = math.ceil(length / block) * block
        pad = padded_length - length
        padded_hidden = F.pad(hidden, (0, 0, 0, pad)) if pad else hidden
        padded_mask = F.pad(mask, (0, pad), value=False) if pad else mask
        block_mask = padded_mask.reshape(batch, -1, block).any(2)
        weights = padded_mask.reshape(batch, -1, block, 1).to(hidden.dtype)
        pooled = (padded_hidden.reshape(batch, -1, block, dim) * weights).sum(2) / weights.sum(2).clamp_min(1)
        blocks = self.input_projection(pooled)
        blocks = blocks + self.position[: blocks.shape[1]].unsqueeze(0).to(blocks.dtype)
        blocks = self.transformer(blocks, src_key_padding_mask=~block_mask)
        blocks = blocks * block_mask.unsqueeze(-1).to(blocks.dtype)
        global_tokens = self.output_projection(blocks).repeat_interleave(block, dim=1)[:, :length]
        return (hidden + self.gamma * global_tokens) * mask.unsqueeze(-1).to(hidden.dtype)


class EncoderStudyPDVQNet(PolarityQueryNet):
    """P11 head with a selectable encoder and unchanged polarity mechanism."""

    def __init__(self, mode, config, vocab_size, pad_id):
        super().__init__(
            mode,
            vocab_size,
            pad_id,
            config["embedding_dim"],
            config["gru_hidden_size"],
            config["num_labels"],
            config["attention_heads"],
            config["bidirectional"],
            config["gru_layers"],
            config.get("representation_dropout", 0.0),
            config.get("query_dim", 2 * config["gru_hidden_size"]),
        )
        self.encoder_type = config["encoder_type"]
        self.encoder_config = {
            key: config[key]
            for key in (
                "transformer_d_model",
                "transformer_layers",
                "transformer_heads",
                "transformer_ffn_dim",
                "window_size",
                "block_size",
                "gamma_init",
            )
        }
        if self.output_dim != 768:
            raise ValueError("Encoder study requires a 768-dimensional PDVQ input")
        if self.encoder_type == "local_transformer":
            del self.encoder
            self.sequence_encoder = LocalWindowEncoder(
                config["max_len"],
                config["embedding_dim"],
                config["transformer_d_model"],
                self.output_dim,
                config["transformer_layers"],
                config["transformer_heads"],
                config["transformer_ffn_dim"],
                config["window_size"],
            )
        elif self.encoder_type == "bigru_local":
            self.sequence_encoder = ResidualLocalRefiner(
                config["max_len"],
                self.output_dim,
                config["transformer_d_model"],
                config["transformer_heads"],
                config["transformer_ffn_dim"],
                config["window_size"],
                config["gamma_init"],
            )
        elif self.encoder_type == "bigru_block_global":
            self.sequence_encoder = BlockGlobalRefiner(
                config["max_len"],
                self.output_dim,
                config["transformer_d_model"],
                config["transformer_layers"],
                config["transformer_heads"],
                config["transformer_ffn_dim"],
                config["block_size"],
                config["gamma_init"],
            )
        else:
            raise ValueError(f"Unknown encoder_type: {self.encoder_type}")

    def encode_tokens(self, input_ids, lengths, mask):
        if self.encoder_type == "local_transformer":
            return self.sequence_encoder(self.embedding(input_ids), mask)
        hidden = super().encode_tokens(input_ids, lengths, mask)
        return self.sequence_encoder(hidden, mask)

    def encoder_audit(self):
        return {
            "encoder_type": self.encoder_type,
            "encoder_config": dict(self.encoder_config),
            "output_dim": self.output_dim,
            "local_attention_complexity": "O(TW)" if "local" in self.encoder_type else None,
            "constructs_full_token_attention": False,
        }
