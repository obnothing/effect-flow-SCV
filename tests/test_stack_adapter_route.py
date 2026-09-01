import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stack_adapter_bert import LoRALinear, StackStructuralEncoder  # noqa: E402


def test_lora_starts_as_identity():
    base = torch.nn.Linear(4, 4)
    layer = LoRALinear(base, rank=2)
    x = torch.randn(3, 4)
    assert torch.allclose(layer(x), base(x))


def test_structural_encoder_handles_empty_edges_and_padding():
    encoder = StackStructuralEncoder(output_dim=16, hidden_dim=16, layers=1, heads=4)
    state = torch.zeros((2, 8, 5), dtype=torch.long)
    mask = torch.tensor([[True] * 8, [True, True, False, False, False, False, False, False]])
    output = encoder(
        state, mask, torch.tensor([0, 0, 0]),
        torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long),
        torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long),
        torch.empty(0, dtype=torch.long), torch.empty(0),
    )
    assert output.shape == (2, 8, 16)
    assert torch.isfinite(output).all()
