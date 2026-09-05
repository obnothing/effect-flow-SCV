"""CPU-only gradient-path smoke test for standalone CRER."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crer_classifier import CRERClassifier, parameter_group_diagnostics  # noqa: E402


def main():
    torch.manual_seed(42)
    model = CRERClassifier(feature_dim=8, num_labels=3, max_chunks=6, hidden_dim=16, num_heads=4, shared_encoder_layers=1, evidence_blocks=2, dropout=0.0)
    features = torch.randn(2, 6, 8, 8)
    mask = torch.tensor([[True, True, True, False, False, False], [True, True, True, True, False, False]])
    labels = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    output = model(features, mask, return_diagnostics=True)
    loss = model.compute_loss(output, labels, torch.ones(3))["loss"]
    loss.backward()
    groups = {row["group"]: row["grad_norm"] for row in parameter_group_diagnostics(model)}
    assert torch.isfinite(loss)
    required = ("label_embedding", "query_projection", "key_projection", "value_projection", "evidence_block_1", "evidence_block_2", "feature_encoder", "prediction_head")
    assert all(groups[name] > 0.0 for name in required), groups
    assert output["recognition_logits"].shape == (2, 3)
    assert torch.isfinite(output["counterfactual_delta"]).all()
    assert torch.all(output["routing_weights"][0, 3:] == 0)
    print({"loss": float(loss.detach()), "gradient_norms": groups, "status": "ok"})


if __name__ == "__main__":
    main()
