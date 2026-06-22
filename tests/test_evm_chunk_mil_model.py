import sys
import unittest
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evm_chunk_mil_model import EVMChunkMILClassifier  # noqa: E402


class ChunkContextMILTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.config = {
            "num_labels": 3,
            "feature_dim": 12,
            "hidden_dim": 16,
            "attn_dim": 8,
            "max_chunks": 5,
            "dropout": 0.0,
            "recognition_aggregation": "label_gated_attention",
            "use_chunk_context": True,
            "chunk_context_num_layers": 2,
            "chunk_context_num_heads": 4,
            "chunk_context_dropout": 0.0,
        }

    def test_context_shapes_and_padding_mask(self):
        model = EVMChunkMILClassifier(self.config).eval()
        features = torch.randn(2, 5, 12)
        mask = torch.tensor(
            [[True, True, True, False, False], [True, True, False, False, False]]
        )
        outputs = model(features, mask)
        self.assertEqual(tuple(outputs["detection_logits"].shape), (2,))
        self.assertEqual(tuple(outputs["recognition_logits"].shape), (2, 3))
        self.assertEqual(tuple(outputs["chunk_logits"].shape), (2, 5, 3))
        self.assertEqual(tuple(outputs["chunk_scores"].shape), (2, 5, 3))
        self.assertEqual(
            torch.count_nonzero(
                outputs["chunk_scores"].masked_select(~mask.unsqueeze(-1))
            ).item(),
            0,
        )

        changed_padding = features.clone()
        changed_padding[~mask] = 10000.0
        changed_outputs = model(changed_padding, mask)
        self.assertTrue(
            torch.allclose(
                outputs["recognition_logits"],
                changed_outputs["recognition_logits"],
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                outputs["detection_logits"],
                changed_outputs["detection_logits"],
                atol=1e-6,
            )
        )

    def test_rejects_all_padding_sample(self):
        model = EVMChunkMILClassifier(self.config)
        with self.assertRaisesRegex(ValueError, "at least one valid chunk"):
            model(torch.randn(1, 5, 12), torch.zeros(1, 5, dtype=torch.bool))

    def test_weighted_loss_backpropagates_through_context_encoder(self):
        model = EVMChunkMILClassifier(self.config)
        model.set_recognition_pos_weight(torch.tensor([1.0, 2.0, 3.0]))
        outputs = model(
            torch.randn(2, 5, 12),
            torch.tensor(
                [[True, True, True, False, False], [True, True, False, False, False]]
            ),
            binary_label=torch.tensor([1.0, 0.0]),
            multi_labels=torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
        )
        outputs["loss"].backward()
        self.assertIsNotNone(
            model.chunk_context_encoder.input_projection.weight.grad
        )


if __name__ == "__main__":
    unittest.main()
