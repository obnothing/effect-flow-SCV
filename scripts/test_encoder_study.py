"""Focused CPU tests for PDVQ encoder variants."""

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polarity_query_model import build_model


class EncoderStudyTest(unittest.TestCase):
    def config(self, encoder_type):
        return {
            "embedding_dim": 128, "gru_hidden_size": 384, "gru_layers": 1, "bidirectional": True,
            "num_labels": 6, "attention_heads": 4, "encoder_type": encoder_type,
            "transformer_d_model": 24, "transformer_layers": 2 if encoder_type != "bigru_local" else 1,
            "transformer_heads": 6, "transformer_ffn_dim": 48, "window_size": 8,
            "block_size": 4, "gamma_init": 0.0, "max_len": 32,
        }

    def test_shapes_padding_and_gradients(self):
        for encoder_type in ("local_transformer", "bigru_local", "bigru_block_global"):
            with self.subTest(encoder_type=encoder_type):
                torch.manual_seed(42)
                model = build_model("P11", self.config(encoder_type), 40, 0)
                ids = torch.randint(1, 40, (2, 17)); lengths = torch.tensor([17, 11])
                mask = torch.arange(17).unsqueeze(0) < lengths.unsqueeze(1)
                ids[1, 11:] = 0
                hidden = model.encode_tokens(ids, lengths, mask)
                self.assertEqual(tuple(hidden.shape), (2, 17, 768))
                self.assertTrue(torch.isfinite(hidden).all())
                self.assertEqual(float(hidden[1, 11:].detach().abs().max()), 0.0)
                output = model(ids, lengths, mask)
                self.assertEqual(tuple(output["logits"].shape), (2, 6))
                output["logits"].sum().backward()
                self.assertIsNotNone(model.queries.grad)
                self.assertGreater(float(model.queries.grad.abs().sum()), 0.0)
                if hasattr(model.sequence_encoder, "gamma"):
                    self.assertIsNotNone(model.sequence_encoder.gamma.grad)

    def test_padding_values_do_not_change_logits(self):
        for encoder_type in ("local_transformer", "bigru_local", "bigru_block_global"):
            with self.subTest(encoder_type=encoder_type):
                torch.manual_seed(7)
                model = build_model("P11", self.config(encoder_type), 40, 0).eval()
                lengths = torch.tensor([9]); mask = torch.arange(16).unsqueeze(0) < lengths.unsqueeze(1)
                left = torch.randint(1, 40, (1, 16)); right = left.clone()
                left[:, 9:] = 0; right[:, 9:] = torch.randint(1, 40, (1, 7))
                with torch.no_grad():
                    a = model(left, lengths, mask)["logits"]
                    b = model(right, lengths, mask)["logits"]
                self.assertTrue(torch.allclose(a, b, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
