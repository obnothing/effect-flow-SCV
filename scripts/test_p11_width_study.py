"""Shape and isolation checks for the four confirmed width variants."""

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))

import run_polarity_queries as base
from polarity_query_model import build_model


EXPECTED = {
    "e1.yaml": (256, 256, 512, 4, 64, 256),
    "e2.yaml": (384, 384, 768, 6, 64, 384),
    "e3.yaml": (512, 512, 1024, 8, 64, 512),
    "e4.yaml": (128, 128, 256, 2, 64, 128),
}


class WidthStudyTest(unittest.TestCase):
    def test_requested_dimensions_reach_model(self):
        for filename, expected in EXPECTED.items():
            with self.subTest(filename=filename):
                config = base.load_config(ROOT / "configs/p11_width_study" / filename)
                embedding, hidden, output, heads, head_dim, query = expected
                model = build_model("P11", config, 64, 0)
                self.assertEqual(model.embedding.embedding_dim, embedding)
                self.assertEqual(model.encoder.hidden_size, hidden)
                self.assertTrue(model.encoder.bidirectional)
                self.assertEqual(model.output_dim, output)
                self.assertEqual(model.cross_attention.num_heads, heads)
                self.assertEqual(model.cross_attention.head_dim, head_dim)
                self.assertEqual(model.query_dim, query)
                self.assertEqual(tuple(model.queries.shape), (6, 2, query))
                ids = torch.randint(1, 64, (2, 13)); lengths = torch.tensor([13, 8])
                mask = torch.arange(13).unsqueeze(0) < lengths.unsqueeze(1)
                result = model(ids, lengths, mask)
                self.assertEqual(tuple(result["logits"].shape), (2, 6))
                self.assertEqual(tuple(result["representations"].shape), (2, 6, 2, query))


if __name__ == "__main__":
    unittest.main()
