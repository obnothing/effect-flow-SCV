import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from solidity_graph_dataset import build_graph_attention
from solidity_graphcodebert_model import multilabel_supervised_contrastive, symmetric_kl


class SolidityGraphRouteTest(unittest.TestCase):
    def test_sparse_graph_attention_respects_padding_and_edges(self):
        mask = torch.tensor([[[True, True, True, True, True, False]]])
        code_ends = torch.tensor([[3]], dtype=torch.int16)
        nodes = torch.tensor([[[0, 1, -1]]], dtype=torch.int16)
        edge_src = torch.tensor([[[0, -1, -1]]], dtype=torch.int16)
        edge_dst = torch.tensor([[[1, -1, -1]]], dtype=torch.int16)
        attention = build_graph_attention(mask, code_ends, nodes, edge_src, edge_dst)
        self.assertEqual(tuple(attention.shape), (1, 1, 6, 6))
        self.assertTrue(attention[0, 0, 3, 1] and attention[0, 0, 1, 3])
        self.assertTrue(attention[0, 0, 3, 4] and attention[0, 0, 4, 3])
        self.assertFalse(attention[0, 0, 5].any())

    def test_multilabel_contrastive_is_finite_with_zero_label_contracts(self):
        projection = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
        labels = torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
        loss = multilabel_supervised_contrastive(projection, labels)
        self.assertTrue(torch.isfinite(loss) and loss >= 0)

    def test_symmetric_kl_is_zero_for_identical_logits(self):
        logits = torch.tensor([[0.2, -0.3], [1.0, 0.0]])
        self.assertEqual(symmetric_kl(logits, logits).item(), 0.0)


if __name__ == "__main__":
    unittest.main()
