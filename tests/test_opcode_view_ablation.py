import sys
import unittest

import torch

sys.path.insert(0, "src")

from lse_mil_model import LabelSpecificEvidenceRoutingMIL  # noqa: E402


def config(view_indices, context_view_index):
    return {
        "num_labels": 6,
        "num_views": len(view_indices),
        "feature_dim": 768,
        "hidden_dim": 32,
        "evidence_slots": 3,
        "evidence_topk_chunks": 2,
        "context_layers": 1,
        "context_heads": 4,
        "dropout": 0.0,
        "context_view_index": context_view_index,
    }


class OpcodeViewAblationTest(unittest.TestCase):
    def test_selected_view_shapes_and_attention(self):
        for indices, context in [([0], 0), ([1], 0), ([2], 0), ([0, 1], 1), ([1, 2], 0), ([0, 1, 2], 1), (list(range(8)), 1)]:
            model = LabelSpecificEvidenceRoutingMIL(config(indices, context)).eval()
            features = torch.randn(2, 4, len(indices), 768)
            mask = torch.ones(2, 4, dtype=torch.bool)
            output = model(features, mask, return_attention=True)
            self.assertEqual(tuple(output["recognition_logits"].shape), (2, 6))
            self.assertEqual(tuple(output["view_attention"].shape), (2, 4, 6, 3, len(indices)))
            self.assertTrue(torch.allclose(output["view_attention"].sum(dim=-1), torch.ones(2, 4, 6, 3), atol=1e-5))

    def test_slot_count_is_independent_from_view_count(self):
        model = LabelSpecificEvidenceRoutingMIL(config([0, 1], 1)).eval()
        self.assertEqual(model.slots, 3)
        self.assertEqual(model.num_views, 2)

    def test_invalid_context_view_fails(self):
        model = LabelSpecificEvidenceRoutingMIL(config([0, 1], 2)).eval()
        with self.assertRaises(ValueError):
            model(torch.randn(1, 2, 2, 768), torch.ones(1, 2, dtype=torch.bool))


if __name__ == "__main__":
    unittest.main()
