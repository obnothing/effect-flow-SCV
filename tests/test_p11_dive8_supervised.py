"""Eight-label capacity/configuration and reporting contract checks."""
import io
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train_p11_dive8 import build_model, load_config, loss, metrics


class SupervisedP11Test(unittest.TestCase):
    def test_both_architectures_apply_configuration_and_receive_gradients(self):
        torch.set_num_threads(2)
        for variant, embedding, hidden, heads, query in (
            ("original", 128, 384, 4, 768), ("e3", 512, 512, 8, 512)
        ):
            with self.subTest(variant=variant):
                c = load_config(variant)
                model = build_model("P11", c, 100, 0)
                self.assertEqual(model.embedding.embedding_dim, embedding)
                self.assertEqual(model.encoder.hidden_size, hidden)
                self.assertEqual(tuple(model.queries.shape), (8, 2, query))
                self.assertEqual(model.cross_attention.num_heads, heads)
                self.assertEqual(model.cross_attention.k_proj.in_features, 2 * hidden)
                x = torch.tensor([[2, 3, 4, 0], [5, 6, 7, 8]])
                mask = x != 0; lengths = mask.sum(1)
                target = torch.tensor([[1., 0., 1., 0., 1., 0., 1., 0.], [0., 1., 0., 1., 0., 1., 0., 1.]])
                output = model(x, lengths, mask)
                self.assertEqual(tuple(output["logits"].shape), (2, 8))
                value, _, _ = loss(output, target, torch.ones(8), c)
                value.backward()
                self.assertTrue(torch.isfinite(model.queries.grad).all())
                self.assertTrue((model.queries.grad.abs().sum(-1) > 0).all())
                model.eval()
                expected = model(x, lengths, mask)["logits"].detach()
                saved = io.BytesIO(); torch.save(model.state_dict(), saved); saved.seek(0)
                restored = build_model("P11", c, 100, 0).eval()
                restored.load_state_dict(torch.load(saved, weights_only=False))
                torch.testing.assert_close(expected, restored(x, lengths, mask)["logits"])

    def test_eight_label_metrics_and_threshold_output(self):
        c = load_config("original")
        y = torch.tensor([[1., 0.] * 4, [0., 1.] * 4])
        result = metrics(y, (y * 2 - 1) * 4, c)
        self.assertEqual(len(result["thresholds"]), 8)
        self.assertEqual(result["fixed"]["macro_f1"], 1.0)
        self.assertEqual(result["tuned"]["micro_f1"], 1.0)
        self.assertEqual(result["per_label_ap"], [1.0] * 8)
        self.assertEqual(result["fixed"]["tp"], [1] * 8)
        self.assertEqual(result["fixed"]["fp"], [0] * 8)
        self.assertFalse(c["allow_test"])


if __name__ == "__main__":
    unittest.main()
