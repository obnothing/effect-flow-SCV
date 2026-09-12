"""Focused regression tests for the capacity-trial configuration boundary."""
import sys
from pathlib import Path
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_hidden384_trials import configuration, model_for, optimizer_for, weights_for, TRIALS


class Tokenizer:
    pad_token_id = 0
    def __len__(self): return 32


class TrialTests(unittest.TestCase):
    def test_architecture_changes_reach_parameters(self):
        c = configuration()
        for key, value in [("gru_hidden_size", 512), ("gru_layers", 2), ("embedding_dim", 256)]:
            changed = dict(c, **{key:value})
            model = model_for(changed, Tokenizer(), "cpu")
            if key == "gru_layers": self.assertIn("encoder.weight_ih_l1", model.state_dict())
            if key == "gru_hidden_size": self.assertEqual(model.label_queries.shape[1], 1024)
            if key == "embedding_dim": self.assertEqual(model.encoder.weight_ih_l0.shape[1], 256)

    def test_dropout_and_checkpoint(self):
        c = dict(configuration(), gru_hidden_size=8, embedding_dim=8, representation_dropout=0.1)
        m = model_for(c, Tokenizer(), "cpu")
        x = torch.ones(2, 8, dtype=torch.long); lengths = torch.tensor([8, 8]); mask = x.bool()
        m.train(); self.assertFalse(torch.equal(m(x,lengths,mask)["logits"], m(x,lengths,mask)["logits"]))
        m.eval(); expected = m(x,lengths,mask)["logits"]
        restored = model_for(c, Tokenizer(), "cpu"); restored.load_state_dict(m.state_dict()); restored.eval()
        torch.testing.assert_close(restored(x,lengths,mask)["logits"], expected)

    def test_optimizer_scheduler_and_weights(self):
        c = dict(configuration(), scheduler="cosine")
        m = model_for(c, Tokenizer(), "cpu"); optimizer, scheduler = optimizer_for(c,m)
        for _ in range(30): optimizer.step(); scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.0001)
        class Data: pass
        d = Data(); d.labels = torch.tensor([[1.]*6] + [[0.]*6]*8); d.indices=list(range(9))
        base = weights_for(c,d); altered=weights_for(dict(c,dos_weight_multiplier=1.5),d)
        self.assertTrue(torch.equal(base[:4],altered[:4])); self.assertEqual(base[5],altered[5]); self.assertGreater(altered[4],base[4])

    def test_each_trial_changes_exactly_one_field(self):
        c = configuration()
        for name, delta in TRIALS.items():
            self.assertEqual(len(delta), 0 if name.startswith("T0") else 1)
            for key, value in delta.items(): self.assertNotEqual(c[key], value)


if __name__ == "__main__": unittest.main()
