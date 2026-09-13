"""Focused tests for the LR trial configuration boundary and schedules."""

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from light_label_model import LabelGuidedOpcodeNet, validate_model_config  # noqa: E402
from run_lr_schedule_trials import TRIALS, base_config, model_for, scheduled_lr  # noqa: E402


class TinyTokenizer:
    pad_token_id = 0

    def __len__(self):
        return 32


class TrialTests(unittest.TestCase):
    def test_trial_changes_are_one_factor(self):
        config = base_config()
        for name, delta in TRIALS.items():
            if name.startswith("R0"):
                self.assertEqual(delta["learning_rate"], config["learning_rate"])
            else:
                self.assertEqual(len(delta), 2 if "trial_group" in delta else 1)
            for key, value in delta.items():
                if key != "trial_group":
                    if name not in {"R0_fixed_lr"}:
                        self.assertNotEqual(config[key], value)

    def test_architecture_is_constructed_from_config(self):
        config = base_config()
        config["representation_dropout"] = 0.1
        model = model_for(config, TinyTokenizer(), "cpu")
        validate_model_config(model, config)
        self.assertEqual(model.encoder.hidden_size, 384)
        self.assertEqual(model.encoder.num_layers, 1)
        self.assertEqual(model.representation_dropout.p, 0.1)
        self.assertEqual(model.label_queries.shape, (6, 768))

    def test_schedules_have_expected_boundaries(self):
        config = base_config()
        self.assertAlmostEqual(scheduled_lr(config, 1), 0.001, places=10)
        self.assertAlmostEqual(scheduled_lr(config, 45), 0.001, places=10)
        config["scheduler"] = "cosine_immediate"
        self.assertAlmostEqual(scheduled_lr(config, 1), 0.001, places=10)
        self.assertAlmostEqual(scheduled_lr(config, 45), 0.0001, places=10)
        config["scheduler"] = "cosine_delayed20"
        self.assertAlmostEqual(scheduled_lr(config, 20), 0.001, places=10)
        self.assertAlmostEqual(scheduled_lr(config, 21), 0.001, places=10)
        self.assertAlmostEqual(scheduled_lr(config, 45), 0.0001, places=10)
        config["scheduler"] = "multistep_26_36"
        self.assertAlmostEqual(scheduled_lr(config, 25), 0.001, places=10)
        self.assertAlmostEqual(scheduled_lr(config, 26), 0.0005, places=10)
        self.assertAlmostEqual(scheduled_lr(config, 36), 0.00025, places=10)

    def test_two_layer_model_changes_parameter_shapes(self):
        config = base_config()
        config["gru_layers"] = 2
        model = model_for(config, TinyTokenizer(), "cpu")
        self.assertIn("encoder.weight_ih_l1", model.state_dict())
        self.assertEqual(model.encoder.num_layers, 2)


if __name__ == "__main__":
    unittest.main()
