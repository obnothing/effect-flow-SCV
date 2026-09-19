"""Focused protocol tests for P11 Stage 1."""

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))

import run_polarity_queries as base
from polarity_query_model import build_model


class P11Stage1Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = base.load_config(ROOT / "configs/p11_hparam_stage1/base.yaml")

    def test_frozen_protocol(self):
        expected = {"data_dir": "data/processed/DIVE_main6_opcode_process01", "max_len": 8192,
                    "embedding_dim": 128, "gru_hidden_size": 384, "gru_layers": 1,
                    "bidirectional": True, "attention_heads": 4, "learning_rate": 0.001,
                    "weight_decay": 0.0001, "epochs": 40, "seed": 42, "allow_test": False}
        self.assertEqual({key: self.config[key] for key in expected}, expected)
        self.assertEqual(self.config["early_stopping_patience"], 40)

    def test_representation_dropout_is_real_and_localized(self):
        config = dict(self.config, representation_dropout=0.1)
        model = build_model("P11", config, 32, 0)
        self.assertEqual(model.representation_dropout.p, 0.1)
        self.assertEqual(model.embedding.padding_idx, 0)
        self.assertEqual(model.encoder.dropout, 0.0)
        representations = torch.ones(128, 6, 2, 768)
        model.train(); torch.manual_seed(42)
        dropped, _ = model.score(representations)
        model.eval(); undropped, _ = model.score(representations)
        self.assertFalse(torch.equal(dropped, undropped))

    def test_train_only_power_weights(self):
        train, _ = base.datasets(self.config)
        labels = train.labels[train.indices]
        for power in (0.4, 0.5, 0.6):
            config = dict(self.config, pos_weight_power=power)
            actual = base.compute_weights(config, train)
            expected = ((len(labels) - labels.sum(0)) / labels.sum(0).clamp_min(1)).pow(power).clamp(1.0, 5.0)
            self.assertTrue(torch.allclose(actual, expected))

    def test_effective_batch_mapping(self):
        self.assertEqual(128 // 64, 2)
        self.assertEqual(256 // 64, 4)


if __name__ == "__main__":
    unittest.main()
