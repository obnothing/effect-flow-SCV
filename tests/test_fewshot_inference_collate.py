"""Regression coverage for label-free validation inference batches."""

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import light_label_data
from fewshot.run_pilot import m5_parameter_counts, retention_metrics, base_dev_macro_f1_from_audit


class InferenceCollateTest(unittest.TestCase):
    def test_label_free_items_collate_without_synthesizing_labels(self):
        items = [
            {"id": "c1", "input_ids": torch.tensor([3, 4]), "length": 2, "original_length": 2},
            {"id": "c2", "input_ids": torch.tensor([8]), "length": 1, "original_length": 5},
        ]
        collate = getattr(
            light_label_data,
            "collate_light_label_inference",
            light_label_data.collate_light_label,
        )

        batch = collate(items, pad_id=0)

        self.assertEqual(batch["input_ids"].tolist(), [[3, 4], [8, 0]])
        self.assertEqual(batch["mask"].tolist(), [[True, True], [True, False]])
        self.assertEqual(batch["lengths"].tolist(), [2, 1])
        self.assertEqual(batch["ids"], ["c1", "c2"])
        self.assertNotIn("labels", batch)

    def test_training_collate_still_returns_labels(self):
        item = {
            "id": "train1",
            "input_ids": torch.tensor([3, 4]),
            "length": 2,
            "original_length": 2,
            "labels": torch.tensor([1.0, 0.0]),
        }

        batch = light_label_data.collate_light_label([item], pad_id=0)

        self.assertEqual(batch["labels"].tolist(), [[1.0, 0.0]])

    def test_old_label_retention_metrics_are_available_for_final_eval(self):
        metrics = retention_metrics([[1, 0], [0, 1]], [[2.0, -2.0], [-2.0, 2.0]])
        self.assertAlmostEqual(metrics["macro_f1"], 1.0)
        self.assertAlmostEqual(metrics["micro_f1"], 1.0)
        self.assertEqual(metrics["per_label_f1"], [1.0, 1.0])

    def test_base_audit_report_reads_the_written_macro_f1_key(self):
        audit = {"best_epoch": 5, "base_dev_macro_f1_fixed05": 0.7717}
        self.assertAlmostEqual(base_dev_macro_f1_from_audit(audit), 0.7717)

    def test_m5_parameter_groups_are_read_from_head_metadata(self):
        counts = {"query": 1024, "scorer": 514, "encoder": 3746304, "attention": 1572864}
        self.assertEqual(m5_parameter_counts({"metadata": {"parameter_counts": counts}}), counts)


if __name__ == "__main__":
    unittest.main()
