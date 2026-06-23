import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fuse_predictions_validation_search import (  # noqa: E402
    align_model_predictions,
    fuse_arrays,
    paired_bootstrap,
    search_fusion,
    weight_grid,
)


class FusionValidationSearchTest(unittest.TestCase):
    def test_weight_grids_are_simplex(self):
        self.assertEqual(len(weight_grid(2)), 21)
        self.assertEqual(len(weight_grid(3)), 66)
        for size in (2, 3):
            for weights in weight_grid(size):
                self.assertAlmostEqual(sum(weights), 1.0)

    def test_probability_and_logit_fusion(self):
        first = np.asarray([[0.2, 0.8]])
        second = np.asarray([[0.6, 0.4]])
        probability = fuse_arrays([first, second], [0.25, 0.75], "probability")
        self.assertTrue(np.allclose(probability, [[0.5, 0.5]]))
        logits = fuse_arrays([first, second], [0.0, 1.0], "logit")
        self.assertTrue(np.allclose(logits, second))

    def test_alignment_search_and_bootstrap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            truths = {
                "a": (0, [0, 0]),
                "b": (1, [1, 0]),
                "c": (1, [0, 1]),
                "d": (1, [1, 1]),
                "e": (0, [0, 0]),
                "f": (1, [1, 0]),
            }
            model_probs = {
                "m1": {
                    "a": (0.1, [0.1, 0.1]),
                    "b": (0.8, [0.8, 0.2]),
                    "c": (0.7, [0.3, 0.7]),
                    "d": (0.9, [0.8, 0.6]),
                    "e": (0.2, [0.2, 0.2]),
                    "f": (0.7, [0.7, 0.3]),
                },
                "m2": {
                    "a": (0.2, [0.2, 0.1]),
                    "b": (0.7, [0.6, 0.1]),
                    "c": (0.8, [0.1, 0.9]),
                    "d": (0.8, [0.6, 0.9]),
                    "e": (0.1, [0.1, 0.2]),
                    "f": (0.8, [0.9, 0.2]),
                },
            }
            specs = []
            for model_name, probabilities in model_probs.items():
                path = root / f"{model_name}.jsonl"
                order = list(reversed(truths)) if model_name == "m2" else list(truths)
                with path.open("w", encoding="utf-8") as f:
                    for sample_id in order:
                        binary_true, multi_true = truths[sample_id]
                        binary_prob, multi_prob = probabilities[sample_id]
                        f.write(
                            json.dumps(
                                {
                                    "id": sample_id,
                                    "binary_true": binary_true,
                                    "binary_prob": binary_prob,
                                    "multi_true": multi_true,
                                    "multi_prob": multi_prob,
                                }
                            )
                            + "\n"
                        )
                specs.append(
                    {
                        "name": model_name,
                        "valid_path": str(path),
                        "test_path": str(path),
                    }
                )

            aligned = align_model_predictions(specs, "valid")
            self.assertEqual(aligned["ids"], list(truths))
            rows, recognition, detection = search_fusion(
                aligned,
                {
                    "models": specs,
                    "max_models_per_fusion": 2,
                    "fusion_types": ["logit", "probability"],
                },
            )
            self.assertEqual(len(rows), 21 * 2 * 3)
            self.assertEqual(recognition["model_names"], ["m1", "m2"])
            self.assertEqual(detection["model_names"], ["m1", "m2"])

            y_multi = aligned["multi_true"]
            perfect = y_multi.copy()
            y_binary = aligned["binary_true"]
            bootstrap = paired_bootstrap(
                y_multi,
                perfect,
                perfect,
                y_binary,
                y_binary,
                y_binary,
                samples=20,
                seed=42,
            )
            for metric in bootstrap["metrics"].values():
                self.assertAlmostEqual(metric["observed_delta"], 0.0)
                self.assertTrue(metric["ci_crosses_zero"])


if __name__ == "__main__":
    unittest.main()
