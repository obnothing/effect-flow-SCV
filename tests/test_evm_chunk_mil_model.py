import sys
import unittest
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evm_chunk_mil_model import (  # noqa: E402
    EVEFMVDV2MultiScaleMIL,
    EVMChunkMILClassifier,
)


class ChunkContextMILTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.config = {
            "num_labels": 3,
            "feature_dim": 12,
            "hidden_dim": 16,
            "attn_dim": 8,
            "max_chunks": 5,
            "dropout": 0.0,
            "recognition_aggregation": "label_gated_attention",
            "use_chunk_context": True,
            "chunk_context_num_layers": 2,
            "chunk_context_num_heads": 4,
            "chunk_context_dropout": 0.0,
        }

    def test_context_shapes_and_padding_mask(self):
        model = EVMChunkMILClassifier(self.config).eval()
        if hasattr(torch.backends, "mha"):
            self.assertFalse(torch.backends.mha.get_fastpath_enabled())
        features = torch.randn(2, 5, 12)
        mask = torch.tensor(
            [[True, True, True, False, False], [True, True, False, False, False]]
        )
        outputs = model(features, mask)
        self.assertEqual(tuple(outputs["detection_logits"].shape), (2,))
        self.assertEqual(tuple(outputs["recognition_logits"].shape), (2, 3))
        self.assertEqual(tuple(outputs["chunk_logits"].shape), (2, 5, 3))
        self.assertEqual(tuple(outputs["chunk_scores"].shape), (2, 5, 3))
        self.assertEqual(
            torch.count_nonzero(
                outputs["chunk_scores"].masked_select(~mask.unsqueeze(-1))
            ).item(),
            0,
        )

        changed_padding = features.clone()
        changed_padding[~mask] = 10000.0
        changed_outputs = model(changed_padding, mask)
        self.assertTrue(
            torch.allclose(
                outputs["recognition_logits"],
                changed_outputs["recognition_logits"],
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                outputs["detection_logits"],
                changed_outputs["detection_logits"],
                atol=1e-6,
            )
        )

    def test_rejects_all_padding_sample(self):
        model = EVMChunkMILClassifier(self.config)
        with self.assertRaisesRegex(ValueError, "at least one valid chunk"):
            model(torch.randn(1, 5, 12), torch.zeros(1, 5, dtype=torch.bool))

    def test_weighted_loss_backpropagates_through_context_encoder(self):
        model = EVMChunkMILClassifier(self.config)
        model.set_recognition_pos_weight(torch.tensor([1.0, 2.0, 3.0]))
        outputs = model(
            torch.randn(2, 5, 12),
            torch.tensor(
                [[True, True, True, False, False], [True, True, False, False, False]]
            ),
            binary_label=torch.tensor([1.0, 0.0]),
            multi_labels=torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
        )
        outputs["loss"].backward()
        self.assertIsNotNone(
            model.chunk_context_encoder.input_projection.weight.grad
        )

    def test_multiscale_forward_backward_and_padding(self):
        labels = [
            "Reentrancy",
            "Access Control",
            "Arithmetic",
            "Unchecked Return Values",
            "DoS",
            "Time manipulation",
        ]
        config = {
            "num_labels": 6,
            "label_names": labels,
            "feature_dim": 12,
            "hidden_dim": 16,
            "attn_dim": 8,
            "max_chunks": 5,
            "dropout": 0.0,
            "recognition_head_type": "label_branch_mlp",
            "label_branch_hidden_dim": 8,
            "label_branch_dropout": 0.0,
            "use_chunk_context": True,
            "chunk_context_num_layers": 2,
            "chunk_context_num_heads": 4,
            "chunk_context_dropout": 0.0,
            "top_k": 2,
            "multiscale_top_k": 2,
            "multiscale_cross_attention_heads": 4,
            "semantic_dropout": 0.0,
            "warmup_epochs_neural_only": 0,
            "enable_reliable_semantic_epoch": 1,
            "beta_reliable_init": 0.1,
            "gamma_reliable_init": 0.1,
            "efpp_dim": 22,
            "etp_dim": 16,
            "relation_dim": 6,
            "ontology_path": "configs/effect_flow_ontology.yaml",
            "template_path": "configs/vulnerability_templates_dive_main6_new.yaml",
            "efpp_pattern_config": "configs/effect_flow_efpp_conservative_22.json",
            "risk_evidence_weight": 1.0,
            "missing_check_evidence_weight": 1.0,
            "protective_evidence_weight": 1.0,
            "effect_type_evidence_weight": 0.5,
            "relation_evidence_weight": 0.5,
            "detection_head_enabled": False,
            "detection_loss_weight": 0.0,
            "recognition_loss_weight": 1.0,
            "use_weak_semantic_features": False,
        }
        model = EVEFMVDV2MultiScaleMIL(config)
        features = torch.randn(2, 5, 12)
        mask = torch.tensor([[True, True, True, False, False], [True, False, False, False, False]])
        outputs = model(
            features,
            mask,
            efpp_probs=torch.rand(2, 5, 22),
            etp_distribution=torch.softmax(torch.randn(2, 5, 16), dim=-1),
            relation_distribution=torch.softmax(torch.randn(2, 5, 6), dim=-1),
            binary_label=torch.tensor([1.0, 0.0]),
            multi_labels=torch.randint(0, 2, (2, 6)).float(),
        )
        self.assertEqual(tuple(outputs["recognition_logits"].shape), (2, 6))
        self.assertEqual(tuple(outputs["attn_weights"].shape), (2, 5, 6))
        self.assertTrue(torch.isfinite(outputs["loss"]))
        outputs["loss"].backward()
        self.assertIsNotNone(model.multiscale_label_queries.grad)

        checkpoint = model.state_dict()
        restored = EVEFMVDV2MultiScaleMIL(config).eval()
        restored.load_state_dict(checkpoint)
        model.eval()
        semantic_efpp = torch.rand(2, 5, 22)
        semantic_etp = torch.softmax(torch.randn(2, 5, 16), dim=-1)
        semantic_relation = torch.softmax(torch.randn(2, 5, 6), dim=-1)
        with torch.no_grad():
            source_logits = model(
                features, mask,
                efpp_probs=semantic_efpp,
                etp_distribution=semantic_etp,
                relation_distribution=semantic_relation,
            )["recognition_logits"]
            restored_logits = restored(
                features, mask,
                efpp_probs=semantic_efpp,
                etp_distribution=semantic_etp,
                relation_distribution=semantic_relation,
            )["recognition_logits"]
        self.assertTrue(torch.allclose(source_logits, restored_logits, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
