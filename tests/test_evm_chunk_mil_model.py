import sys
import unittest
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evm_chunk_mil_model import (  # noqa: E402
    EVMChunkMILClassifier,
    LDETPCrossAttentionMIL,
    MLM8ViewMultiSlotMIL,
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

    def test_ld_etp_cross_attention_masks_padding_and_separates_roles(self):
        config = {
            "num_labels": 3, "label_names": ["a", "b", "c"], "feature_dim": 12,
            "hidden_dim": 16, "attn_dim": 8, "max_chunks": 3, "dropout": 0.0,
            "recognition_head_type": "label_branch_mlp", "label_branch_hidden_dim": 8,
            "label_branch_dropout": 0.0, "use_chunk_context": True,
            "chunk_context_num_layers": 1, "chunk_context_num_heads": 4,
            "chunk_context_dropout": 0.0, "detection_head_enabled": False,
            "detection_loss_weight": 0.0, "num_effect_types": 17,
            "etp_embedding_dim": 8, "etp_query_slots": 2, "etp_max_len": 7,
            "etp_cross_attention_heads": 2, "lambda_sep": 0.05,
        }
        model = LDETPCrossAttentionMIL(config).eval()
        features = torch.randn(2, 3, 12)
        mask = torch.tensor([[True, True, False], [True, False, False]])
        role_ids = torch.full((2, 3, 7, 2), 255, dtype=torch.uint8)
        confidence = torch.zeros_like(role_ids)
        role_ids[:, :, 1, 0] = 1
        confidence[:, :, 1, 0] = 255
        outputs = model(features, mask, role_ids, confidence,
                        binary_label=torch.ones(2),
                        multi_labels=torch.tensor([[1., 0., 1.], [0., 1., 0.]]),
                        return_attention=True)
        self.assertEqual(tuple(outputs["recognition_logits"].shape), (2, 3))
        self.assertEqual(tuple(outputs["token_attention"].shape), (2, 3, 3, 2, 7))
        self.assertTrue(torch.isfinite(outputs["loss"]))
        self.assertTrue(torch.allclose(outputs["chunk_attention"].sum(dim=1), torch.ones(2, 3), atol=1e-6))
        outputs["loss"].backward()
        self.assertIsNotNone(model.etp_embedding.weight.grad)

    def test_ld_etp_separation_uses_only_discordant_label_pairs(self):
        config = {
            "num_labels": 3, "label_names": ["a", "b", "c"], "feature_dim": 12,
            "hidden_dim": 16, "attn_dim": 8, "max_chunks": 2, "dropout": 0.0,
            "recognition_head_type": "label_branch_mlp", "label_branch_hidden_dim": 8,
            "use_chunk_context": False, "detection_head_enabled": False,
            "num_effect_types": 17, "etp_embedding_dim": 8, "etp_query_slots": 2,
            "etp_max_len": 7, "etp_cross_attention_heads": 2, "lambda_sep": 0.05,
        }
        model = LDETPCrossAttentionMIL(config)
        profiles = torch.tensor([[[1., 0.], [1., 0.], [0., 1.]]])
        self.assertEqual(model._separation_loss(profiles, torch.tensor([[1., 1., 1.]])).item(), 0.0)
        self.assertGreater(model._separation_loss(profiles, torch.tensor([[1., 0., 1.]])).item(), 0.0)

    def test_mlm8view_multislot_masks_padding_and_normalizes_attention(self):
        config = {
            "num_labels": 3, "label_names": ["a", "b", "c"], "num_views": 8,
            "feature_dim": 12, "hidden_dim": 16, "attn_dim": 8, "max_chunks": 3,
            "dropout": 0.0, "recognition_head_type": "label_branch_mlp",
            "label_branch_hidden_dim": 8, "label_branch_dropout": 0.0,
            "use_chunk_context": True, "chunk_context_num_layers": 1,
            "chunk_context_num_heads": 4, "chunk_context_dropout": 0.0,
            "detection_head_enabled": False, "detection_loss_weight": 0.0,
            "label_query_slots": 3, "view_residual_init": 0.1,
        }
        model = MLM8ViewMultiSlotMIL(config).eval()
        features = torch.randn(2, 3, 8, 12)
        mask = torch.tensor([[True, True, False], [True, False, False]])
        outputs = model(
            features, mask,
            binary_label=torch.ones(2),
            multi_labels=torch.tensor([[1., 0., 1.], [0., 1., 0.]]),
            return_attention=True,
        )
        self.assertEqual(tuple(outputs["recognition_logits"].shape), (2, 3))
        self.assertEqual(tuple(outputs["view_attention"].shape), (2, 3, 3, 3, 8))
        self.assertTrue(torch.isfinite(outputs["loss"]))
        self.assertTrue(torch.allclose(outputs["view_attention"].sum(dim=-1), torch.ones(2, 3, 3, 3), atol=1e-6))
        self.assertTrue(torch.allclose(outputs["slot_chunk_attention"].sum(dim=1), torch.ones(2, 3, 3), atol=1e-6))
        changed = features.clone()
        changed[~mask] = 10000.0
        self.assertTrue(torch.allclose(outputs["recognition_logits"], model(changed, mask)["recognition_logits"], atol=1e-6))


if __name__ == "__main__":
    unittest.main()
