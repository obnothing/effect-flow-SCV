import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evm_control_stack_graph import EDGE_TYPES, build_evm_graph  # noqa: E402
from evm_opcode_graph_residual_mil import OpcodeGraphResidualMIL  # noqa: E402


class SimpleTokenizer:
    def tokenize(self, value, add_special_tokens=False):
        return str(value).split()


def model_config():
    return {
        "num_labels": 6,
        "label_names": ["a", "b", "c", "d", "e", "f"],
        "feature_dim": 768,
        "hidden_dim": 512,
        "attn_dim": 256,
        "max_chunks": 64,
        "num_views": 8,
        "dropout": 0.0,
        "task_mode": "recognition_only",
        "detection_head_enabled": False,
        "detection_loss_weight": 0.0,
        "recognition_loss_weight": 1.0,
        "recognition_head_type": "label_branch_mlp",
        "label_branch_hidden_dim": 256,
        "label_branch_dropout": 0.0,
        "label_branch_use_layernorm": True,
        "use_chunk_context": False,
        "label_query_slots": 3,
        "graph_hidden_dim": 32,
        "graph_query_slots": 3,
        "graph_dropout": 0.0,
        "graph_edge_type_mask": [0, 1, 2, 3, 4, 5],
    }


class OpcodeGraphTest(unittest.TestCase):
    def test_direct_jump_and_basic_block_ranges(self):
        graph = build_evm_graph("PUSH1 0x03 JUMP JUMPDEST STOP", SimpleTokenizer())
        self.assertEqual(graph["report"]["basic_block_count"], 2)
        self.assertEqual(graph["report"]["direct_jump_count"], 1)
        self.assertTrue(any(edge["type"] == EDGE_TYPES["direct_jump"] for edge in graph["edges"]))
        self.assertEqual(graph["report"]["token_coverage"], 1.0)

    def test_dynamic_jump_is_not_guessed(self):
        graph = build_evm_graph("CALLER JUMP JUMPDEST STOP", SimpleTokenizer())
        self.assertEqual(graph["report"]["direct_jump_count"], 0)
        self.assertEqual(graph["report"]["unresolved_jump_count"], 1)

    def test_cross_block_stack_dependency_is_recorded(self):
        graph = build_evm_graph(
            "PUSH1 0x01 PUSH1 0x05 JUMP JUMPDEST PUSH1 0x00 SSTORE STOP",
            SimpleTokenizer(),
        )
        self.assertGreaterEqual(graph["report"]["stack_edge_count"], 1)
        value_graph = graph["instruction_value"]
        self.assertTrue(any(edge["type"] == EDGE_TYPES["value_produces"] for edge in value_graph["edges"]))
        self.assertTrue(any(node["node_type"] == 1 for node in value_graph["nodes"]))

    def test_stack_analysis_budget_caps_pathological_loop(self):
        graph = build_evm_graph(
            "JUMPDEST PUSH1 0x00 DUP1 JUMP",
            max_worklist_steps=20000,
            max_instruction_visits=64,
        )
        self.assertFalse(graph["report"]["stack_analysis_capped"])
        self.assertEqual(graph["report"]["basic_block_count"], 1)
        capped = build_evm_graph(
            "JUMPDEST PUSH1 0x00 DUP1 JUMP",
            max_worklist_steps=0,
            max_instruction_visits=64,
        )
        self.assertTrue(capped["report"]["stack_analysis_capped"])

    def test_zero_residual_reproduces_sequence_logits(self):
        config = model_config()
        model = OpcodeGraphResidualMIL(config).eval()
        sequence = torch.randn(2, 4, 8, 768)
        mask = torch.ones(2, 4, dtype=torch.bool)
        nodes = [torch.randn(3, 768), torch.randn(1, 768)]
        node_mask = [torch.ones(3, dtype=torch.bool), torch.ones(1, dtype=torch.bool)]
        edges = [torch.tensor([[0, 1], [1, 2]]), torch.empty((2, 0), dtype=torch.long)]
        types = [torch.tensor([0, 4]), torch.empty((0,), dtype=torch.long)]
        result = model(sequence, mask, nodes, node_mask, edges, types)
        baseline = model.sequence_branch(sequence, mask)["recognition_logits"]
        self.assertTrue(torch.equal(result["recognition_logits"], baseline))

    def test_sparse_attention_does_not_select_padding(self):
        config = model_config()
        config["graph_sparse_topk"] = 2
        model = OpcodeGraphResidualMIL(config).eval()
        sequence = torch.randn(1, 2, 8, 768)
        mask = torch.ones(1, 2, dtype=torch.bool)
        nodes = [torch.randn(4, 768)]
        node_mask = [torch.tensor([True, True, False, False])]
        edges = [torch.empty((2, 0), dtype=torch.long)]
        types = [torch.empty((0,), dtype=torch.long)]
        output = model(sequence, mask, nodes, node_mask, edges, types, return_attention=True)
        selected = output["graph_attention"][0]["selected_nodes"]
        if selected is not None:
            self.assertFalse(bool(selected[:, 2:].any().item()))
        attention = output["graph_attention"][0]["node_attention"]
        self.assertTrue(torch.allclose(attention[:, :, 2:], torch.zeros_like(attention[:, :, 2:])))

    def test_empty_graph_is_finite(self):
        model = OpcodeGraphResidualMIL(model_config()).eval()
        sequence = torch.randn(1, 2, 8, 768)
        mask = torch.ones(1, 2, dtype=torch.bool)
        output = model(
            sequence,
            mask,
            [torch.empty((0, 768))],
            [torch.empty((0,), dtype=torch.bool)],
            [torch.empty((2, 0), dtype=torch.long)],
            [torch.empty((0,), dtype=torch.long)],
        )
        self.assertTrue(torch.isfinite(output["recognition_logits"]).all())

    def test_graph_half_precision_mask_is_finite(self):
        config = model_config()
        model = OpcodeGraphResidualMIL(config).eval().half()
        output = model(
            torch.randn(1, 2, 8, 768).half(),
            torch.ones(1, 2, dtype=torch.bool),
            [torch.randn(3, 768).half()],
            [torch.tensor([True, True, False])],
            [torch.tensor([[0, 1], [1, 0]], dtype=torch.long)],
            [torch.tensor([0, 1], dtype=torch.long)],
        )
        self.assertTrue(torch.isfinite(output["recognition_logits"]).all())

    def test_graph_mixed_precision_edge_aggregation_is_finite(self):
        config = model_config()
        model = OpcodeGraphResidualMIL(config).eval()
        output = model(
            torch.randn(1, 2, 8, 768),
            torch.ones(1, 2, dtype=torch.bool),
            [torch.randn(3, 768)],
            [torch.tensor([True, True, False])],
            [torch.tensor([[0, 1], [1, 0]], dtype=torch.long)],
            [torch.tensor([0, 1], dtype=torch.long)],
        )
        self.assertTrue(torch.isfinite(output["recognition_logits"]).all())

    def test_dynamic_gate_is_labelwise_and_bounded(self):
        config = model_config()
        config["fusion_mode"] = "dynamic_convex"
        model = OpcodeGraphResidualMIL(config).eval()
        output = model(
            torch.randn(2, 2, 8, 768), torch.ones(2, 2, dtype=torch.bool),
            [torch.randn(2, 768), torch.randn(2, 768)],
            [torch.ones(2, dtype=torch.bool), torch.ones(2, dtype=torch.bool)],
            [torch.tensor([[0], [1]]), torch.tensor([[0], [1]])],
            [torch.tensor([4]), torch.tensor([4])],
        )
        self.assertEqual(tuple(output["dynamic_gate"].shape), (2, 6))
        self.assertTrue(bool(((output["dynamic_gate"] >= 0) & (output["dynamic_gate"] <= 1)).all()))

    def test_local_pooling_accepts_ragged_token_features(self):
        config = model_config()
        config["graph_pooling"] = "local_attention"
        model = OpcodeGraphResidualMIL(config).eval()
        output = model(
            torch.randn(1, 2, 8, 768), torch.ones(1, 2, dtype=torch.bool),
            [torch.randn(2, 768)], [torch.ones(2, dtype=torch.bool)],
            [torch.tensor([[0], [1]])], [torch.tensor([4])],
            node_local_features=[torch.randn(5, 768)], node_local_offsets=[torch.tensor([0, 2, 5])],
        )
        self.assertTrue(torch.isfinite(output["recognition_logits"]).all())


if __name__ == "__main__":
    unittest.main()
