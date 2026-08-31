import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evm_stack_relation import RELATION_INDEX, analyze_stack_relations, pack_contract_relations  # noqa: E402
from stack_relation_mlm_dataset import StackRelationMLMDataset  # noqa: E402


class TinyTokenizer:
    pad_token_id = 0
    cls_token = "[CLS]"
    sep_token = "[SEP]"
    vocab = {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2}

    def tokenize(self, text, add_special_tokens=False):
        return str(text).split()

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab.setdefault(token, len(self.vocab)) for token in tokens]


def test_direct_stack_lineage_has_operand_slots():
    result = analyze_stack_relations("PUSH1 0x01 PUSH1 0x02 ADD SSTORE", TinyTokenizer())
    direct = [item for item in result["relations"] if item["relation"] == "direct_forward"]
    assert len(result["relations"]) >= 3
    assert len(direct) >= 2
    assert {item["slot"] for item in direct} >= {0, 1}


def test_dup_and_swap_preserve_execution_identity():
    result = analyze_stack_relations("PUSH1 0x01 DUP1 SWAP1 ADD", TinyTokenizer())
    names = {item["relation"] for item in result["relations"]}
    assert "alias" in names
    assert "reorder" in names


def test_dynamic_jump_does_not_guess_target():
    result = analyze_stack_relations("CALLER JUMP", TinyTokenizer())
    control = [item for item in result["relations"] if item["relation"] == "control_operand"]
    assert len(control) == 1
    assert control[0]["producer"] == 0


def test_packed_relations_do_not_include_padding_edges():
    tokenizer = TinyTokenizer()
    chunks, boundary, analysis = pack_contract_relations("PUSH1 0x01 ADD", tokenizer, max_len=16, chunk_stride=8, max_chunks=2)
    assert chunks
    assert all(0 < edge[0] < 16 and 0 < edge[1] < 16 for chunk in chunks for edge in chunk["edges"])
    assert len(boundary) == len(chunks)
    assert analysis["report"]["token_count"] == 3


def test_cross_chunk_relation_is_summarized_through_cls():
    tokenizer = TinyTokenizer()
    chunks, _, _ = pack_contract_relations(
        "PUSH1 0x01 PUSH1 0x02 ADD SSTORE",
        tokenizer,
        max_len=5,
        chunk_stride=3,
        max_chunks=3,
    )
    # The second chunk starts at token 3, so SSTORE's producer is outside it.
    assert any(edge[0] == 0 and edge[1] > 0 for edge in chunks[1]["edges"])


def test_empty_opcode_chunk_has_no_mlm_targets_instead_of_nan():
    # The trainer must skip this batch; the model loss path separately guards
    # the all-ignore-index case for callers that still execute the forward.
    path = Path(__file__).with_name("_empty_stack_cache.pt")
    payload = {
        "schema": "main6_opcode_stack_relational_v2",
        "input_ids": torch.tensor([[1, 2, 0, 0]], dtype=torch.int32),
        "attention_mask": torch.tensor([[True, True, False, False]]),
        "stack_state": torch.zeros((1, 4, 5), dtype=torch.uint8),
        "boundary_state": torch.zeros((1, 4), dtype=torch.uint8),
        "edge_offsets": torch.tensor([0, 0]),
        "edge_src": torch.empty(0, dtype=torch.int16),
        "edge_dst": torch.empty(0, dtype=torch.int16),
        "edge_type": torch.empty(0, dtype=torch.uint8),
        "edge_slot": torch.empty(0, dtype=torch.uint8),
        "edge_distance": torch.empty(0, dtype=torch.uint8),
        "edge_confidence": torch.empty(0, dtype=torch.float16),
        "multi_labels": torch.zeros((1, 6)),
        "binary_labels": torch.zeros(1),
    }
    torch.save(payload, path)
    try:
        dataset = StackRelationMLMDataset(path)
        dataset.special_ids = {0, 1, 2, 3, 4}
        dataset.set_random_token_ids([], 4)
        assert int(dataset[0]["labels"].ne(-100).sum()) == 0
    finally:
        path.unlink(missing_ok=True)
