"""Unit tests for the Phase 2 parser and local abstract stack."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spor.evm_parser import parse_disassembled_opcode
from spor.stack_provenance import SOURCE_BITS, analyze_basic_blocks


class SporTest(unittest.TestCase):
    def parse(self, text):
        instructions, blocks, meta = parse_disassembled_opcode(text)
        analyze_basic_blocks(instructions, blocks)
        return instructions, blocks, meta

    def test_push_widths_and_no_immediate_instruction(self):
        instructions, blocks, _ = self.parse("PUSH0 PUSH1 0x01 PUSH2 0x0102 PUSH20 0x" + "11" * 20 + " PUSH32 0x" + "22" * 32)
        self.assertEqual([item.opcode for item in instructions], ["PUSH0", "PUSH1", "PUSH2", "PUSH20", "PUSH32"])
        self.assertEqual([item.immediate_width for item in instructions], [0, 1, 2, 20, 32])
        self.assertEqual([item.derived_pc for item in instructions], [0, 1, 3, 6, 27])

    def test_dup_and_swap_order(self):
        instructions, _, _ = self.parse("PUSH1 0x01 PUSH1 0x02 DUP2 SWAP1 POP")
        self.assertEqual(instructions[2].operation_role, "STACK_DUP")
        self.assertEqual(instructions[3].operation_role, "STACK_SWAP")
        self.assertEqual(instructions[4].operation_role, "STACK_POP")

    def test_multi_source_comparison(self):
        instructions, _, _ = self.parse("CALLER PUSH1 0x01 EQ ISZERO")
        comparison = instructions[2]
        self.assertEqual(comparison.operation_role, "COMPARISON")
        sources = comparison.operand_sources["operands"]
        self.assertTrue(sources[0].provenance_bits & SOURCE_BITS["CONSTANT"])
        self.assertTrue(sources[1].provenance_bits & SOURCE_BITS["CALLER"])
        self.assertEqual(instructions[3].operation_role, "COMPARISON")

    def test_jumpi_destination_and_condition_are_separate(self):
        instructions, _, _ = self.parse("CALLER ISZERO PUSH1 0x20 JUMPI")
        jump = instructions[-1]
        self.assertEqual(jump.operation_role, "CONTROL_FLOW")
        self.assertIn("destination", jump.operand_sources)
        self.assertIn("condition", jump.operand_sources)
        self.assertTrue(jump.operand_sources["destination"].provenance_bits & SOURCE_BITS["CONSTANT"])

    def test_call_and_sstore_operands(self):
        call_text = " ".join(f"PUSH1 0x{i:02x}" for i in range(1, 8)) + " CALL"
        instructions, _, _ = self.parse(call_text)
        call = instructions[-1]
        self.assertEqual(call.operation_role, "CALL")
        self.assertEqual(set(call.operand_sources), {"gas", "target", "value", "input_offset", "input_size", "output_offset", "output_size"})
        instructions, _, _ = self.parse("PUSH1 0x01 PUSH1 0x02 SSTORE")
        self.assertEqual(set(instructions[-1].operand_sources), {"key", "value"})
        instructions, _, _ = self.parse("PUSH1 0x01 PUSH1 0x02 SSTORE")
        self.assertEqual(instructions[-1].operand_sources["value"].known_constant, 2)
        self.assertEqual(instructions[-1].operand_sources["key"].known_constant, 1)

    def test_call_operand_order(self):
        call_text = " ".join(f"PUSH1 0x{i:02x}" for i in range(1, 8)) + " CALL"
        instructions, _, _ = self.parse(call_text)
        call = instructions[-1]
        self.assertEqual(call.operand_sources["gas"].known_constant, 1)
        self.assertEqual(call.operand_sources["target"].known_constant, 2)
        self.assertEqual(call.operand_sources["output_size"].known_constant, 7)

    def test_unknown_annotation_alignment(self):
        from evm_tokenizer import EVMOpcodeTokenizer
        tokenizer = EVMOpcodeTokenizer.from_vocab_file(ROOT / "data/processed/ethereum_public_pretrain_19143_unique_runtime/evm_vocab.json")
        instructions, _, meta = parse_disassembled_opcode("PUSH1 0x01 'FE' (Unknown Opcode) POP", tokenizer)
        self.assertEqual(instructions[1].parse_status, "unknown_opcode_annotation")
        self.assertEqual(instructions[1].source_token_indices, [2, 3, 4])
        self.assertEqual(len(instructions[1].model_token_indices), 3)
        self.assertEqual(meta["token_count"], sum(len(item.model_token_indices) for item in instructions))

    def test_blocks_do_not_share_stack(self):
        instructions, blocks, _ = self.parse("CALLER JUMPDEST POP")
        self.assertEqual(len(blocks), 2)
        self.assertEqual(instructions[-1].analysis_status, "unknown_entry")

    def test_unknown_instruction_is_failure_not_empty(self):
        instructions, _, _ = self.parse("PUSH1 0x01 UNKNOWN_OPCODE POP")
        self.assertEqual(instructions[1].analysis_status, "analysis_failure_unsupported_stack_effect")
        self.assertNotEqual(instructions[2].analysis_status, "known")

    def test_real_token_alignment(self):
        from evm_tokenizer import EVMOpcodeTokenizer
        tokenizer = EVMOpcodeTokenizer.from_vocab_file(ROOT / "data/processed/ethereum_public_pretrain_19143_unique_runtime/evm_vocab.json")
        instructions, _, meta = parse_disassembled_opcode("PUSH1 0x80 ADD PUSH0", tokenizer)
        self.assertTrue(meta["token_count"] >= 4)
        self.assertEqual(instructions[0].model_token_indices, [0, 1])
        self.assertEqual(instructions[1].model_token_indices, [2])
        self.assertEqual(instructions[2].model_token_indices, [3])


if __name__ == "__main__":
    unittest.main()
