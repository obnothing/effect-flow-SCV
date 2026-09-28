"""Focused stack-effect and partial-CFG invariants for SPOR Phase 2.5."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spor.evm_parser import parse_disassembled_opcode
from spor.phase25 import analyze_partial_cfg, join_stacks
from spor.stack_effects import stack_effect
from spor.stack_provenance import AbstractValue, SOURCE_BITS


class Phase25Test(unittest.TestCase):
    def analyze(self, text):
        ins, blocks, _ = parse_disassembled_opcode(text)
        stats = analyze_partial_cfg(ins, blocks)
        return ins, blocks, stats

    def test_registry_heights(self):
        expected = {"BALANCE": (1, 1), "CALLDATACOPY": (3, 0), "EXTCODECOPY": (4, 0),
                    "LOG4": (6, 0), "CALL": (7, 1), "DELEGATECALL": (6, 1),
                    "DUP16": (0, 1), "SWAP16": (0, 0), "PUSH0": (0, 1)}
        for op, counts in expected.items():
            effect = stack_effect(op)
            self.assertEqual((effect.pop_count, effect.push_count), counts)
        self.assertIsNone(stack_effect("UNKNOWN_0x4F"))

    def test_push0_zero_width_placeholder_is_not_an_instruction(self):
        from evm_tokenizer import EVMOpcodeTokenizer
        tokenizer = EVMOpcodeTokenizer.from_vocab_file(ROOT / "data/processed/ethereum_public_pretrain_19143_unique_runtime/evm_vocab.json")
        ins, _, _ = parse_disassembled_opcode("PUSH0 0x PUSH1 0x01 ADD", tokenizer)
        self.assertEqual([item.opcode for item in ins], ["PUSH0", "PUSH1", "ADD"])
        self.assertEqual(ins[0].immediate_width, 0)
        self.assertEqual(ins[0].model_token_indices, [0, 1])

    def test_known_effect_unknown_provenance_keeps_height(self):
        ins, _, _ = self.analyze("PUSH1 0x01 BALANCE PUSH1 0x02 SSTORE")
        store = ins[-1]
        self.assertEqual(store.operand_sources["key"].known_constant, 2)
        self.assertEqual(store.operand_sources["value"].analysis_status, "known_stack_effect_but_unknown_provenance")
        self.assertEqual(store.failure_reason, "known_stack_effect_but_unknown_provenance")

    def test_gas_is_known_environment_source(self):
        ins, _, _ = self.analyze("PUSH1 0x02 PUSH1 0x03 PUSH1 0x04 PUSH1 0x05 PUSH1 0x06 PUSH1 0x07 GAS CALL")
        call = ins[-1]
        self.assertEqual(call.operand_sources["gas"].provenance_bits, SOURCE_BITS["BLOCK_ENV"])
        self.assertEqual(call.operand_sources["gas"].analysis_status, "known")

    def test_static_jump_propagates_stack(self):
        ins, blocks, stats = self.analyze("PUSH1 0x2a PUSH1 0x06 JUMP STOP JUMPDEST POP")
        self.assertEqual(stats["edges"]["static_jump"], 1)
        self.assertEqual(ins[-1].operand_sources["operands"][0].known_constant, 42)

    def test_jumpi_has_fallthrough_and_static_target(self):
        ins, _, stats = self.analyze("PUSH1 0x2a PUSH1 0x01 PUSH1 0x09 JUMPI POP STOP JUMPDEST POP")
        self.assertEqual(stats["edges"]["jumpi_fallthrough"], 1)
        self.assertEqual(stats["edges"]["static_jumpi"], 1)
        self.assertEqual(ins[4].operand_sources["operands"][0].known_constant, 42)
        self.assertEqual(ins[-1].operand_sources["operands"][0].known_constant, 42)

    def test_jump_target_resolved_after_predecessor_stack_propagation(self):
        _, _, stats = self.analyze("PUSH1 0x0a PUSH1 0x01 PUSH1 0x08 JUMPI JUMP JUMPDEST STOP JUMPDEST STOP")
        self.assertEqual(stats["edges"]["static_jumpi"], 1)
        self.assertEqual(stats["edges"]["static_jump"], 1)
        self.assertEqual(stats["unresolved_jumps"], 0)

    def test_dynamic_jump_does_not_create_target_edge(self):
        _, _, stats = self.analyze("CALLER JUMP JUMPDEST POP")
        self.assertEqual(stats["unresolved_jumps"], 1)
        self.assertEqual(stats["edges"].get("static_jump", 0), 0)

    def test_equal_height_join_and_conflict(self):
        a = AbstractValue(SOURCE_BITS["CALLER"], "CALLER")
        b = AbstractValue(SOURCE_BITS["CALLVALUE"], "CALLVALUE")
        joined, reason = join_stacks([(a,), (b,)])
        self.assertIsNone(reason)
        self.assertEqual(joined[0].provenance_bits, a.provenance_bits | b.provenance_bits)
        self.assertIsNone(joined[0].known_constant)
        joined, reason = join_stacks([(a,), ()])
        self.assertIsNone(joined)
        self.assertEqual(reason, "predecessor_stack_height_conflict")

    def test_unknown_effect_poison_and_parser_failure(self):
        ins, blocks, _ = self.analyze("UNKNOWN_0x4F PUSH1 0x01 JUMPDEST PUSH1 0x02")
        self.assertEqual(ins[0].failure_reason, "unsupported_opcode_stack_effect")
        self.assertEqual(ins[1].failure_reason, "previous_analysis_failure")
        self.assertEqual(ins[-1].analysis_status, "known")
        ins, _, _ = self.analyze("PUSH2 0x01 POP")
        self.assertEqual(ins[0].failure_reason, "parser_failure")


if __name__ == "__main__":
    unittest.main()
