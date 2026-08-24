"""Label-agnostic local EVM stack analysis for the execution-aware route.

The analyzer is intentionally conservative. It tracks producer instruction
indices, not concrete 256-bit values, and turns underflow/unsupported effects
into unknown values instead of inventing dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from evm_control_stack_graph import (
    DUP_OPCODES,
    PUSH_OPCODES,
    SWAP_OPCODES,
    _raw_instructions,
    _stack_effect,
    build_token_spans,
)


ROLE_NAMES = [
    "producer", "consumer", "unary_transform", "binary_transform",
    "multi_consumer", "reordering", "control_operand", "external_interaction",
    "reader", "writer",
]
ROLE_INDEX = {name: index for index, name in enumerate(ROLE_NAMES)}

PROVENANCE_NAMES = [
    "constant", "calldata", "caller", "origin", "callvalue", "environment",
    "storage_read", "memory_read", "external_return", "arithmetic_result",
]
PROVENANCE_INDEX = {name: index for index, name in enumerate(PROVENANCE_NAMES)}


@dataclass
class StackValue:
    producers: Tuple[int, ...] = ()
    provenance: Tuple[int, ...] = ()
    unknown: bool = False


def _unknown() -> StackValue:
    return StackValue(unknown=True)


def _merge_producers(values: Sequence[StackValue], max_producers: int) -> StackValue:
    producers = sorted({p for value in values for p in value.producers})
    unknown = any(value.unknown for value in values) or len(producers) > max_producers
    return StackValue(tuple(producers[:max_producers] if not unknown else ()), (), unknown)


def _roles(opcode: str, pop: int, push: int) -> List[int]:
    roles = []
    if push:
        roles.append(ROLE_INDEX["producer"])
    if pop:
        roles.append(ROLE_INDEX["consumer"])
    if pop == 1 and push == 1:
        roles.append(ROLE_INDEX["unary_transform"])
    if pop == 2 and push == 1:
        roles.append(ROLE_INDEX["binary_transform"])
    if pop >= 3:
        roles.append(ROLE_INDEX["multi_consumer"])
    if opcode in DUP_OPCODES or opcode in SWAP_OPCODES:
        roles.append(ROLE_INDEX["reordering"])
    if opcode in {"JUMP", "JUMPI"}:
        roles.append(ROLE_INDEX["control_operand"])
    if opcode in {"CALL", "CALLCODE", "DELEGATECALL", "STATICCALL", "CREATE", "CREATE2", "EXTCODESIZE", "EXTCODEHASH", "BALANCE"}:
        roles.append(ROLE_INDEX["external_interaction"])
    if opcode in {"SLOAD", "MLOAD", "CALLDATALOAD", "RETURNDATASIZE", "BLOCKHASH"}:
        roles.append(ROLE_INDEX["reader"])
    if opcode in {"SSTORE", "MSTORE", "MSTORE8", "CALLDATACOPY", "CODECOPY", "RETURNDATACOPY"}:
        roles.append(ROLE_INDEX["writer"])
    return sorted(set(roles))


def _provenance(opcode: str) -> Tuple[int, ...]:
    mapping = {
        "CALLER": "caller", "ORIGIN": "origin", "CALLVALUE": "callvalue",
        "CALLDATALOAD": "calldata", "SLOAD": "storage_read", "MLOAD": "memory_read",
    }
    if opcode in mapping:
        return (PROVENANCE_INDEX[mapping[opcode]],)
    if opcode in {"TIMESTAMP", "NUMBER", "BLOCKHASH", "BASEFEE", "GASLIMIT", "CHAINID", "COINBASE", "PREVRANDAO"}:
        return (PROVENANCE_INDEX["environment"],)
    return ()


def analyze_opcode_execution(
    opcode_text: str,
    tokenizer=None,
    max_producers: int = 4,
    max_stack_depth: int = 1024,
) -> Dict:
    token_spans = None
    token_count = 0
    if tokenizer is not None:
        _, token_spans = build_token_spans(opcode_text, tokenizer)
        token_count = max((end for _, end in token_spans), default=0)
    instructions = _raw_instructions(opcode_text, token_spans)
    role = np.zeros((token_count, len(ROLE_NAMES)), dtype=np.float32)
    numeric = np.zeros((token_count, 8), dtype=np.float32)
    provenance = np.zeros((token_count, len(PROVENANCE_NAMES)), dtype=np.float32)
    dependencies = []
    stack: List[StackValue] = []
    max_height = 0
    underflow_count = 0
    unknown_dependency_count = 0

    for instruction in instructions:
        opcode = instruction.opcode
        pop, push = _stack_effect(opcode)
        before = len(stack)
        consumed = []
        for _ in range(pop):
            if stack:
                consumed.append(stack.pop())
            else:
                consumed.append(_unknown())
                underflow_count += 1
        for value in consumed:
            for producer in value.producers:
                dependencies.append({
                    "producer": int(producer),
                    "consumer": int(instruction.index),
                    "distance": int(instruction.index - producer),
                    "confidence": 0.0 if value.unknown else 1.0,
                })
        if any(value.unknown for value in consumed):
            unknown_dependency_count += 1
        roles = _roles(opcode, pop, push)
        prov = _provenance(opcode)
        start, end = instruction.token_start, instruction.token_end
        if end > start and token_count:
            role[start:end, roles] = 1.0
            numeric[start:end, 0] = float(min(pop, 7))
            numeric[start:end, 1] = float(min(push, 3))
            numeric[start:end, 2] = min(before, max_stack_depth) / float(max_stack_depth)
            numeric[start:end, 3] = min(before - pop + push, max_stack_depth) / float(max_stack_depth)
            numeric[start:end, 4] = float(sum(bool(value.producers) for value in consumed))
            numeric[start:end, 5] = float(sum(value.unknown for value in consumed))
            numeric[start:end, 6] = float(max((instruction.index - p for value in consumed for p in value.producers), default=0))
            numeric[start:end, 7] = float(np.mean([1.0 if not value.unknown else 0.0 for value in consumed]) if consumed else 1.0)
            if prov:
                provenance[start:end, list(prov)] = 1.0
        if opcode in PUSH_OPCODES:
            produced_prov = (PROVENANCE_INDEX["constant"],)
        elif opcode in {"CALL", "CALLCODE", "DELEGATECALL", "STATICCALL"}:
            produced_prov = (PROVENANCE_INDEX["external_return"],)
        elif opcode in {"ADD", "MUL", "SUB", "DIV", "SDIV", "MOD", "SMOD", "EXP", "ADDMOD", "MULMOD", "SHL", "SHR", "SAR", "AND", "OR", "XOR", "EQ", "LT", "GT", "SLT", "SGT", "ISZERO", "NOT"}:
            produced_prov = (PROVENANCE_INDEX["arithmetic_result"],)
        else:
            produced_prov = ()
        if opcode in DUP_OPCODES:
            depth = int(opcode[3:])
            value = stack[-depth] if len(stack) >= depth else _unknown()
            stack.append(value)
        elif opcode in SWAP_OPCODES:
            depth = int(opcode[4:])
            if len(stack) > depth:
                stack[-1], stack[-1 - depth] = stack[-1 - depth], stack[-1]
        else:
            for _ in range(push):
                stack.append(StackValue((instruction.index,), produced_prov, False))
        if len(stack) > max_stack_depth:
            stack = stack[-max_stack_depth:]
        max_height = max(max_height, len(stack))

    feature = np.concatenate([role, numeric, provenance], axis=1)
    return {
        "token_features": feature,
        "token_count": token_count,
        "dependencies": dependencies,
        "report": {
            "instruction_count": len(instructions),
            "token_count": token_count,
            "dependency_count": len(dependencies),
            "max_stack_height": max_height,
            "underflow_count": underflow_count,
            "unknown_dependency_count": unknown_dependency_count,
            "role_names": ROLE_NAMES,
            "provenance_names": PROVENANCE_NAMES,
        },
    }


FEATURE_NAMES = (
    [f"role_{name}" for name in ROLE_NAMES]
    + ["pop_count", "push_count", "stack_height_before", "stack_height_after", "known_producer_count", "unknown_producer_count", "max_dependency_distance", "dependency_confidence"]
    + [f"provenance_{name}" for name in PROVENANCE_NAMES]
)
