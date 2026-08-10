"""Opcode-native EVM control-flow and stack dependency graphs.

The graph is deliberately label agnostic. It uses only disassembled runtime
opcodes and conservative abstract stack propagation; unresolved dynamic jumps
never receive guessed targets.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


EDGE_TYPES = {
    "fallthrough": 0,
    "conditional_true": 1,
    "conditional_false": 2,
    "direct_jump": 3,
    "stack_def_use": 4,
    "storage_dependency": 5,
}

TERMINATORS = {"STOP", "RETURN", "REVERT", "SELFDESTRUCT", "INVALID", "JUMP"}
PUSH_OPCODES = {"PUSH0"} | {f"PUSH{i}" for i in range(1, 33)}
DUP_OPCODES = {f"DUP{i}" for i in range(1, 17)}
SWAP_OPCODES = {f"SWAP{i}" for i in range(1, 17)}


@dataclass(frozen=True)
class Instruction:
    index: int
    pc: int
    opcode: str
    operand: Optional[int]
    raw_start: int
    raw_end: int
    token_start: int
    token_end: int


@dataclass(frozen=True)
class AbstractValue:
    producers: frozenset
    constant: Optional[int] = None


def _parse_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text.startswith("0x"):
        return None
    try:
        return int(text, 16)
    except ValueError:
        return None


def _raw_instructions(opcode_text: str, token_spans: Optional[Sequence[Tuple[int, int]]] = None):
    raw = str(opcode_text or "").strip().split()
    result = []
    raw_index = 0
    pc = 0
    while raw_index < len(raw):
        opcode = raw[raw_index].upper()
        operand = None
        raw_end = raw_index + 1
        size = 1
        if opcode in PUSH_OPCODES:
            width = 0 if opcode == "PUSH0" else int(opcode[4:])
            if width and raw_index + 1 < len(raw):
                operand = _parse_int(raw[raw_index + 1])
                raw_end += 1
                size += width
        token_start, token_end = (0, 0)
        if token_spans is not None and raw_index < len(token_spans):
            token_start, token_end = token_spans[raw_index]
        result.append(
            Instruction(
                index=len(result),
                pc=pc,
                opcode=opcode,
                operand=operand,
                raw_start=raw_index,
                raw_end=raw_end,
                token_start=token_start,
                token_end=token_end,
            )
        )
        pc += size
        raw_index = raw_end
    return result


def build_token_spans(opcode_text: str, tokenizer) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Return tokenizer tokens and raw-token spans for instruction alignment."""
    raw = str(opcode_text or "").strip().split()
    output: List[str] = []
    spans: List[Tuple[int, int]] = [(0, 0)] * len(raw)
    raw_index = 0
    while raw_index < len(raw):
        start = len(output)
        opcode = raw[raw_index].upper()
        if opcode in PUSH_OPCODES and raw_index + 1 < len(raw) and raw[raw_index + 1].lower().startswith("0x"):
            produced = tokenizer.tokenize(
                f"{opcode} {raw[raw_index + 1]}", add_special_tokens=False
            )
            output.extend(produced)
            spans[raw_index] = (start, len(output))
            spans[raw_index + 1] = (start, len(output))
            raw_index += 2
            continue
        produced = tokenizer.tokenize(raw[raw_index], add_special_tokens=False)
        output.extend(produced)
        spans[raw_index] = (start, len(output))
        raw_index += 1
    return output, spans


def _stack_effect(opcode: str) -> Tuple[int, int]:
    if opcode in PUSH_OPCODES:
        return 0, 1
    if opcode in DUP_OPCODES:
        return 0, 1
    if opcode in SWAP_OPCODES:
        return 0, 0
    if opcode == "POP":
        return 1, 0
    if opcode in {"ISZERO", "NOT", "SIGNEXTEND", "MLOAD", "SLOAD", "CALLDATALOAD", "BALANCE", "EXTCODESIZE", "EXTCODEHASH", "BLOCKHASH"}:
        return 1, 1
    if opcode in {"CALLER", "ORIGIN", "CALLVALUE", "CALLDATASIZE", "CODESIZE", "MSIZE", "GAS", "RETURNDATASIZE", "ADDRESS", "COINBASE", "TIMESTAMP", "NUMBER", "PREVRANDAO", "GASLIMIT", "CHAINID", "SELFBALANCE", "BASEFEE", "PC"}:
        return 0, 1
    if opcode in {"MSTORE", "MSTORE8", "SSTORE", "RETURN", "REVERT", "CALLDATACOPY", "CODECOPY", "RETURNDATACOPY"}:
        return {"MSTORE": 2, "MSTORE8": 2, "SSTORE": 2, "RETURN": 2, "REVERT": 2, "CALLDATACOPY": 3, "CODECOPY": 3, "RETURNDATACOPY": 3}[opcode], 0
    if opcode in {"JUMP", "SELFDESTRUCT"}:
        return 1, 0
    if opcode == "JUMPI":
        return 2, 0
    if opcode in {"CREATE", "CREATE2"}:
        return (4 if opcode == "CREATE2" else 3), 1
    if opcode in {"CALL", "CALLCODE", "DELEGATECALL", "STATICCALL"}:
        return 7, 1
    if opcode.startswith("LOG") and opcode[3:].isdigit():
        return int(opcode[3:]) + 2, 0
    if opcode in {"SHA3", "ADDMOD", "MULMOD"}:
        return (2 if opcode == "SHA3" else 3), 1
    if opcode in {"EXP"}:
        return 2, 1
    if opcode in {"ADD", "MUL", "SUB", "DIV", "SDIV", "MOD", "SMOD", "LT", "GT", "SLT", "SGT", "EQ", "AND", "OR", "XOR", "BYTE", "SHL", "SHR", "SAR"}:
        return 2, 1
    return 0, 0


def _merge_stack(old: Optional[List[AbstractValue]], new: List[AbstractValue], max_producers: int):
    if old is None:
        return list(new), True
    length = max(len(old), len(new))
    merged = []
    changed = len(old) != len(new)
    for index in range(length):
        values = []
        for stack in (old, new):
            if index < len(stack):
                values.append(stack[index])
        producers = set()
        constants = set()
        for value in values:
            producers.update(value.producers)
            if value.constant is not None:
                constants.add(value.constant)
        if len(producers) > max_producers:
            producers = set()
            constant = None
        else:
            constant = next(iter(constants)) if len(constants) == 1 and len(values) == 2 else (
                next(iter(constants)) if len(values) == 1 and constants else None
            )
        item = AbstractValue(frozenset(producers), constant)
        merged.append(item)
        if index >= len(old) or old[index] != item:
            changed = True
    return merged, changed


def build_evm_graph(
    opcode_text: str,
    tokenizer=None,
    include_storage_edges: bool = False,
    max_producers: int = 4,
    max_worklist_steps: int = 20000,
    max_instruction_visits: int = 250000,
):
    token_spans = None
    token_count = None
    if tokenizer is not None:
        tokens, token_spans = build_token_spans(opcode_text, tokenizer)
        token_count = len(tokens)
    instructions = _raw_instructions(opcode_text, token_spans)
    if not instructions:
        return {
            "nodes": [],
            "edges": [],
            "edge_types": EDGE_TYPES.copy(),
            "report": {
                "instruction_count": 0,
                "token_count": 0,
                "basic_block_count": 0,
                "edge_count": 0,
                "direct_jump_count": 0,
                "unresolved_jump_count": 0,
                "direct_jump_resolution_rate": 0.0,
                "stack_edge_count": 0,
                "storage_edge_count": 0,
                "token_coverage": 0.0,
                "stack_worklist_steps": 0,
                "stack_instruction_visits": 0,
                "stack_analysis_capped": False,
            },
        }

    boundaries = {0}
    for instruction in instructions:
        if instruction.opcode == "JUMPDEST":
            boundaries.add(instruction.index)
        if instruction.opcode in TERMINATORS or instruction.opcode == "JUMPI":
            if instruction.index + 1 < len(instructions):
                boundaries.add(instruction.index + 1)
    starts = sorted(boundaries)
    start_to_block = {start: idx for idx, start in enumerate(starts)}
    pc_to_block = {}
    instruction_to_block = {}
    nodes = []
    for block_id, start in enumerate(starts):
        end = starts[block_id + 1] if block_id + 1 < len(starts) else len(instructions)
        block_instructions = instructions[start:end]
        for item in block_instructions:
            instruction_to_block[item.index] = block_id
            if item.opcode == "JUMPDEST" or item.index == 0:
                pc_to_block[item.pc] = block_id
        nodes.append({
            "id": block_id,
            "instruction_start": start,
            "instruction_end": end,
            "pc_start": block_instructions[0].pc,
            "pc_end": block_instructions[-1].pc,
            "token_start": min(item.token_start for item in block_instructions),
            "token_end": max(item.token_end for item in block_instructions),
        })

    edge_set: Set[Tuple[int, int, int]] = set()
    unresolved_jumps = 0
    direct_jumps = 0
    for block_id, node in enumerate(nodes):
        last = instructions[node["instruction_end"] - 1]
        if last.opcode == "JUMPI":
            if block_id + 1 < len(nodes):
                edge_set.add((block_id, block_id + 1, EDGE_TYPES["conditional_false"]))
        elif last.opcode not in TERMINATORS and block_id + 1 < len(nodes):
            edge_set.add((block_id, block_id + 1, EDGE_TYPES["fallthrough"]))

    entry_stacks: Dict[int, Optional[List[AbstractValue]]] = {0: []}
    queue = deque([0])
    stack_edges: Set[Tuple[int, int, int]] = set()
    storage_events: Dict[int, List[Tuple[int, bool]]] = {}
    worklist_steps = 0
    instruction_visits = 0
    stack_analysis_capped = False
    while queue and not stack_analysis_capped:
        if worklist_steps >= max_worklist_steps:
            stack_analysis_capped = True
            break
        block_id = queue.popleft()
        worklist_steps += 1
        stack = list(entry_stacks.get(block_id) or [])
        node = nodes[block_id]
        block_successors: List[Tuple[int, int]] = []
        for instruction in instructions[node["instruction_start"]:node["instruction_end"]]:
            if instruction_visits >= max_instruction_visits:
                stack_analysis_capped = True
                break
            instruction_visits += 1
            pops, pushes = _stack_effect(instruction.opcode)
            consumed = stack[-pops:] if pops else []
            for value in consumed:
                for producer in value.producers:
                    stack_edges.add((producer, instruction.index, EDGE_TYPES["stack_def_use"]))
            if instruction.opcode == "SLOAD" and stack and stack[-1].constant is not None:
                storage_events.setdefault(stack[-1].constant, []).append((instruction.index, False))
            if instruction.opcode == "SSTORE" and len(stack) >= 2 and stack[-2].constant is not None:
                storage_events.setdefault(stack[-2].constant, []).append((instruction.index, True))
            if pops:
                stack = stack[:-min(pops, len(stack))]
            if instruction.opcode in DUP_OPCODES:
                depth = int(instruction.opcode[3:])
                value = stack[-depth] if len(stack) >= depth else AbstractValue(frozenset())
                stack.append(value)
            elif instruction.opcode in SWAP_OPCODES:
                depth = int(instruction.opcode[4:])
                if len(stack) > depth:
                    stack[-1], stack[-1 - depth] = stack[-1 - depth], stack[-1]
            elif instruction.opcode in PUSH_OPCODES:
                stack.append(AbstractValue(frozenset({instruction.index}), instruction.operand))
            else:
                constant = None
                if pushes == 1:
                    stack.append(AbstractValue(frozenset({instruction.index}), constant))
            if len(stack) > 1024:
                stack = stack[-1024:]

        if stack_analysis_capped:
            break

        last = instructions[node["instruction_end"] - 1]
        if last.opcode in {"JUMP", "JUMPI"}:
            target_value = consumed[-1] if consumed else AbstractValue(frozenset())
            target_block = pc_to_block.get(target_value.constant) if target_value.constant is not None else None
            if target_block is None:
                unresolved_jumps += 1
            else:
                direct_jumps += 1
                edge_kind = "conditional_true" if last.opcode == "JUMPI" else "direct_jump"
                edge_set.add((block_id, target_block, EDGE_TYPES[edge_kind]))
                block_successors.append((target_block, EDGE_TYPES[edge_kind]))
        if last.opcode == "JUMPI" and block_id + 1 < len(nodes):
            block_successors.append((block_id + 1, EDGE_TYPES["conditional_false"]))
        elif last.opcode not in TERMINATORS and block_id + 1 < len(nodes):
            block_successors.append((block_id + 1, EDGE_TYPES["fallthrough"]))
        for successor, _ in block_successors:
            merged, changed = _merge_stack(entry_stacks.get(successor), stack, max_producers)
            entry_stacks[successor] = merged
            if changed:
                queue.append(successor)

    if include_storage_edges:
        for events in storage_events.values():
            # Preserve the conservative access order without creating every
            # pair of accesses to the same slot.
            for (left, _), (right, _) in zip(events, events[1:]):
                stack_edges.add((left, right, EDGE_TYPES["storage_dependency"]))

    for src_instruction, dst_instruction, edge_type in stack_edges:
        src_block = instruction_to_block.get(src_instruction)
        dst_block = instruction_to_block.get(dst_instruction)
        if src_block is not None and dst_block is not None and src_block != dst_block:
            edge_set.add((src_block, dst_block, edge_type))

    edges = [{"src": src, "dst": dst, "type": edge_type} for src, dst, edge_type in sorted(edge_set)]
    node_token_ranges = [(node["token_start"], node["token_end"]) for node in nodes]
    report = {
        "instruction_count": len(instructions),
        "token_count": token_count,
        "basic_block_count": len(nodes),
        "edge_count": len(edges),
        "direct_jump_count": direct_jumps,
        "unresolved_jump_count": unresolved_jumps,
        "direct_jump_resolution_rate": direct_jumps / max(1, direct_jumps + unresolved_jumps),
        "stack_edge_count": sum(edge["type"] == EDGE_TYPES["stack_def_use"] for edge in edges),
        "storage_edge_count": sum(edge["type"] == EDGE_TYPES["storage_dependency"] for edge in edges),
        "token_coverage": sum(end > start for start, end in node_token_ranges) / max(1, len(nodes)),
        "stack_worklist_steps": worklist_steps,
        "stack_instruction_visits": instruction_visits,
        "stack_analysis_capped": stack_analysis_capped,
    }
    return {"nodes": nodes, "edges": edges, "edge_types": EDGE_TYPES.copy(), "report": report}
