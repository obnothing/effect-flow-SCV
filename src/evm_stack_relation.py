"""Conservative EVM stack-value relations for the isolated stack route.

This module keeps the information that the old execution-aware route reduced
to chunk mean/max statistics: which instruction value is consumed by which
instruction, in which operand slot, and whether DUP/SWAP preserved or
reordered value identity.  Relations are instruction based and are later
projected to the first token of each instruction span.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from evm_control_stack_graph import (
    DUP_OPCODES,
    PUSH_OPCODES,
    SWAP_OPCODES,
    _raw_instructions,
    _stack_effect,
    build_token_spans,
)


RELATION_NAMES = (
    "none",
    "direct_forward",
    "direct_reverse",
    "alias",
    "reorder",
    "control_operand",
    "storage_operand",
    "external_argument",
)
RELATION_INDEX = {name: idx for idx, name in enumerate(RELATION_NAMES)}

DISTANCE_BUCKETS = (0, 1, 2, 3, 7, 15, 31, 63)


@dataclass(frozen=True)
class Value:
    producers: Tuple[int, ...] = ()
    unknown: bool = False


def _unknown() -> Value:
    return Value(unknown=True)


def distance_bucket(distance: int) -> int:
    distance = max(0, int(distance))
    for index, upper in enumerate(DISTANCE_BUCKETS):
        if distance <= upper:
            return index
    return len(DISTANCE_BUCKETS)


def _merge_values(left: Value, right: Value, max_producers: int) -> Value:
    producers = tuple(sorted(set(left.producers).union(right.producers)))
    if left.unknown or right.unknown or len(producers) > max_producers:
        return _unknown()
    return Value(producers=producers)


def _merge_stack(old: Optional[List[Value]], new: List[Value], max_producers: int):
    if old is None:
        return list(new), True
    if len(old) != len(new):
        size = min(len(old), len(new))
        merged = [_unknown() for _ in range(size)]
        return merged, merged != old
    merged = [_merge_values(a, b, max_producers) for a, b in zip(old, new)]
    return merged, merged != old


def _block_layout(instructions):
    terminators = {"STOP", "RETURN", "REVERT", "SELFDESTRUCT", "INVALID", "JUMP"}
    boundaries = {0}
    for item in instructions:
        if item.opcode == "JUMPDEST":
            boundaries.add(item.index)
        if item.opcode in terminators or item.opcode == "JUMPI":
            if item.index + 1 < len(instructions):
                boundaries.add(item.index + 1)
    starts = sorted(boundaries)
    instruction_to_block = {}
    blocks = []
    pc_to_block = {}
    for block_id, start in enumerate(starts):
        end = starts[block_id + 1] if block_id + 1 < len(starts) else len(instructions)
        block = instructions[start:end]
        for item in block:
            instruction_to_block[item.index] = block_id
            if item.index == start or item.opcode == "JUMPDEST":
                pc_to_block[item.pc] = block_id
        blocks.append((start, end))
    return blocks, instruction_to_block, pc_to_block


def _relation_family(opcode: str, slot: int) -> str:
    if opcode in {"JUMP", "JUMPI"}:
        return "control_operand"
    if opcode in {"SLOAD", "SSTORE"}:
        return "storage_operand"
    if opcode in {"CALL", "CALLCODE", "DELEGATECALL", "STATICCALL", "CREATE", "CREATE2"}:
        return "external_argument"
    return "direct_forward"


def _append_relation(relations, producer, consumer, relation, slot, distance, confidence):
    relations.append(
        {
            "producer": int(producer),
            "consumer": int(consumer),
            "relation": relation,
            "slot": int(min(max(slot, 0), 15)),
            "distance": int(distance),
            "confidence": float(confidence),
        }
    )


def analyze_stack_relations(
    opcode_text: str,
    tokenizer=None,
    max_producers: int = 4,
    max_stack_depth: int = 1024,
    max_worklist_steps: int = 20000,
    max_instruction_visits: int = 250000,
) -> Dict:
    token_spans = None
    tokens = []
    if tokenizer is not None:
        tokens, token_spans = build_token_spans(opcode_text, tokenizer)
    instructions = _raw_instructions(opcode_text, token_spans)
    blocks, instruction_to_block, pc_to_block = _block_layout(instructions)
    token_count = len(tokens)
    token_state = np.zeros((token_count, 5), dtype=np.uint8)
    relations = []
    entry_stacks: Dict[int, Optional[List[Value]]] = {0: []} if blocks else {}
    queue = deque([0]) if blocks else deque()
    pending = {0} if blocks else set()
    steps = 0
    visits = 0
    capped = False
    unknown_consumptions = 0
    alias_count = 0
    reorder_count = 0

    while queue and not capped:
        if steps >= max_worklist_steps:
            capped = True
            break
        block_id = queue.popleft()
        pending.discard(block_id)
        steps += 1
        stack = list(entry_stacks.get(block_id) or [])
        start, end = blocks[block_id]
        last_consumed = []
        for instruction in instructions[start:end]:
            if visits >= max_instruction_visits:
                capped = True
                break
            visits += 1
            pops, pushes = _stack_effect(instruction.opcode)
            before = len(stack)
            consumed = []
            for slot in range(pops):
                if stack:
                    consumed.append(stack[-1])
                    stack.pop()
                else:
                    consumed.append(_unknown())
            last_consumed = consumed
            if any(value.unknown for value in consumed):
                unknown_consumptions += 1

            # Token-level local state. The same instruction state is assigned
            # to PUSH operands because the tokenizer treats PUSH + immediate
            # as one semantic token span.
            if instruction.token_end > instruction.token_start:
                state = token_state[instruction.token_start:instruction.token_end]
                state[:, 0] = min(before, 31)
                state[:, 1] = min(max(0, before - pops + pushes), 31)
                state[:, 2] = min(sum(1 << slot for slot in range(min(pops, 8))), 255)
                state[:, 3] = min(sum(len(value.producers) for value in consumed), 15)
                state[:, 4] = 1 if any(value.unknown for value in consumed) else 0

            for slot, value in enumerate(consumed):
                family = _relation_family(instruction.opcode, slot)
                for producer in value.producers:
                    _append_relation(
                        relations,
                        producer,
                        instruction.index,
                        family,
                        slot,
                        instruction.index - producer,
                        0.25 if value.unknown else 1.0,
                    )

            if instruction.opcode in DUP_OPCODES:
                depth = int(instruction.opcode[3:])
                value = stack[-depth] if len(stack) >= depth else _unknown()
                for producer in value.producers:
                    _append_relation(relations, producer, instruction.index, "alias", 0, instruction.index - producer, 0.25 if value.unknown else 1.0)
                alias_count += 1
                stack.append(value)
            elif instruction.opcode in SWAP_OPCODES:
                depth = int(instruction.opcode[4:])
                if len(stack) > depth:
                    top, other = stack[-1], stack[-1 - depth]
                    for value in (top, other):
                        for producer in value.producers:
                            _append_relation(relations, producer, instruction.index, "reorder", 0, instruction.index - producer, 0.25 if value.unknown else 1.0)
                    stack[-1], stack[-1 - depth] = other, top
                reorder_count += 1
            elif instruction.opcode in PUSH_OPCODES:
                stack.append(Value((instruction.index,), False))
            else:
                for _ in range(pushes):
                    stack.append(Value((instruction.index,), False))
            if len(stack) > max_stack_depth:
                stack = stack[-max_stack_depth:]

        if capped:
            break

        last = instructions[end - 1]
        successor_states = []
        if last.opcode in {"JUMP", "JUMPI"}:
            # At JUMPI the top value is the condition and the next value is
            # the destination. JUMP consumes the top value as destination.
            target_slot = 1 if last.opcode == "JUMPI" else 0
            if len(last_consumed) > target_slot:
                target = last_consumed[target_slot]
            else:
                target = _unknown()
            target_block = None
            if target.producers:
                for producer in target.producers:
                    candidate = instructions[producer].operand
                    if candidate is not None:
                        target_block = pc_to_block.get(candidate)
                        if target_block is not None:
                            break
            if target_block is not None:
                successor_states.append((target_block, stack, "direct"))
            if last.opcode == "JUMPI" and block_id + 1 < len(blocks):
                successor_states.append((block_id + 1, stack, "fallthrough"))
        elif last.opcode not in {"STOP", "RETURN", "REVERT", "SELFDESTRUCT", "INVALID"} and block_id + 1 < len(blocks):
            successor_states.append((block_id + 1, stack, "fallthrough"))

        for successor, next_stack, _ in successor_states:
            merged, changed = _merge_stack(entry_stacks.get(successor), next_stack, max_producers)
            entry_stacks[successor] = merged
            if changed and successor not in pending:
                queue.append(successor)
                pending.add(successor)

    return {
        "tokens": tokens,
        "instructions": instructions,
        "relations": relations,
        "token_state": token_state,
        "report": {
            "instruction_count": len(instructions),
            "token_count": token_count,
            "basic_block_count": len(blocks),
            "relation_count": len(relations),
            "direct_relation_count": sum(r["relation"] == "direct_forward" for r in relations),
            "alias_relation_count": alias_count,
            "reorder_relation_count": reorder_count,
            "unknown_consumption_count": unknown_consumptions,
            "stack_worklist_steps": steps,
            "stack_instruction_visits": visits,
            "stack_analysis_capped": capped,
            "relation_names": list(RELATION_NAMES),
        },
    }


def _chunk_starts(token_count: int, content_size: int, stride: int, max_chunks: int):
    starts = list(range(0, token_count, stride)) if token_count else [0]
    return starts[:max_chunks]


def pack_contract_relations(
    opcode_text: str,
    tokenizer,
    max_len: int = 512,
    chunk_stride: int = 256,
    max_chunks: int = 64,
):
    content_size = max_len - 2
    analysis = analyze_stack_relations(opcode_text, tokenizer=tokenizer)
    tokens = analysis["tokens"]
    starts = _chunk_starts(len(tokens), content_size, chunk_stride, max_chunks)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    pad_id = tokenizer.pad_token_id
    cls_id = tokenizer.vocab[tokenizer.cls_token]
    sep_id = tokenizer.vocab[tokenizer.sep_token]
    chunks = []
    for start in starts:
        content = ids[start:start + content_size]
        input_ids = [cls_id] + content + [sep_id]
        attention = [1] * len(input_ids)
        if len(input_ids) < max_len:
            input_ids += [pad_id] * (max_len - len(input_ids))
            attention += [0] * (max_len - len(attention))
        chunks.append({"start": start, "input_ids": input_ids, "attention_mask": attention})

    packed_edges = []
    boundary = []
    relation_names = {name: idx for idx, name in enumerate(RELATION_NAMES)}
    for chunk in chunks:
        start = chunk["start"]
        end = start + content_size
        local_state = np.zeros((max_len, 5), dtype=np.uint8)
        if analysis["token_state"].shape[0]:
            local_state[1:1 + min(content_size, len(tokens) - start)] = analysis["token_state"][start:min(end, len(tokens))]
        local_edges = []
        incoming = outgoing = 0
        for relation in analysis["relations"]:
            source = analysis["instructions"][relation["producer"]]
            target = analysis["instructions"][relation["consumer"]]
            source_token = source.token_start
            target_token = target.token_start
            source_inside = start <= source_token < end
            target_inside = start <= target_token < end
            if source_inside and target_inside:
                relation_type = relation_names.get(relation["relation"], 0)
                distance = distance_bucket(relation["distance"])
                local_edges.append((source_token - start + 1, target_token - start + 1, relation_type, relation["slot"], distance, relation["confidence"]))
                if relation_type in {
                    RELATION_INDEX["direct_forward"],
                    RELATION_INDEX["control_operand"],
                    RELATION_INDEX["storage_operand"],
                    RELATION_INDEX["external_argument"],
                }:
                    local_edges.append((target_token - start + 1, source_token - start + 1, RELATION_INDEX["direct_reverse"], relation["slot"], distance, relation["confidence"]))
            elif target_inside:
                incoming += 1
            elif source_inside:
                outgoing += 1
        boundary.append([min(incoming, 15), min(outgoing, 15), min(analysis["report"]["unknown_consumption_count"], 15), min(max((e[4] for e in local_edges), default=0), 8)])
        chunk["stack_state"] = local_state
        chunk["edges"] = local_edges
    return chunks, boundary, analysis
