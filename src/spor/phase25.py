"""Stack-effect-aware SPOR with a statically resolved partial CFG."""

from collections import Counter, deque
from dataclasses import dataclass

from .stack_effects import stack_effect
from .stack_provenance import AbstractValue, SOURCE_BITS


UNTRUSTED = SOURCE_BITS["UNKNOWN"]
TARGETS = {"JUMPI", "CALL", "SLOAD", "SSTORE"}
TERMINAL = {"STOP", "RETURN", "REVERT", "SELFDESTRUCT", "INVALID", "UNKNOWN_0xFE"}


def unknown(reason):
    return AbstractValue(UNTRUSTED, "UNKNOWN", None, reason)


def combine(values, op, category):
    bits = SOURCE_BITS[category]
    reasons = []
    for value in values:
        bits |= value.provenance_bits
        if value.analysis_status != "known":
            reasons.append(value.analysis_status)
    return AbstractValue(bits, op, None, reasons[0] if reasons else "known")


def pop_many(stack, count, entry_reason, reliable_entry):
    values = []
    for _ in range(count):
        if stack:
            values.append(stack.pop())
        else:
            values.append(unknown("abstract_stack_underflow" if reliable_entry else entry_reason))
    return values


def transfer(instructions, block, incoming, entry_reason, reliable_entry):
    stack = list(incoming) if incoming is not None else []
    poisoned = False
    for index in block.instruction_indices:
        ins = instructions[index]
        op = ins.opcode
        ins.operand_sources = {}
        ins.provenance_bits = 0
        ins.operation_role = "OTHER"
        ins.analysis_status = "known"
        ins.failure_reason = None
        effect = stack_effect(op)
        if ins.parse_status != "ok" and op != "UNKNOWN_0xFE":
            ins.failure_reason = "parser_failure" if ins.parse_status != "unknown_opcode_annotation" else "unsupported_opcode_stack_effect"
            poisoned = True
            stack = None
        elif effect is None:
            ins.failure_reason = "unsupported_opcode_stack_effect"
            poisoned = True
            stack = None
        elif poisoned:
            ins.failure_reason = "previous_analysis_failure"
        else:
            if op.startswith("PUSH"):
                value = 0 if op == "PUSH0" else int(ins.immediate_operand, 16)
                stack.append(AbstractValue(SOURCE_BITS["CONSTANT"], op, value, "known"))
                ins.operation_role = "CONSTANT"
            elif op.startswith("DUP"):
                distance = int(op[3:])
                value = stack[-distance] if len(stack) >= distance else unknown("abstract_stack_underflow" if reliable_entry else entry_reason)
                stack.append(value)
                ins.operand_sources["source"] = value
                ins.operation_role = "STACK_DUP"
            elif op.startswith("SWAP"):
                distance = int(op[4:])
                while len(stack) <= distance:
                    stack.insert(0, unknown("abstract_stack_underflow" if reliable_entry else entry_reason))
                stack[-1], stack[-distance - 1] = stack[-distance - 1], stack[-1]
                ins.operand_sources["top"] = stack[-distance - 1]
                ins.operand_sources["other"] = stack[-1]
                ins.operation_role = "STACK_SWAP"
            else:
                values = pop_many(stack, effect.pop_count, entry_reason, reliable_entry)
                if op in {"JUMP", "JUMPI"}:
                    ins.operand_sources["destination"] = values[0]
                    if op == "JUMPI":
                        ins.operand_sources["condition"] = values[1]
                    ins.operation_role = "CONTROL_FLOW"
                elif op in {"CALL", "CALLCODE", "DELEGATECALL", "STATICCALL"}:
                    names = ["gas", "target"] + (["value"] if op in {"CALL", "CALLCODE"} else []) + ["input_offset", "input_size", "output_offset", "output_size"]
                    ins.operand_sources.update(zip(names, values))
                    ins.operation_role = "CALL"
                elif op in {"SLOAD", "MLOAD", "TLOAD"}:
                    ins.operand_sources["key" if op in {"SLOAD", "TLOAD"} else "offset"] = values[0]
                    ins.operation_role = "STORAGE" if op in {"SLOAD", "TLOAD"} else "MEMORY"
                elif op in {"SSTORE", "MSTORE", "MSTORE8", "TSTORE"}:
                    ins.operand_sources.update({"key" if op in {"SSTORE", "TSTORE"} else "offset": values[0], "value": values[1]})
                    ins.operation_role = "STORAGE" if op in {"SSTORE", "TSTORE"} else "MEMORY"
                else:
                    if values:
                        ins.operand_sources["operands"] = values
                category = None
                if op in {"ADD", "SUB", "MUL", "DIV", "SDIV", "MOD", "SMOD", "ADDMOD", "MULMOD", "AND", "OR", "XOR", "NOT", "SHL", "SHR", "SAR", "EXP", "SIGNEXTEND", "BYTE"}:
                    category = "ARITHMETIC"
                    ins.operation_role = "ARITHMETIC"
                elif op in {"EQ", "LT", "GT", "SLT", "SGT", "ISZERO"}:
                    category = "COMPARISON"
                    ins.operation_role = "COMPARISON"
                elif op in {"CALLER", "CALLVALUE", "TIMESTAMP"}:
                    category = op
                    ins.operation_role = "BLOCK_ENV" if op == "TIMESTAMP" else op
                elif op in {"CALLDATALOAD", "CALLDATASIZE"}:
                    category = "CALLDATA"
                    ins.operation_role = "CALLDATA"
                elif op in {"NUMBER", "COINBASE", "DIFFICULTY", "PREVRANDAO", "GASLIMIT", "CHAINID", "BASEFEE", "BLOCKHASH", "GAS"}:
                    category = "BLOCK_ENV"
                    ins.operation_role = "BLOCK_ENV"
                elif op in {"SLOAD", "TLOAD"}:
                    category = "STORAGE"
                elif op == "MLOAD":
                    category = "MEMORY"
                elif op in {"CALL", "CALLCODE", "DELEGATECALL", "STATICCALL"}:
                    category = "CALL_RESULT"
                if effect.push_count:
                    if category is None:
                        produced = unknown("known_stack_effect_but_unknown_provenance")
                        ins.failure_reason = "known_stack_effect_but_unknown_provenance"
                    elif op in {"CALLER", "CALLVALUE", "TIMESTAMP", "CALLDATASIZE", "NUMBER", "COINBASE", "DIFFICULTY", "PREVRANDAO", "GASLIMIT", "CHAINID", "BASEFEE", "GAS"}:
                        produced = AbstractValue(SOURCE_BITS[category], op, None, "known")
                    else:
                        produced = combine(values, op, category)
                    stack.append(produced)
        if stack is None:
            ins.analysis_status = "analysis_failure"
            ins.provenance_bits = UNTRUSTED
            continue
        flattened = [v for item in ins.operand_sources.values() for v in (item if isinstance(item, list) else [item])]
        produced = stack[-1] if effect is not None and effect.push_count and stack else None
        values = flattened + ([produced] if produced else [])
        for value in values:
            ins.provenance_bits |= value.provenance_bits
        if ins.failure_reason is None:
            for value in values:
                if value.analysis_status != "known":
                    ins.failure_reason = value.analysis_status
                    break
        if ins.failure_reason:
            ins.analysis_status = "unknown_entry" if ins.failure_reason in {"basic_block_entry_unknown", "dynamic_jump_or_unresolved_cfg"} else "analysis_failure"
            ins.provenance_bits |= UNTRUSTED
    return tuple(stack) if stack is not None else None


def join_stacks(states):
    if not states:
        return None, "basic_block_entry_unknown"
    if any(state is None for state in states):
        return None, "previous_analysis_failure"
    sizes = {len(state) for state in states}
    if len(sizes) != 1:
        return None, "predecessor_stack_height_conflict"
    joined = []
    for values in zip(*states):
        bits = 0
        for value in values:
            bits |= value.provenance_bits
        same = all(value.known_constant is not None and value.known_constant == values[0].known_constant for value in values)
        constant = values[0].known_constant if same else None
        status = "known" if all(value.analysis_status == "known" for value in values) else next(value.analysis_status for value in values if value.analysis_status != "known")
        joined.append(AbstractValue(bits, values[0].generating_operation if same else "JOIN", constant, status))
    return tuple(joined), None


def analyze_partial_cfg(instructions, blocks, max_iterations=32):
    """Propagate along explicit fallthrough and statically proven jump edges."""
    pc_to_block = {}
    for block in blocks:
        first = instructions[block.entry_instruction_index]
        if first.opcode == "JUMPDEST" and first.derived_pc is not None:
            pc_to_block.setdefault(first.derived_pc, []).append(block.block_id)
    successors = {block.block_id: set() for block in blocks}
    edge_types = Counter()
    unresolved_blocks = set()
    # A local pass recovers constant jump targets without assuming predecessor state.
    for block in blocks:
        transfer(instructions, block, None, "basic_block_entry_unknown", False)
        last = instructions[block.instruction_indices[-1]]
        if block.block_id + 1 < len(blocks) and last.opcode not in TERMINAL | {"JUMP"}:
            successors[block.block_id].add(block.block_id + 1)
            edge_types["jumpi_fallthrough" if last.opcode == "JUMPI" else "sequential_fallthrough"] += 1
        if last.opcode in {"JUMP", "JUMPI"}:
            dest = last.operand_sources.get("destination")
            targets = pc_to_block.get(dest.known_constant, []) if dest and dest.analysis_status == "known" else []
            if len(targets) == 1:
                successors[block.block_id].add(targets[0])
                edge_types["static_jump" if last.opcode == "JUMP" else "static_jumpi"] += 1
            else:
                unresolved_blocks.add(block.block_id)
    predecessors = {block.block_id: set() for block in blocks}
    for source, targets in successors.items():
        for target in targets:
            predecessors[target].add(source)
    outputs = {block.block_id: None for block in blocks}
    queued = deque(range(len(blocks)))
    visits = Counter()
    conflicts = set()
    while queued:
        block_id = queued.popleft()
        visits[block_id] += 1
        if visits[block_id] > max_iterations:
            for index in blocks[block_id].instruction_indices:
                ins = instructions[index]
                ins.failure_reason = "other"
                ins.analysis_status = "analysis_failure"
                ins.provenance_bits |= UNTRUSTED
            outputs[block_id] = None
            continue
        pred = predecessors[block_id]
        available = [outputs[p] for p in pred if visits[p]]
        if block_id == 0:
            incoming, reason = (), None
        elif pred and len(available) == len(pred):
            incoming, reason = join_stacks(available)
        else:
            incoming, reason = None, "dynamic_jump_or_unresolved_cfg" if not pred else "basic_block_entry_unknown"
        if reason == "predecessor_stack_height_conflict":
            conflicts.add(block_id)
        new_output = transfer(instructions, blocks[block_id], incoming, reason or "basic_block_entry_unknown", incoming is not None)
        last = instructions[blocks[block_id].instruction_indices[-1]]
        if last.opcode in {"JUMP", "JUMPI"}:
            dest = last.operand_sources.get("destination")
            targets = pc_to_block.get(dest.known_constant, []) if dest and dest.analysis_status == "known" else []
            if len(targets) == 1:
                unresolved_blocks.discard(block_id)
                target = targets[0]
                if target not in successors[block_id]:
                    successors[block_id].add(target)
                    predecessors[target].add(block_id)
                    edge_types["static_jump" if last.opcode == "JUMP" else "static_jumpi"] += 1
                    if target not in queued:
                        queued.append(target)
            else:
                unresolved_blocks.add(block_id)
        if new_output != outputs[block_id]:
            outputs[block_id] = new_output
            for target in successors[block_id]:
                if target not in queued:
                    queued.append(target)
    return {"edges": dict(edge_types), "unresolved_jumps": len(unresolved_blocks), "stack_height_conflicts": len(conflicts),
            "fixpoint_visits": sum(visits.values()), "nonconverged_blocks": sum(count > max_iterations for count in visits.values()),
            "blocks": len(blocks)}
