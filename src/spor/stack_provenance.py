"""Basic-block-local conservative abstract stack provenance."""

from dataclasses import dataclass, field


SOURCE_CATEGORIES = ["CONSTANT", "CALLER", "CALLVALUE", "CALLDATA", "TIMESTAMP", "BLOCK_ENV",
                     "STORAGE", "MEMORY", "CALL_RESULT", "ARITHMETIC", "COMPARISON", "OTHER", "UNKNOWN"]
SOURCE_BITS = {name: 1 << index for index, name in enumerate(SOURCE_CATEGORIES)}


@dataclass
class AbstractValue:
    provenance_bits: int = 0
    generating_operation: str = "OTHER"
    known_constant: int | None = None
    analysis_status: str = "known"

    @classmethod
    def unknown_entry(cls):
        return cls(SOURCE_BITS["UNKNOWN"], "UNKNOWN", None, "unknown_entry")

    @classmethod
    def failure(cls):
        return cls(SOURCE_BITS["UNKNOWN"], "UNKNOWN", None, "analysis_failure")

    @classmethod
    def source(cls, name, operation=None):
        return cls(SOURCE_BITS[name], operation or name, None, "known")


def _merge(values, operation, category=None):
    bits = 0
    status = "known"
    for value in values:
        bits |= value.provenance_bits
        if value.analysis_status == "analysis_failure":
            status = "analysis_failure"
        elif value.analysis_status == "unknown_entry" and status == "known":
            status = "unknown_entry"
    if category:
        bits |= SOURCE_BITS[category]
    return AbstractValue(bits, operation, None, status)


def _pop(stack):
    return stack.pop() if stack else AbstractValue.unknown_entry()


def _pop_many(stack, count):
    return [_pop(stack) for _ in range(count)]


def _push_binary(stack, operation, category):
    values = _pop_many(stack, 2)
    stack.append(_merge(values, operation, category))
    return values


def _unsupported(stack, instruction):
    # Unknown stack effect makes the current local stack unreliable. Keep one
    # explicit failure value instead of silently treating the instruction as a no-op.
    stack.clear(); stack.append(AbstractValue.failure())
    instruction.analysis_status = "analysis_failure_unsupported_stack_effect"


def _status_from_values(instruction, values):
    if any(value.analysis_status == "analysis_failure" for value in values):
        instruction.analysis_status = "analysis_failure"
    elif any(value.analysis_status == "unknown_entry" for value in values):
        instruction.analysis_status = "unknown_entry"


def analyze_basic_blocks(instructions, blocks):
    records = []
    for block in blocks:
        stack = []
        poisoned = False
        for index in block.instruction_indices:
            instruction = instructions[index]
            op = instruction.opcode
            instruction.analysis_status = "known"
            instruction.provenance_bits = 0
            instruction.operand_sources = {}
            if instruction.parse_status != "ok":
                _unsupported(stack, instruction)
                instruction.analysis_status = "analysis_failure_parse_status"
                instruction.operation_role = "OTHER"
                poisoned = True
            elif op == "PUSH0" or (op.startswith("PUSH") and op[4:].isdigit() and 1 <= int(op[4:]) <= 32):
                value = AbstractValue(SOURCE_BITS["CONSTANT"], "CONSTANT", 0 if op == "PUSH0" else int(instruction.immediate_operand, 16), "known")
                stack.append(value); instruction.operation_role = "CONSTANT"
            elif op == "POP":
                instruction.operand_sources["value"] = _pop(stack); instruction.operation_role = "STACK_POP"
                if instruction.operand_sources["value"].analysis_status != "known":
                    instruction.analysis_status = instruction.operand_sources["value"].analysis_status
            elif op.startswith("DUP") and op[3:].isdigit() and 1 <= int(op[3:]) <= 16:
                distance = int(op[3:]); value = stack[-distance] if len(stack) >= distance else AbstractValue.unknown_entry()
                stack.append(value); instruction.operand_sources["source"] = value; instruction.operation_role = "STACK_DUP"
            elif op.startswith("SWAP") and op[4:].isdigit() and 1 <= int(op[4:]) <= 16:
                distance = int(op[4:])
                while len(stack) <= distance:
                    stack.insert(0, AbstractValue.unknown_entry())
                stack[-1], stack[-1 - distance] = stack[-1 - distance], stack[-1]
                instruction.operand_sources["top"] = stack[-1 - distance]
                instruction.operand_sources["other"] = stack[-1]
                _status_from_values(instruction, list(instruction.operand_sources.values()))
                instruction.operation_role = "STACK_SWAP"
            elif op in {"ADD", "SUB", "MUL", "DIV", "MOD", "SDIV", "SMOD", "AND", "OR", "XOR", "SHL", "SHR", "SAR", "EXP", "SIGNEXTEND", "BYTE"}:
                instruction.operand_sources["operands"] = _push_binary(stack, op, "ARITHMETIC"); instruction.operation_role = "ARITHMETIC"
            elif op in {"ADDMOD", "MULMOD"}:
                values = _pop_many(stack, 3)
                stack.append(_merge(values, op, "ARITHMETIC"))
                instruction.operand_sources["operands"] = values; instruction.operation_role = "ARITHMETIC"
            elif op in {"EQ", "LT", "GT", "SLT", "SGT", "ISZERO"}:
                values = _pop_many(stack, 1 if op == "ISZERO" else 2)
                stack.append(_merge(values, op, "COMPARISON")); instruction.operand_sources["operands"] = values; instruction.operation_role = "COMPARISON"
            elif op == "NOT":
                values = _pop_many(stack, 1)
                stack.append(_merge(values, op, "ARITHMETIC")); instruction.operand_sources["operands"] = values; instruction.operation_role = "ARITHMETIC"
            elif op == "CALLER":
                stack.append(AbstractValue.source("CALLER")); instruction.operation_role = "CALLER"
            elif op == "CALLVALUE":
                stack.append(AbstractValue.source("CALLVALUE")); instruction.operation_role = "CALLVALUE"
            elif op == "CALLDATALOAD":
                instruction.operand_sources["offset"] = _pop(stack); stack.append(_merge([instruction.operand_sources["offset"]], op, "CALLDATA")); instruction.operation_role = "CALLDATA"
            elif op == "CALLDATASIZE":
                stack.append(AbstractValue.source("CALLDATA")); instruction.operation_role = "CALLDATA"
            elif op in {"TIMESTAMP"}:
                stack.append(AbstractValue.source("TIMESTAMP")); instruction.operation_role = "BLOCK_ENV"
            elif op in {"NUMBER", "COINBASE", "DIFFICULTY", "PREVRANDAO", "GASLIMIT", "CHAINID", "BASEFEE", "BLOCKHASH"}:
                stack.append(AbstractValue.source("BLOCK_ENV")); instruction.operation_role = "BLOCK_ENV"
            elif op == "MLOAD":
                instruction.operand_sources["offset"] = _pop(stack); stack.append(AbstractValue.source("MEMORY")); instruction.operation_role = "MEMORY"
            elif op == "MSTORE":
                instruction.operand_sources["offset"] = _pop(stack); instruction.operand_sources["value"] = _pop(stack); instruction.operation_role = "MEMORY"
            elif op == "SLOAD":
                instruction.operand_sources["key"] = _pop(stack); stack.append(_merge([instruction.operand_sources["key"]], op, "STORAGE")); instruction.operation_role = "STORAGE"
            elif op == "SSTORE":
                instruction.operand_sources["key"] = _pop(stack); instruction.operand_sources["value"] = _pop(stack); instruction.operation_role = "STORAGE"
            elif op == "MSTORE8":
                instruction.operand_sources["offset"] = _pop(stack); instruction.operand_sources["value"] = _pop(stack); instruction.operation_role = "MEMORY"
            elif op in {"CALL", "CALLCODE", "DELEGATECALL", "STATICCALL"}:
                count = 7 if op in {"CALL", "CALLCODE"} else 6
                values = _pop_many(stack, count)
                names = ["gas", "target", "value", "input_offset", "input_size", "output_offset", "output_size"] if count == 7 else ["gas", "target", "input_offset", "input_size", "output_offset", "output_size"]
                instruction.operand_sources.update(dict(zip(names, values))); stack.append(_merge(values, op, "CALL_RESULT")); instruction.operation_role = "CALL"
            elif op in {"JUMP", "JUMPI"}:
                instruction.operand_sources["destination"] = _pop(stack)
                if op == "JUMPI": instruction.operand_sources["condition"] = _pop(stack)
                instruction.operation_role = "CONTROL_FLOW"
            elif op in {"RETURN", "REVERT"}:
                instruction.operand_sources.update(dict(zip(("offset", "size"), _pop_many(stack, 2)))); instruction.operation_role = "CONTROL_FLOW"
            elif op == "SELFDESTRUCT":
                instruction.operand_sources["beneficiary"] = _pop(stack); instruction.operation_role = "CONTROL_FLOW"
            elif op in {"JUMPDEST", "STOP"}:
                instruction.operation_role = "CONTROL_FLOW"
            else:
                _unsupported(stack, instruction); instruction.operation_role = "OTHER"
                poisoned = True
            operands = []
            for item in instruction.operand_sources.values():
                operands.extend(item if isinstance(item, list) else [item])
            if instruction.analysis_status == "known":
                _status_from_values(instruction, operands)
            produces_value = (
                op == "PUSH0" or (op.startswith("PUSH") and op[4:].isdigit()) or
                op.startswith("DUP") or op in {"ADD", "SUB", "MUL", "DIV", "MOD", "SDIV", "SMOD", "ADDMOD", "MULMOD",
                    "AND", "OR", "XOR", "NOT", "SHL", "SHR", "SAR", "EXP", "SIGNEXTEND", "BYTE", "EQ", "LT", "GT",
                    "SLT", "SGT", "ISZERO", "CALLER", "CALLVALUE", "CALLDATALOAD", "CALLDATASIZE", "TIMESTAMP",
                    "NUMBER", "COINBASE", "DIFFICULTY", "PREVRANDAO", "GASLIMIT", "CHAINID", "BASEFEE", "BLOCKHASH",
                    "MLOAD", "SLOAD", "CALL", "CALLCODE", "DELEGATECALL", "STATICCALL"}
            )
            if produces_value and stack:
                instruction.provenance_bits = stack[-1].provenance_bits
                if instruction.analysis_status == "known":
                    _status_from_values(instruction, [stack[-1]])
            else:
                instruction.provenance_bits = 0
                for value in operands:
                    instruction.provenance_bits |= value.provenance_bits
            if instruction.analysis_status.startswith("analysis_failure"):
                instruction.provenance_bits |= SOURCE_BITS["UNKNOWN"]
            if poisoned and instruction.analysis_status == "known":
                instruction.analysis_status = "analysis_failure_unsupported_stack_effect"
                instruction.provenance_bits |= SOURCE_BITS["UNKNOWN"]
            records.append(instruction)
    return records
