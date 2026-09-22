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
        if value.analysis_status != "known":
            status = value.analysis_status
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


def analyze_basic_blocks(instructions, blocks):
    records = []
    for block in blocks:
        stack = []
        for index in block.instruction_indices:
            instruction = instructions[index]
            op = instruction.opcode
            instruction.analysis_status = "known"
            instruction.provenance_bits = 0
            instruction.operand_sources = {}
            if op.startswith("PUSH"):
                value = AbstractValue(SOURCE_BITS["CONSTANT"], "CONSTANT", int(instruction.immediate_operand or "0x0", 16), "known")
                stack.append(value); instruction.operation_role = "CONSTANT"
            elif op == "POP":
                instruction.operand_sources["value"] = _pop(stack); instruction.operation_role = "STACK_POP"
                if instruction.operand_sources["value"].analysis_status != "known":
                    instruction.analysis_status = instruction.operand_sources["value"].analysis_status
            elif op.startswith("DUP") and op[3:].isdigit():
                distance = int(op[3:]); value = stack[-distance] if len(stack) >= distance else AbstractValue.unknown_entry()
                stack.append(value); instruction.operand_sources["source"] = value; instruction.operation_role = "STACK_DUP"
            elif op.startswith("SWAP") and op[4:].isdigit():
                distance = int(op[4:])
                if len(stack) <= distance:
                    instruction.analysis_status = "analysis_failure_stack_underflow"; stack.clear(); stack.append(AbstractValue.failure())
                else:
                    stack[-1], stack[-1 - distance] = stack[-1 - distance], stack[-1]
                instruction.operation_role = "STACK_SWAP"
            elif op in {"ADD", "SUB", "MUL", "DIV", "MOD", "SDIV", "SMOD", "ADDMOD", "MULMOD", "AND", "OR", "XOR", "SHL", "SHR", "SAR"}:
                instruction.operand_sources["operands"] = _push_binary(stack, op, "ARITHMETIC"); instruction.operation_role = "ARITHMETIC"
            elif op in {"EQ", "LT", "GT", "SLT", "SGT", "ISZERO"}:
                values = _pop_many(stack, 1 if op == "ISZERO" else 2)
                stack.append(_merge(values, op, "COMPARISON")); instruction.operand_sources["operands"] = values; instruction.operation_role = "COMPARISON"
            elif op == "CALLER":
                stack.append(AbstractValue.source("CALLER")); instruction.operation_role = "CALLER"
            elif op == "CALLVALUE":
                stack.append(AbstractValue.source("CALLVALUE")); instruction.operation_role = "CALLVALUE"
            elif op == "CALLDATALOAD":
                instruction.operand_sources["offset"] = _pop(stack); stack.append(AbstractValue.source("CALLDATA")); instruction.operation_role = "CALLDATA"
            elif op in {"TIMESTAMP"}:
                stack.append(AbstractValue.source("TIMESTAMP")); instruction.operation_role = "BLOCK_ENV"
            elif op in {"NUMBER", "COINBASE", "DIFFICULTY", "PREVRANDAO", "GASLIMIT", "CHAINID", "BASEFEE", "BLOCKHASH"}:
                stack.append(AbstractValue.source("BLOCK_ENV")); instruction.operation_role = "BLOCK_ENV"
            elif op == "MLOAD":
                instruction.operand_sources["offset"] = _pop(stack); stack.append(AbstractValue.source("MEMORY")); instruction.operation_role = "MEMORY"
            elif op == "MSTORE":
                # EVM pops value first, then offset; keep semantic field names
                # independent of the physical stack-pop order.
                instruction.operand_sources["value"] = _pop(stack); instruction.operand_sources["offset"] = _pop(stack); instruction.operation_role = "MEMORY"
            elif op == "SLOAD":
                instruction.operand_sources["key"] = _pop(stack); stack.append(_merge([instruction.operand_sources["key"]], op, "STORAGE")); instruction.operation_role = "STORAGE"
            elif op == "SSTORE":
                # SSTORE pops value first and storage key second.
                instruction.operand_sources["value"] = _pop(stack); instruction.operand_sources["key"] = _pop(stack); instruction.operation_role = "STORAGE"
            elif op in {"CALL", "CALLCODE", "DELEGATECALL", "STATICCALL"}:
                count = 7 if op in {"CALL", "CALLCODE"} else 6
                values = _pop_many(stack, count)
                names = ["gas", "target", "value", "input_offset", "input_size", "output_offset", "output_size"] if count == 7 else ["gas", "target", "input_offset", "input_size", "output_offset", "output_size"]
                # The top of the EVM stack is output_size (or the last
                # argument), so reverse the popped list before naming fields.
                instruction.operand_sources.update(dict(zip(names, reversed(values)))); stack.append(AbstractValue.source("CALL_RESULT")); instruction.operation_role = "CALL"
            elif op in {"JUMP", "JUMPI"}:
                instruction.operand_sources["destination"] = _pop(stack)
                if op == "JUMPI": instruction.operand_sources["condition"] = _pop(stack)
                instruction.operation_role = "CONTROL_FLOW"
            elif op in {"JUMPDEST", "STOP", "RETURN", "REVERT", "SELFDESTRUCT", "INVALID"}:
                instruction.operation_role = "CONTROL_FLOW"
            else:
                _unsupported(stack, instruction); instruction.operation_role = "OTHER"
            if not instruction.provenance_bits:
                output_roles = {"CONSTANT", "STACK_DUP", "ARITHMETIC", "COMPARISON", "CALLER", "CALLVALUE",
                                "CALLDATA", "BLOCK_ENV", "MEMORY", "STORAGE", "CALL"}
                if instruction.operation_role in output_roles and stack:
                    instruction.provenance_bits = stack[-1].provenance_bits
                else:
                    values = []
                    for item in instruction.operand_sources.values():
                        values.extend(item if isinstance(item, list) else [item])
                    instruction.provenance_bits = sum(item.provenance_bits for item in values)
            if instruction.parse_status != "ok" and instruction.analysis_status == "known":
                instruction.analysis_status = "analysis_failure_parse_status"
            records.append(instruction)
    return records
