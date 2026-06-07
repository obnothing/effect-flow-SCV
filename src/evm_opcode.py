import re
import warnings


try:
    from pyevmasm import disassemble_all
except Exception:
    disassemble_all = None


OPCODES = {
    0x00: "STOP",
    0x01: "ADD",
    0x02: "MUL",
    0x03: "SUB",
    0x04: "DIV",
    0x05: "SDIV",
    0x06: "MOD",
    0x07: "SMOD",
    0x08: "ADDMOD",
    0x09: "MULMOD",
    0x0A: "EXP",
    0x0B: "SIGNEXTEND",
    0x10: "LT",
    0x11: "GT",
    0x12: "SLT",
    0x13: "SGT",
    0x14: "EQ",
    0x15: "ISZERO",
    0x16: "AND",
    0x17: "OR",
    0x18: "XOR",
    0x19: "NOT",
    0x1A: "BYTE",
    0x1B: "SHL",
    0x1C: "SHR",
    0x1D: "SAR",
    0x20: "SHA3",
    0x30: "ADDRESS",
    0x31: "BALANCE",
    0x32: "ORIGIN",
    0x33: "CALLER",
    0x34: "CALLVALUE",
    0x35: "CALLDATALOAD",
    0x36: "CALLDATASIZE",
    0x37: "CALLDATACOPY",
    0x38: "CODESIZE",
    0x39: "CODECOPY",
    0x3A: "GASPRICE",
    0x3B: "EXTCODESIZE",
    0x3C: "EXTCODECOPY",
    0x3D: "RETURNDATASIZE",
    0x3E: "RETURNDATACOPY",
    0x3F: "EXTCODEHASH",
    0x40: "BLOCKHASH",
    0x41: "COINBASE",
    0x42: "TIMESTAMP",
    0x43: "NUMBER",
    0x44: "PREVRANDAO",
    0x45: "GASLIMIT",
    0x46: "CHAINID",
    0x47: "SELFBALANCE",
    0x48: "BASEFEE",
    0x49: "BLOBHASH",
    0x4A: "BLOBBASEFEE",
    0x50: "POP",
    0x51: "MLOAD",
    0x52: "MSTORE",
    0x53: "MSTORE8",
    0x54: "SLOAD",
    0x55: "SSTORE",
    0x56: "JUMP",
    0x57: "JUMPI",
    0x58: "PC",
    0x59: "MSIZE",
    0x5A: "GAS",
    0x5B: "JUMPDEST",
    0x5F: "PUSH0",
    0xA0: "LOG0",
    0xA1: "LOG1",
    0xA2: "LOG2",
    0xA3: "LOG3",
    0xA4: "LOG4",
    0xF0: "CREATE",
    0xF1: "CALL",
    0xF2: "CALLCODE",
    0xF3: "RETURN",
    0xF4: "DELEGATECALL",
    0xF5: "CREATE2",
    0xFA: "STATICCALL",
    0xFD: "REVERT",
    0xFE: "INVALID",
    0xFF: "SELFDESTRUCT",
}

for i in range(1, 33):
    OPCODES[0x5F + i] = f"PUSH{i}"
for i in range(1, 17):
    OPCODES[0x7F + i] = f"DUP{i}"
    OPCODES[0x8F + i] = f"SWAP{i}"


def normalize_opcode_sequence(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return " ".join(text.split())


def normalize_bytecode(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    if text.startswith(("0x", "0X")):
        text = text[2:]
    text = re.sub(r"\s+", "", text)
    if len(text) % 2 != 0:
        warnings.warn("Skipping bytecode with odd hex length.")
        return None
    if not re.fullmatch(r"[0-9a-fA-F]*", text):
        warnings.warn("Skipping bytecode with non-hex characters.")
        return None
    return text


def bytecode_to_opcode_sequence(bytecode):
    hex_text = normalize_bytecode(bytecode)
    if not hex_text:
        return None

    try:
        bytecode_bytes = bytes.fromhex(hex_text)
    except ValueError:
        warnings.warn("Skipping bytecode that cannot be parsed as hex.")
        return None

    if disassemble_all is not None:
        try:
            opcodes = pyevmasm_instructions_to_tokens(disassemble_all(bytecode_bytes))
            if opcodes and push_operands_are_preserved(opcodes):
                return " ".join(opcodes)
            warnings.warn(
                "pyevmasm output did not preserve PUSH operands; falling back."
            )
        except Exception as exc:
            warnings.warn(f"pyevmasm disassembly failed; falling back. {exc}")

    return fallback_bytecode_to_opcode(bytecode_bytes)


def push_operands_are_preserved(opcodes):
    for idx, token in enumerate(opcodes):
        if re.fullmatch(r"PUSH(?:[1-9]|[12][0-9]|3[0-2])", token):
            return idx + 1 < len(opcodes) and opcodes[idx + 1].startswith("0x")
    return True


def pyevmasm_instructions_to_tokens(instructions):
    opcodes = []
    for instruction in instructions:
        name = str(getattr(instruction, "name", "")).upper()
        if not name:
            continue
        opcodes.append(name)
        if re.fullmatch(r"PUSH(?:[1-9]|[12][0-9]|3[0-2])", name):
            operand = instruction_operand_to_hex(instruction)
            if operand:
                opcodes.append(operand)
    return opcodes


def instruction_operand_to_hex(instruction):
    for attr in ("operand", "argument", "immediate"):
        value = getattr(instruction, attr, None)
        if value is None:
            continue
        if isinstance(value, bytes):
            return f"0x{value.hex()}"
        if isinstance(value, int):
            size = getattr(instruction, "operand_size", None)
            width = int(size) * 2 if size else 0
            return f"0x{value:0{width}x}" if width else f"0x{value:x}"
        text = str(value).strip().lower()
        if text.startswith("0x"):
            return text
        if re.fullmatch(r"[0-9a-f]+", text):
            return f"0x{text}"
    return None


def fallback_bytecode_to_opcode(bytecode_bytes):
    opcodes = []
    i = 0
    while i < len(bytecode_bytes):
        code = bytecode_bytes[i]
        name = OPCODES.get(code, f"UNKNOWN_{code:02X}")
        opcodes.append(name)
        i += 1
        if 0x60 <= code <= 0x7F:
            operand_size = code - 0x5F
            operand = bytecode_bytes[i : i + operand_size]
            opcodes.append(f"0x{operand.hex()}")
            i += operand_size
    return " ".join(opcodes) if opcodes else None
