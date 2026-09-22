"""Conservative parser for disassembled EVM opcode text.

The parser derives local program counters from instruction widths. It does not
claim that those offsets were independently verified against runtime bytes.
"""

from dataclasses import dataclass, field
import re


PUSH_RE = re.compile(r"^PUSH([0-9]+)$", re.IGNORECASE)
HEX_RE = re.compile(r"^0x[0-9a-fA-F]+$")
UNKNOWN_RE = re.compile(r"^'([0-9a-fA-F]{2})'(?:\(Unknown)?$", re.IGNORECASE)
TERMINATORS = {"STOP", "RETURN", "REVERT", "SELFDESTRUCT", "INVALID", "RETURNCONTRACT"}
CONTROL_FLOW = {"JUMP", "JUMPI", "JUMPDEST", *TERMINATORS}


@dataclass
class Instruction:
    instruction_index: int
    derived_pc: int
    opcode: str
    immediate_operand: str | None
    immediate_width: int
    source_text_offset: int
    source_token_indices: list[int]
    model_token_indices: list[int]
    parse_status: str = "ok"
    basic_block_id: int = -1
    analysis_status: str = "not_analyzed"
    operation_role: str = "OTHER"
    provenance_bits: int = 0
    operand_sources: dict = field(default_factory=dict)


@dataclass
class BasicBlock:
    block_id: int
    instruction_indices: list[int]
    entry_instruction_index: int
    terminator: str | None


def _raw_token_spans(text):
    return [(match.group(0), match.start()) for match in re.finditer(r"\S+", str(text or ""))]


def _is_push(opcode):
    match = PUSH_RE.match(opcode)
    return match is not None and 0 <= int(match.group(1)) <= 32


def _push_width(opcode):
    return int(PUSH_RE.match(opcode).group(1))


def _unknown_annotation(tokens, index):
    if index + 1 >= len(tokens):
        return None
    match = UNKNOWN_RE.match(tokens[index])
    if not match:
        return None
    if tokens[index + 1].upper() == "OPCODE)":
        return f"UNKNOWN_0x{match.group(1).upper()}"
    if index + 2 < len(tokens) and tokens[index + 1].upper() == "(UNKNOWN" and tokens[index + 2].upper() == "OPCODE)":
        return f"UNKNOWN_0x{match.group(1).upper()}"
    return None


def _unknown_annotation_width(tokens, index):
    if _unknown_annotation(tokens, index) is None:
        return 0
    return 3 if index + 2 < len(tokens) and tokens[index + 1].upper() == "(UNKNOWN" else 2


def _model_token_alignment(raw_tokens, tokenizer):
    model_tokens = tokenizer.tokenize(" ".join(raw_tokens), add_special_tokens=False)
    positions = []
    raw_index = 0
    model_index = 0
    while raw_index < len(raw_tokens):
        opcode = raw_tokens[raw_index].upper()
        source_indices = [raw_index]
        model_indices = [model_index]
        model_index += 1
        raw_index += 1
        if _is_push(opcode) and _push_width(opcode) > 0 and raw_index < len(raw_tokens) and HEX_RE.match(raw_tokens[raw_index]):
            source_indices.append(raw_index)
            model_indices.append(model_index)
            model_index += 1
            raw_index += 1
        elif _unknown_annotation(raw_tokens, raw_index - 1):
            # The annotation consumed two source tokens but one normalized
            # unknown instruction may still occupy two model tokens because
            # the existing tokenizer has no special UNKNOWN token.
            if raw_index < len(raw_tokens) and raw_tokens[raw_index].upper() == "OPCODE)":
                source_indices.append(raw_index)
                model_indices.append(model_index)
                model_index += 1
                raw_index += 1
            elif raw_index + 1 < len(raw_tokens) and raw_tokens[raw_index].upper() == "(UNKNOWN" and raw_tokens[raw_index + 1].upper() == "OPCODE)":
                source_indices.extend([raw_index, raw_index + 1])
                model_indices.extend([model_index, model_index + 1])
                model_index += 2
                raw_index += 2
        positions.append((source_indices, model_indices))
    if model_index != len(model_tokens):
        raise ValueError(f"Tokenizer alignment mismatch: derived={model_index}, tokenizer={len(model_tokens)}")
    return positions


def parse_disassembled_opcode(opcode_text, tokenizer=None):
    spans = _raw_token_spans(opcode_text)
    tokens = [token for token, _ in spans]
    alignment = _model_token_alignment(tokens, tokenizer) if tokenizer is not None else None
    instructions, errors = [], []
    token_index = 0
    instruction_index = 0
    pc = 0
    while token_index < len(tokens):
        source_start = spans[token_index][1]
        annotation = _unknown_annotation(tokens, token_index)
        if annotation:
            annotation_width = _unknown_annotation_width(tokens, token_index)
            source_indices = list(range(token_index, token_index + annotation_width))
            model_indices = alignment[instruction_index][1] if alignment else []
            instruction = Instruction(instruction_index, pc, annotation, None, 0, source_start,
                                      source_indices, model_indices, "unknown_opcode_annotation")
            instructions.append(instruction); errors.append(instruction.parse_status)
            instruction_index += 1; pc += 1; token_index += annotation_width; continue
        opcode = tokens[token_index].upper()
        source_indices = [token_index]
        immediate = None; width = 0; status = "ok"
        if _is_push(opcode):
            width = _push_width(opcode)
            if width and token_index + 1 < len(tokens) and HEX_RE.match(tokens[token_index + 1]):
                immediate = tokens[token_index + 1]
                source_indices.append(token_index + 1)
                actual_width = max(0, (len(immediate) - 2) // 2)
                if actual_width != width:
                    status = "push_immediate_width_mismatch"
            elif width:
                status = "missing_push_immediate"
        elif HEX_RE.match(tokens[token_index]):
            status = "orphan_immediate"
        elif opcode.startswith("'FE'"):
            status = "unknown_opcode"
        model_indices = alignment[instruction_index][1] if alignment else []
        instruction = Instruction(instruction_index, pc, opcode, immediate, width, source_start,
                                  source_indices, model_indices, status)
        instructions.append(instruction)
        if status != "ok":
            errors.append(status)
        token_index += len(source_indices)
        instruction_index += 1
        pc += 1 + width
    blocks = _partition_blocks(instructions)
    return instructions, blocks, {"errors": errors, "token_count": len(tokens), "instruction_count": len(instructions)}


def _partition_blocks(instructions):
    blocks, current = [], []
    for instruction in instructions:
        if current and instruction.opcode == "JUMPDEST":
            blocks.append(current); current = []
        current.append(instruction)
        if instruction.opcode in TERMINATORS or instruction.opcode in {"JUMP", "JUMPI"}:
            blocks.append(current); current = []
    if current:
        blocks.append(current)
    result = []
    for block_id, block in enumerate(blocks):
        terminator = block[-1].opcode if block[-1].opcode in CONTROL_FLOW else None
        for instruction in block:
            instruction.basic_block_id = block_id
        result.append(BasicBlock(block_id, [item.instruction_index for item in block], block[0].instruction_index, terminator))
    return result
