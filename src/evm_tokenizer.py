import json
import re
from pathlib import Path

from evm_opcode import OPCODES


SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
NORMALIZED_OPERAND_TOKENS = [
    "<ADDR>",
    "<BYTES32>",
    "<SELECTOR>",
    "<HEX>",
    "<HEX_SHORT>",
    "<HEX1>",
    "<HEX2>",
    "<HEX4>",
    "<HEX8>",
    "<INVALID_HEX>",
]
COMMON_OPERANDS = {
    "0x00",
    "0x01",
    "0x02",
    "0x03",
    "0x04",
    "0x05",
    "0x08",
    "0x0a",
    "0x10",
    "0x20",
    "0x40",
    "0x60",
    "0x80",
    "0xff",
    "0xffff",
    "0xffffffff",
}
MNEMONIC_TOKENS = sorted(set(OPCODES.values()))
PUSH_PATTERN = re.compile(r"^PUSH(?:0|[1-9]|[12][0-9]|3[0-2])$")


def canonical_hex(value):
    text = str(value).strip().lower()
    if not text:
        return ""
    if not text.startswith("0x"):
        text = f"0x{text}"
    return text


def is_hex_literal(value):
    return str(value).strip().lower().startswith("0x")


def hex_byte_length(value):
    text = canonical_hex(value)
    if not re.fullmatch(r"0x[0-9a-f]*", text):
        return None
    hex_part = text[2:]
    if len(hex_part) == 0 or len(hex_part) % 2 != 0:
        return None
    return len(hex_part) // 2


def normalize_operand(
    value,
    preserved_operands=None,
    preserve_common_operands=True,
    normalize_rare_long_operands=True,
):
    operand = canonical_hex(value)
    preserved_operands = preserved_operands or set()
    if preserve_common_operands and operand in COMMON_OPERANDS:
        return operand
    if operand in preserved_operands:
        return operand

    if not re.fullmatch(r"0x[0-9a-f]*", operand):
        return "<INVALID_HEX>"
    byte_len = hex_byte_length(operand)
    if byte_len is None:
        return "<INVALID_HEX>"

    if normalize_rare_long_operands:
        if byte_len == 20:
            return "<ADDR>"
        if byte_len == 32:
            return "<BYTES32>"
        if byte_len == 4:
            return "<SELECTOR>"
        if byte_len > 8:
            return "<HEX>"

    if byte_len in {1, 2, 4, 8}:
        return f"<HEX{byte_len}>"
    return "<HEX_SHORT>"


def split_opcode_sequence(opcode_sequence):
    if opcode_sequence is None:
        return []
    return str(opcode_sequence).strip().split()


def tokenize_opcode_sequence(
    opcode_sequence,
    preserved_operands=None,
    preserve_common_operands=True,
    normalize_rare_long_operands=True,
    add_special_tokens=True,
):
    raw_tokens = split_opcode_sequence(opcode_sequence)
    output_tokens = []
    index = 0
    while index < len(raw_tokens):
        token = raw_tokens[index]
        upper_token = token.upper()
        if PUSH_PATTERN.match(upper_token):
            output_tokens.append(upper_token)
            if index + 1 < len(raw_tokens):
                next_token = raw_tokens[index + 1]
                if is_hex_literal(next_token):
                    output_tokens.append(
                        normalize_operand(
                            next_token,
                            preserved_operands=preserved_operands,
                            preserve_common_operands=preserve_common_operands,
                            normalize_rare_long_operands=normalize_rare_long_operands,
                        )
                    )
                    index += 2
                    continue
        elif upper_token in MNEMONIC_TOKENS:
            output_tokens.append(upper_token)
        elif is_hex_literal(token):
            output_tokens.append(
                normalize_operand(
                    token,
                    preserved_operands=preserved_operands,
                    preserve_common_operands=preserve_common_operands,
                    normalize_rare_long_operands=normalize_rare_long_operands,
                )
            )
        else:
            output_tokens.append(upper_token)
        index += 1

    if add_special_tokens:
        return ["[CLS]"] + output_tokens + ["[SEP]"]
    return output_tokens


class EVMOpcodeTokenizer:
    def __init__(
        self,
        vocab,
        preserved_operands=None,
        preserve_common_operands=True,
        normalize_rare_long_operands=True,
    ):
        self.vocab = dict(vocab)
        self.id_to_token = {idx: token for token, idx in self.vocab.items()}
        self.preserved_operands = set(preserved_operands or [])
        self.preserve_common_operands = preserve_common_operands
        self.normalize_rare_long_operands = normalize_rare_long_operands
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.mask_token = "[MASK]"
        self.pad_token_id = self.vocab[self.pad_token]
        self.unk_token_id = self.vocab[self.unk_token]

    @classmethod
    def from_vocab_file(cls, path):
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            data["token_to_id"],
            preserved_operands=data.get("preserved_operands", []),
            preserve_common_operands=data.get("preserve_common_operands", True),
            normalize_rare_long_operands=data.get(
                "normalize_rare_long_operands", True
            ),
        )

    def tokenize(self, opcode_sequence, add_special_tokens=True):
        return tokenize_opcode_sequence(
            opcode_sequence,
            preserved_operands=self.preserved_operands,
            preserve_common_operands=self.preserve_common_operands,
            normalize_rare_long_operands=self.normalize_rare_long_operands,
            add_special_tokens=add_special_tokens,
        )

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab.get(token, self.unk_token_id) for token in tokens]

    def encode(self, opcode_sequence, add_special_tokens=True):
        return self.convert_tokens_to_ids(
            self.tokenize(opcode_sequence, add_special_tokens=add_special_tokens)
        )

    def __call__(
        self,
        opcode_sequence,
        add_special_tokens=True,
        truncation=False,
        max_length=None,
    ):
        input_ids = self.encode(
            opcode_sequence, add_special_tokens=add_special_tokens
        )
        if truncation and max_length is not None:
            input_ids = input_ids[:max_length]
        return {"input_ids": input_ids}

    def __len__(self):
        return len(self.vocab)
