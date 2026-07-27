"""Solidity source parsing and conservative local data-flow extraction."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path


IDENTIFIER = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")
TOKEN = re.compile(
    r"(?:0x[0-9A-Fa-f]+|[A-Za-z_$][A-Za-z0-9_$]*|\d+(?:\.\d+)?|==|!=|<=|>=|=>|\+\+|--|&&|\|\||\+=|-=|\*=|/=|.)",
    re.DOTALL,
)
EXECUTABLE_NODE_TYPES = {
    "function_definition",
    "constructor_definition",
    "fallback_receive_definition",
    "modifier_definition",
}
DECLARATION_WORDS = {
    "address", "bool", "bytes", "int", "mapping", "string", "uint", "var",
}
KEYWORDS = DECLARATION_WORDS | {
    "abstract", "as", "assembly", "break", "calldata", "constant", "constructor",
    "contract", "continue", "do", "else", "emit", "enum", "event", "external",
    "fallback", "for", "from", "function", "if", "import", "in", "indexed",
    "interface", "internal", "is", "library", "memory", "modifier", "new", "override",
    "payable", "pragma", "private", "public", "pure", "receive", "return", "returns",
    "revert", "storage", "struct", "true", "false", "try", "type", "unchecked",
    "using", "view", "virtual", "while",
}


@dataclass(frozen=True)
class SourceUnit:
    kind: str
    start_byte: int
    end_byte: int
    source: str
    tokens: list[str]
    dfg_nodes: list[tuple[str, int]]
    dfg_edges: list[tuple[int, int]]


def normalize_source(text: str) -> str:
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.split("\n")).strip() + "\n"


def source_sha256(text: str) -> str:
    return hashlib.sha256(normalize_source(text).encode("utf-8")).hexdigest()


def load_solidity_parser():
    try:
        from tree_sitter import Language, Parser
        from tree_sitter_solidity import language as solidity_language
    except ImportError as exc:
        raise RuntimeError(
            "Solidity parsing requires tree-sitter==0.22.3 and tree-sitter-solidity==1.2.13"
        ) from exc
    parser = Parser(Language(solidity_language(), "solidity"))
    return parser


def parse_solidity(text: str):
    parser = load_solidity_parser()
    tree = parser.parse(text.encode("utf-8"))
    if tree.root_node.has_error:
        raise ValueError("Solidity AST contains parse errors")
    return tree


def lexical_tokens(text: str) -> list[str]:
    return [token for token in TOKEN.findall(text) if not token.isspace()]


def _is_identifier(token: str) -> bool:
    return bool(IDENTIFIER.fullmatch(token)) and token not in KEYWORDS


def conservative_dfg(tokens: list[str]) -> tuple[list[tuple[str, int]], list[tuple[int, int]]]:
    """Build lexical def-use edges without applying vulnerability-specific rules."""
    latest_definition: dict[str, int] = {}
    nodes: list[tuple[str, int]] = []
    edges: list[tuple[int, int]] = []
    previous = ""
    for index, token in enumerate(tokens):
        if not _is_identifier(token):
            previous = token
            continue
        assignment = index + 1 < len(tokens) and tokens[index + 1] in {"=", "+=", "-=", "*=", "/="}
        declaration = previous in DECLARATION_WORDS or previous in {"(", ","} and (
            index > 1 and tokens[index - 2] in DECLARATION_WORDS
        )
        if declaration or assignment:
            node_id = len(nodes)
            nodes.append((token, index))
            if token in latest_definition:
                edges.append((latest_definition[token], node_id))
            latest_definition[token] = node_id
        elif token in latest_definition:
            node_id = len(nodes)
            nodes.append((token, index))
            edges.append((latest_definition[token], node_id))
        previous = token
    return nodes, edges


def _walk(node):
    yield node
    for child in node.children:
        yield from _walk(child)


def extract_source_units(text: str) -> list[SourceUnit]:
    tree = parse_solidity(text)
    encoded = text.encode("utf-8")
    nodes = [node for node in _walk(tree.root_node) if node.type in EXECUTABLE_NODE_TYPES]
    nodes.sort(key=lambda node: (node.start_byte, node.end_byte))
    units: list[SourceUnit] = []
    prefix_end = nodes[0].start_byte if nodes else len(encoded)
    if prefix_end:
        prefix = encoded[:prefix_end].decode("utf-8", errors="replace")
        tokens = lexical_tokens(prefix)
        if tokens:
            dfg_nodes, dfg_edges = conservative_dfg(tokens)
            units.append(SourceUnit("global", 0, prefix_end, prefix, tokens, dfg_nodes, dfg_edges))
    for node in nodes:
        source = encoded[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
        tokens = lexical_tokens(source)
        if not tokens:
            continue
        dfg_nodes, dfg_edges = conservative_dfg(tokens)
        units.append(SourceUnit(node.type, node.start_byte, node.end_byte, source, tokens, dfg_nodes, dfg_edges))
    if not units:
        raise ValueError("No executable Solidity source unit could be extracted")
    return units


def select_units(units: list[SourceUnit], max_units: int) -> list[SourceUnit]:
    if len(units) <= max_units:
        return units
    global_units = [unit for unit in units if unit.kind == "global"][:1]
    remaining = units[len(global_units):]
    budget = max(0, max_units - len(global_units))
    if budget == 0:
        return global_units
    indices = sorted({round(index * (len(remaining) - 1) / max(1, budget - 1)) for index in range(budget)})
    selected = global_units + [remaining[index] for index in indices]
    return selected[:max_units]


def source_file_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")
