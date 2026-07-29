"""AST-aligned Solidity units and contract relations for Source-Main6 v2."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path


UNIT_KINDS = ("global", "constructor", "fallback_receive", "modifier", "function")
NODE_TYPES = ("contract", "function", "modifier", "state_variable", "parameter", "local_variable")
ROLES = ("definition", "use", "parameter", "state_read", "state_write")
EDGE_TYPES = ("def_use", "state_read", "state_write", "call", "modifier_application", "inheritance")
EXECUTABLE = {"function_definition", "constructor_definition", "fallback_receive_definition", "modifier_definition"}


@dataclass
class Symbol:
    name: str
    start_byte: int
    end_byte: int
    role: str
    node_type: str
    unit_index: int


@dataclass
class SourceUnit:
    kind: str
    start_byte: int
    end_byte: int
    source: str
    mandatory: bool
    symbols: list[Symbol] = field(default_factory=list)
    local_edges: list[tuple[int, int, str]] = field(default_factory=list)


@dataclass
class ContractGraph:
    units: list[SourceUnit]
    node_types: list[str]
    node_names: list[str]
    node_units: list[int]
    node_spans: list[tuple[int, int]]
    edges: list[tuple[int, int, str]]
    unresolved_calls: int = 0
    unresolved_inheritance: int = 0


def _walk(node):
    yield node
    for child in node.children:
        yield from _walk(child)


def _text(source: bytes, node) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _identifiers(source: bytes, node):
    return [child for child in _walk(node) if child.type == "identifier"]


def _first_identifier(source: bytes, node):
    values = _identifiers(source, node)
    return values[0] if values else None


def _unit_kind(node, source: bytes) -> str:
    if node.type == "constructor_definition":
        return "constructor"
    if node.type == "modifier_definition":
        return "modifier"
    if node.type == "fallback_receive_definition":
        return "fallback_receive"
    first = _first_identifier(source, node)
    return "fallback_receive" if first is None else "function"


def parse_source(text: str):
    from solidity_graph_utils import load_solidity_parser

    parser = load_solidity_parser()
    tree = parser.parse(text.encode("utf-8"))
    if tree.root_node.has_error:
        raise ValueError("Solidity AST contains parse errors")
    return tree


def normalize_source_ast(text: str) -> str:
    """Mask comments with a byte lexer while preserving literals and offsets."""
    raw = bytearray(str(text).replace("\r\n", "\n").replace("\r", "\n").encode("utf-8"))
    index, state, escaped = 0, "code", False
    while index < len(raw):
        byte = raw[index]
        nxt = raw[index + 1] if index + 1 < len(raw) else None
        if state in {"single", "double"}:
            if escaped:
                escaped = False
                raw[index] = 32
            elif byte == 92:
                escaped = True
                raw[index] = 32
            elif (state == "single" and byte == 39) or (state == "double" and byte == 34):
                state = "code"
            else:
                raw[index] = 32
            index += 1
            continue
        if state == "line":
            if byte == 10:
                state = "code"
            else:
                raw[index] = 32
            index += 1
            continue
        if state == "block":
            if byte == 42 and nxt == 47:
                raw[index] = raw[index + 1] = 32
                index += 2; state = "code"; continue
            if byte != 10:
                raw[index] = 32
            index += 1
            continue
        if byte == 39:
            state = "single"
        elif byte == 34:
            state = "double"
        elif byte == 47 and nxt == 47:
            raw[index] = raw[index + 1] = 32
            index += 2; state = "line"; continue
        elif byte == 47 and nxt == 42:
            raw[index] = raw[index + 1] = 32
            index += 2; state = "block"; continue
        index += 1
    return raw.decode("utf-8", errors="strict")


def normalized_sha256(text: str) -> str:
    return hashlib.sha256(normalize_source_ast(text).encode("utf-8")).hexdigest()


def extract_audit_units(text: str) -> list[SourceUnit]:
    """Extract source ranges from one complete, normalized contract AST.

    This intentionally does not parse function snippets as standalone Solidity
    files: a function definition is only valid inside its enclosing contract.
    """
    normalized = normalize_source_ast(text)
    raw = normalized.encode("utf-8")
    tree = parse_source(normalized)
    executable = sorted((node for node in _walk(tree.root_node) if node.type in EXECUTABLE), key=lambda value: (value.start_byte, value.end_byte))
    if not executable:
        raise ValueError("No executable Solidity source unit could be extracted")
    units = []
    if executable[0].start_byte:
        units.append(SourceUnit("global", 0, executable[0].start_byte, raw[:executable[0].start_byte].decode("utf-8", errors="replace"), True))
    for node in executable:
        kind = _unit_kind(node, raw)
        units.append(SourceUnit(kind, node.start_byte, node.end_byte, _text(raw, node), kind != "function"))
    return units


def _declaration_nodes(node):
    return [item for item in _walk(node) if item.type in {"variable_declaration", "parameter", "state_variable_declaration"}]


def _symbol_role(node, state_names: set[str]) -> tuple[str, str]:
    if node.type == "parameter":
        return "parameter", "parameter"
    if node.type == "state_variable_declaration":
        return "definition", "state_variable"
    return "definition", "local_variable"


def extract_contract_graph(text: str) -> ContractGraph:
    normalized = normalize_source_ast(text)
    raw = normalized.encode("utf-8")
    tree = parse_source(normalized)
    executable = sorted((node for node in _walk(tree.root_node) if node.type in EXECUTABLE), key=lambda value: (value.start_byte, value.end_byte))
    if not executable:
        raise ValueError("No executable Solidity source unit could be extracted")
    first_start = executable[0].start_byte
    units = [SourceUnit("global", 0, first_start, raw[:first_start].decode("utf-8", errors="replace"), True)] if first_start else []
    for node in executable:
        kind = _unit_kind(node, raw)
        units.append(SourceUnit(kind, node.start_byte, node.end_byte, _text(raw, node), kind != "function"))

    state_nodes = [node for node in _walk(tree.root_node) if node.type == "state_variable_declaration"]
    state_names = {_text(raw, ident) for node in state_nodes for ident in [_first_identifier(raw, node)] if ident is not None}
    node_types, node_names, node_units, node_spans = ["contract"], ["contract"], [-1], [(0, 0)]
    graph_index: dict[tuple[str, str, int], int] = {("contract", "contract", -1): 0}
    edges: list[tuple[int, int, str]] = []
    function_by_name: dict[str, int] = {}
    modifier_by_name: dict[str, int] = {}
    unit_graph_ids: dict[int, int] = {}

    def add_graph_node(kind: str, name: str, unit_index: int, start: int, end: int) -> int:
        key = (kind, name, unit_index)
        if key not in graph_index:
            graph_index[key] = len(node_types)
            node_types.append(kind); node_names.append(name); node_units.append(unit_index); node_spans.append((start, end))
        return graph_index[key]

    for unit_index, unit in enumerate(units):
        if unit.kind == "global":
            continue
        unit_tree = parse_source(unit.source).root_node
        identifier = _first_identifier(unit.source.encode("utf-8"), unit_tree)
        name = _text(unit.source.encode("utf-8"), identifier) if identifier is not None else unit.kind
        kind = "modifier" if unit.kind == "modifier" else "function"
        idx = add_graph_node(kind, name, unit_index, unit.start_byte, unit.end_byte)
        unit_graph_ids[unit_index] = idx
        if kind == "modifier": modifier_by_name[name] = idx
        else: function_by_name[name] = idx

    # State declarations are anchored to the global unit when present.
    for node in state_nodes:
        ident = _first_identifier(raw, node)
        if ident is not None:
            add_graph_node("state_variable", _text(raw, ident), 0 if units and units[0].kind == "global" else -1, ident.start_byte, ident.end_byte)

    unresolved_calls = unresolved_inheritance = 0
    for unit_index, unit in enumerate(units):
        if unit.kind == "global":
            continue
        unit_bytes = unit.source.encode("utf-8")
        root = parse_source(unit.source).root_node
        definitions: dict[str, int] = {}
        symbol_to_graph: dict[str, int] = {}
        for declaration in _declaration_nodes(root):
            ident = _first_identifier(unit_bytes, declaration)
            if ident is None:
                continue
            name = _text(unit_bytes, ident)
            role, node_type = _symbol_role(declaration, state_names)
            absolute_start, absolute_end = unit.start_byte + ident.start_byte, unit.start_byte + ident.end_byte
            unit.symbols.append(Symbol(name, absolute_start, absolute_end, role, node_type, unit_index))
            graph_node = add_graph_node(node_type, name, unit_index, absolute_start, absolute_end)
            definitions[name] = len(unit.symbols) - 1; symbol_to_graph[name] = graph_node
        for ident in _identifiers(unit_bytes, root):
            name = _text(unit_bytes, ident)
            if name in definitions and unit.symbols[definitions[name]].start_byte == unit.start_byte + ident.start_byte:
                continue
            role = "state_read" if name in state_names else "use"
            absolute_start, absolute_end = unit.start_byte + ident.start_byte, unit.start_byte + ident.end_byte
            unit.symbols.append(Symbol(name, absolute_start, absolute_end, role, "state_variable" if name in state_names else "local_variable", unit_index))
            use_index = len(unit.symbols) - 1
            if name in definitions:
                unit.local_edges.append((definitions[name], use_index, "def_use"))
        for call in (item for item in _walk(root) if item.type == "call_expression"):
            ident = _first_identifier(unit_bytes, call)
            if ident is None:
                continue
            name = _text(unit_bytes, ident)
            caller = unit_graph_ids.get(unit_index)
            if caller is not None and name in function_by_name:
                edges.append((caller, function_by_name[name], "call"))
            elif name not in function_by_name:
                unresolved_calls += 1
        if unit.kind in {"function", "constructor", "fallback_receive"}:
            target = unit_graph_ids.get(unit_index)
            for invocation in (item for item in _walk(root) if item.type == "modifier_invocation"):
                ident = _first_identifier(unit_bytes, invocation)
                if ident is None or target is None:
                    continue
                modifier = modifier_by_name.get(_text(unit_bytes, ident))
                if modifier is not None:
                    edges.append((modifier, target, "modifier_application"))
    return ContractGraph(units, node_types, node_names, node_units, node_spans, edges, unresolved_calls, unresolved_inheritance)


def source_file_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")
