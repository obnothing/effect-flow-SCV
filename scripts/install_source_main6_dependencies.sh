#!/usr/bin/env bash
# tree-sitter 0.22.3 and tree-sitter-solidity 1.2.13 declare Python >=3.9,
# but their runtime implementation is compatible with Python 3.8. Solidity's
# grammar uses language ABI 15, which requires the 0.22 runtime.
set -euo pipefail

TREE_SITTER_SOURCE="${1:-tree-sitter==0.22.3}"
SOLIDITY_WHEEL="${2:-tree-sitter-solidity==1.2.13}"

# Passing local paths supports compute servers without PyPI/DNS access.
python -m pip install --ignore-requires-python --no-build-isolation --no-deps "$TREE_SITTER_SOURCE"
python -m pip install "peft==0.12.0"
python -m pip install --ignore-requires-python --no-deps "$SOLIDITY_WHEEL"

python - <<'PY'
from tree_sitter import Language, Parser
from tree_sitter_solidity import language

parser = Parser(Language(language()))
assert parser.parse(b"pragma solidity ^0.4.24; contract C {} ").root_node.type
print("[OK] Python 3.8 Solidity parser and LoRA dependencies are ready")
PY
