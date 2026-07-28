#!/usr/bin/env bash
# tree-sitter-solidity 1.2.13 ships a cp38 abi3 wheel but declares Python >=3.9.
# The project uses its native language pointer with tree-sitter 0.21.3 on Python 3.8.
set -euo pipefail

python -m pip install --upgrade "tree-sitter==0.21.3" "peft==0.12.0"
python -m pip install --ignore-requires-python --no-deps "tree-sitter-solidity==1.2.13"

python - <<'PY'
from tree_sitter import Language, Parser
from tree_sitter_solidity import language

parser = Parser()
parser.set_language(Language(language(), "solidity"))
assert parser.parse(b"pragma solidity ^0.4.24; contract C {} ").root_node.type
print("[OK] Python 3.8 Solidity parser and LoRA dependencies are ready")
PY
