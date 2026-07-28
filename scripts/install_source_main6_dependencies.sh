#!/usr/bin/env bash
# Python 3.8 supports tree-sitter 0.21.3 (language ABI 13-14), but not the
# current Solidity wheel (ABI 15 and a Python >=3.9 runtime). Build the pinned
# ABI 14 grammar source instead of bypassing those version constraints.
set -euo pipefail

GRAMMAR_ARCHIVE="${1:?Pass the pinned tree-sitter-solidity-v1.2.2.tar.gz archive}"
EXPECTED_ARCHIVE_SHA256="bdd3f2834426b28e1fe1eb3c83b3507956ab90a1924592414e8b8040c903a48c"
BUILD_DIR="third_party_grammars/build_solidity_v1_2_2"
GRAMMAR_DIR="$BUILD_DIR/tree-sitter-solidity-1.2.2"
GRAMMAR_LIBRARY="third_party_grammars/solidity_v1_2_2_abi14.so"

actual_sha256=$(sha256sum "$GRAMMAR_ARCHIVE" | awk '{print $1}')
if [[ "$actual_sha256" != "$EXPECTED_ARCHIVE_SHA256" ]]; then
  echo "Unexpected Solidity grammar archive SHA256: $actual_sha256" >&2
  exit 1
fi

python -m pip install "tree-sitter==0.21.3"
python -m pip install "peft==0.12.0"

mkdir -p "$BUILD_DIR" third_party_grammars
tar -xzf "$GRAMMAR_ARCHIVE" -C "$BUILD_DIR"
gcc -shared -fPIC -O2 -I"$GRAMMAR_DIR/src" \
  "$GRAMMAR_DIR/src/parser.c" -o "$GRAMMAR_LIBRARY"

SOLIDITY_GRAMMAR_LIBRARY="$GRAMMAR_LIBRARY" python - <<'PY'
import os
from tree_sitter import Language, Parser

parser = Parser()
parser.set_language(Language(os.environ["SOLIDITY_GRAMMAR_LIBRARY"], "solidity"))
assert parser.parse(b"pragma solidity ^0.4.24; contract C {} ").root_node.type
print("[OK] Python 3.8 Solidity parser and LoRA dependencies are ready")
PY
