import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evm_execution_analysis import ROLE_INDEX, analyze_opcode_execution  # noqa: E402


class TinyTokenizer:
    def tokenize(self, text, add_special_tokens=False):
        return str(text).split()


def test_stack_lineage_and_roles():
    result = analyze_opcode_execution("PUSH1 0x01 PUSH1 0x02 ADD SSTORE", TinyTokenizer())
    assert result["report"]["dependency_count"] >= 3
    features = result["token_features"]
    assert features.shape[1] == 28
    assert features[4, ROLE_INDEX["binary_transform"]] == 1.0


def test_dup_swap_and_underflow_are_stable():
    result = analyze_opcode_execution("PUSH1 0x01 DUP1 SWAP1 ADD", TinyTokenizer())
    assert result["report"]["dependency_count"] >= 1
    underflow = analyze_opcode_execution("ADD", TinyTokenizer())
    assert underflow["report"]["underflow_count"] == 2
    assert np.isfinite(underflow["token_features"]).all()

