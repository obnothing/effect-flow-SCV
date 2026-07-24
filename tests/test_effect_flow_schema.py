import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from effect_flow_schema import (  # noqa: E402
    EFFECT_TO_ID,
    EFPP_PATTERNS,
    annotate_effect_types,
    annotate_efpp_patterns,
    build_token_units,
)


class TinyTokenizer:
    def tokenize(self, value, add_special_tokens=False):
        text = str(value).lower()
        digits = text[2:] if text.startswith("0x") else text
        if len(digits) == 40:
            return ["<ADDR>"]
        if len(digits) == 8:
            return ["<SELECTOR>"]
        return ["<HEX_SHORT>"]


def test_effect_types_keep_primary_and_multihot_semantics():
    units = build_token_units(
        "SLOAD SSTORE CALL ISZERO JUMPI MSTORE PUSH1 0x01 PUSH20 "
        "0x1111111111111111111111111111111111111111",
        TinyTokenizer(),
    )
    effects = annotate_effect_types(units)
    sload = effects[0]
    assert sload["primary_id"] == EFFECT_TO_ID["StateRead"]
    assert sload["multihot"][EFFECT_TO_ID["StorageOrMemoryHeavy"]] == 1
    call = effects[2]
    assert call["primary_id"] == EFFECT_TO_ID["ExternalCall"]
    assert call["multihot"][EFFECT_TO_ID["ReturnCheck"]] == 1
    push1_operand = effects[7]
    assert push1_operand["primary_id"] == -100
    assert push1_operand["loss_mask"] == 0
    address_operand = effects[-1]
    assert address_operand["primary_id"] == EFFECT_TO_ID["AddressLiteral"]


def test_efpp_patterns_cover_core_weak_rules():
    sequence = (
        "CALLER EQ JUMPI SSTORE CALL ISZERO JUMPI PUSH20 "
        "0x1111111111111111111111111111111111111111 "
        "TIMESTAMP LT JUMPI DIV ADD MUL INVALID"
    )
    units = build_token_units(sequence, TinyTokenizer())
    labels, matches = annotate_efpp_patterns(units)
    active = {name for name, value in zip(EFPP_PATTERNS, labels) if value}
    expected = {
        "has_external_call",
        "has_state_write",
        "state_write_before_call",
        "call_with_return_check",
        "state_write_with_auth_guard",
        "sensitive_call_with_auth_guard",
        "env_used_in_condition",
        "arithmetic_chain",
        "div_before_mul",
        "arithmetic_followed_by_guard_or_revert",
        "has_hardcoded_address",
        "hardcoded_address_near_call",
        "revert_or_invalid_present",
    }
    assert expected.issubset(active)
    assert "call_without_return_check" not in active
    assert set(matches) == active


def test_panic_selector_is_retained_as_revert_effect():
    units = build_token_units("PUSH4 0x4e487b71", TinyTokenizer())
    effects = annotate_effect_types(units)
    assert effects[1]["primary_id"] == EFFECT_TO_ID["RevertOrAssert"]
    labels, _ = annotate_efpp_patterns(units)
    active = {name for name, value in zip(EFPP_PATTERNS, labels) if value}
    assert "panic_selector_present" in active


def test_multirole_etp_has_control_transfer_and_maximum_two_roles():
    units = build_token_units("SLOAD GASPRICE JUMP JUMPDEST CALL ISZERO JUMPI", TinyTokenizer())
    effects = annotate_effect_types(units)
    assert len(EFFECT_TO_ID) == 17
    assert effects[0]["multihot"][EFFECT_TO_ID["StateRead"]] == 1
    assert effects[0]["multihot"][EFFECT_TO_ID["StorageOrMemoryHeavy"]] == 1
    assert effects[1]["multihot"][EFFECT_TO_ID["EnvDependency"]] == 1
    assert effects[1]["multihot"][EFFECT_TO_ID["GasOrValue"]] == 1
    assert effects[2]["multihot"][EFFECT_TO_ID["ControlTransfer"]] == 1
    assert effects[3]["multihot"][EFFECT_TO_ID["ControlTransfer"]] == 1
    assert effects[-1]["multihot"][EFFECT_TO_ID["ControlTransfer"]] == 0
    assert all(1 <= sum(effect["multihot"]) <= 2 for effect in effects if effect["loss_mask"])
