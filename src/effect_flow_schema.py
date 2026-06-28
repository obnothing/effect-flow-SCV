"""Rule-based EVM effect annotations used by Stage 16A.

The rules in this module are weak semantic signals. They are not vulnerability
labels and should not be interpreted as proof that a contract is vulnerable.
"""

from dataclasses import dataclass


EFFECT_TYPES = [
    "Normal",
    "ExternalCall",
    "StateRead",
    "StateWrite",
    "AuthSource",
    "ControlGuard",
    "ReturnCheck",
    "EnvDependency",
    "Arithmetic",
    "AddressLiteral",
    "CreateContract",
    "Destructive",
    "RevertOrAssert",
    "GasOrValue",
    "HashOrCrypto",
    "StorageOrMemoryHeavy",
]
EFFECT_TO_ID = {name: index for index, name in enumerate(EFFECT_TYPES)}

EFFECT_PRIORITY = [
    "RevertOrAssert",
    "Destructive",
    "CreateContract",
    "ExternalCall",
    "StateWrite",
    "StateRead",
    "AuthSource",
    "AddressLiteral",
    "EnvDependency",
    "ReturnCheck",
    "ControlGuard",
    "Arithmetic",
    "GasOrValue",
    "HashOrCrypto",
    "StorageOrMemoryHeavy",
    "Normal",
]

EFPP_PATTERNS = [
    "has_external_call",
    "has_delegatecall",
    "has_state_read",
    "has_state_write",
    "call_before_state_write",
    "state_write_before_call",
    "call_without_return_check",
    "call_with_return_check",
    "state_write_with_auth_guard",
    "state_write_without_auth_guard",
    "sensitive_call_with_auth_guard",
    "sensitive_call_without_auth_guard",
    "env_used_in_condition",
    "env_used_in_arithmetic_or_hash",
    "arithmetic_chain",
    "div_before_mul",
    "arithmetic_followed_by_guard_or_revert",
    "has_hardcoded_address",
    "hardcoded_address_near_call",
    "value_or_gas_sensitive_call",
    "dense_storage_or_call_ops",
    "revert_or_invalid_present",
    "panic_selector_present",
    "selfdestruct_present",
    "create_contract_present",
    "external_call_near_loop_candidate",
    "storage_write_near_loop_candidate",
]
PATTERN_TO_ID = {name: index for index, name in enumerate(EFPP_PATTERNS)}
NOISY_PATTERNS = {
    "external_call_near_loop_candidate",
    "storage_write_near_loop_candidate",
}

DEFAULT_PATTERN_PARAMS = {
    "call_state_window": 64,
    "return_check_window": 12,
    "auth_window": 128,
    "condition_window": 32,
    "arithmetic_window": 32,
    "arithmetic_min_count": 3,
    "address_call_window": 32,
    "dense_ops_window": 64,
    "dense_ops_min_count": 8,
    "loop_candidate_window": 48,
}

CALL_OPS = {"CALL", "DELEGATECALL", "STATICCALL", "CALLCODE"}
SENSITIVE_OPS = CALL_OPS | {"SELFDESTRUCT", "SUICIDE", "CREATE", "CREATE2"}
AUTH_OPS = {"CALLER", "ORIGIN"}
GUARD_OPS = {"JUMPI", "EQ", "LT", "GT", "SLT", "SGT", "ISZERO"}
RETURN_CHECK_OPS = {
    "ISZERO",
    "JUMPI",
    "REVERT",
    "RETURNDATASIZE",
    "RETURNDATACOPY",
}
ENV_OPS = {
    "TIMESTAMP",
    "NUMBER",
    "BLOCKHASH",
    "COINBASE",
    "DIFFICULTY",
    "PREVRANDAO",
    "GASPRICE",
    "BASEFEE",
}
CONDITION_ENV_OPS = {
    "TIMESTAMP",
    "NUMBER",
    "BLOCKHASH",
    "PREVRANDAO",
    "DIFFICULTY",
}
ARITHMETIC_OPS = {
    "ADD",
    "SUB",
    "MUL",
    "DIV",
    "SDIV",
    "MOD",
    "SMOD",
    "EXP",
    "ADDMOD",
    "MULMOD",
}
GAS_VALUE_OPS = {
    "GAS",
    "CALLVALUE",
    "BALANCE",
    "SELFBALANCE",
    "GASLIMIT",
    "GASPRICE",
    "BASEFEE",
}
HASH_OPS = {"SHA3", "KECCAK256"}
STORAGE_MEMORY_OPS = {
    "SLOAD",
    "SSTORE",
    "MLOAD",
    "MSTORE",
    "MSTORE8",
    "CALLDATACOPY",
    "CODECOPY",
    "RETURNDATACOPY",
}
ADDRESS_NEAR_OPS = CALL_OPS | {"BALANCE", "EXTCODESIZE", "EXTCODEHASH", "SSTORE"}
REVERT_OPS = {"REVERT", "INVALID"}
RELATION_TYPES = [
    "call_then_state_write",
    "state_write_then_call",
    "call_then_return_check",
    "state_write_then_auth_guard",
    "env_then_branch",
    "div_then_mul",
]


@dataclass(frozen=True)
class TokenUnit:
    token: str
    raw: str
    opcode: str
    is_operand: bool = False
    push_opcode: str = ""


def _is_hex(token):
    text = str(token).strip().lower()
    if text.startswith("0x"):
        text = text[2:]
    if not text:
        return False
    return all(char in "0123456789abcdef" for char in text)


def _canonical_hex(token):
    text = str(token).strip().lower()
    return text if text.startswith("0x") else f"0x{text}"


def build_token_units(opcode_sequence, tokenizer):
    """Tokenize an opcode sequence while retaining raw PUSH operands."""
    raw_tokens = str(opcode_sequence or "").strip().split()
    units = []
    index = 0
    while index < len(raw_tokens):
        raw = raw_tokens[index]
        opcode = raw.upper()
        units.append(TokenUnit(token=opcode, raw=raw, opcode=opcode))
        if opcode.startswith("PUSH") and index + 1 < len(raw_tokens):
            operand = raw_tokens[index + 1]
            if _is_hex(operand):
                normalized = tokenizer.tokenize(operand, add_special_tokens=False)
                token = normalized[0] if normalized else "[UNK]"
                units.append(
                    TokenUnit(
                        token=token,
                        raw=_canonical_hex(operand),
                        opcode="",
                        is_operand=True,
                        push_opcode=opcode,
                    )
                )
                index += 2
                continue
        index += 1
    return units


def _has_panic_selector(unit):
    return unit.is_operand and unit.raw.lower() == "0x4e487b71"


def _is_address_operand(unit):
    return unit.is_operand and (
        unit.push_opcode == "PUSH20" or unit.token == "<ADDR>"
    )


def _window_has(opcodes, start, end, candidates):
    start = max(0, start)
    end = min(len(opcodes), end)
    return any(opcodes[index] in candidates for index in range(start, end))


def _auth_guard_before(opcodes, index, window):
    start = max(0, index - window)
    auth_positions = [
        pos for pos in range(start, index) if opcodes[pos] in AUTH_OPS
    ]
    if not auth_positions:
        return False
    last_auth = auth_positions[-1]
    return _window_has(opcodes, last_auth + 1, index + 1, GUARD_OPS)


def annotate_effect_types(units, include_other_operands=False, return_check_window=12):
    opcodes = [unit.opcode for unit in units]
    effects = []
    checked_calls = set()
    for index, opcode in enumerate(opcodes):
        if opcode in CALL_OPS and _window_has(
            opcodes, index + 1, index + 1 + return_check_window, RETURN_CHECK_OPS
        ):
            checked_calls.add(index)

    for index, unit in enumerate(units):
        names = set()
        opcode = unit.opcode
        if unit.is_operand:
            if _is_address_operand(unit):
                names.add("AddressLiteral")
            if _has_panic_selector(unit):
                names.add("RevertOrAssert")
            participates = bool(names) or include_other_operands
        else:
            participates = True
            if opcode in CALL_OPS:
                names.add("ExternalCall")
            if opcode == "SLOAD":
                names.add("StateRead")
            if opcode == "SSTORE":
                names.add("StateWrite")
            if opcode in AUTH_OPS:
                names.add("AuthSource")
            if opcode in GUARD_OPS:
                names.add("ControlGuard")
            if opcode in ENV_OPS:
                names.add("EnvDependency")
            if opcode in ARITHMETIC_OPS:
                names.add("Arithmetic")
            if opcode in {"CREATE", "CREATE2"}:
                names.add("CreateContract")
            if opcode in {"SELFDESTRUCT", "SUICIDE"}:
                names.add("Destructive")
            if opcode in REVERT_OPS:
                names.add("RevertOrAssert")
            if opcode in GAS_VALUE_OPS:
                names.add("GasOrValue")
            if opcode in HASH_OPS:
                names.add("HashOrCrypto")
            if opcode in STORAGE_MEMORY_OPS:
                names.add("StorageOrMemoryHeavy")
            if index in checked_calls or (
                opcode in RETURN_CHECK_OPS
                and any(
                    opcodes[pos] in CALL_OPS
                    for pos in range(max(0, index - return_check_window), index)
                )
            ):
                names.add("ReturnCheck")
        if participates and not names:
            names.add("Normal")
        primary_name = next(
            (name for name in EFFECT_PRIORITY if name in names), "Normal"
        )
        primary_id = EFFECT_TO_ID[primary_name] if participates else -100
        multihot = [int(name in names) for name in EFFECT_TYPES]
        effects.append(
            {
                "primary_id": primary_id,
                "primary_name": primary_name if participates else None,
                "multihot": multihot,
                "loss_mask": int(participates),
            }
        )
    return effects


def _near_pair(opcodes, left_ops, right_ops, window):
    for index, opcode in enumerate(opcodes):
        if opcode in left_ops and _window_has(
            opcodes, index + 1, index + 1 + window, right_ops
        ):
            return index
    return None


def _dense_window(opcodes, candidates, window, minimum):
    count = sum(opcode in candidates for opcode in opcodes[:window])
    if count >= minimum:
        return 0
    for index in range(window, len(opcodes)):
        count += int(opcodes[index] in candidates)
        count -= int(opcodes[index - window] in candidates)
        if count >= minimum:
            return index - window + 1
    return None


def _snippet(units, center, radius=8):
    start = max(0, center - radius)
    end = min(len(units), center + radius + 1)
    return " ".join(unit.raw for unit in units[start:end])


def annotate_efpp_patterns(units, params=None):
    params = {**DEFAULT_PATTERN_PARAMS, **(params or {})}
    opcodes = [unit.opcode for unit in units]
    matches = {}

    def mark(name, index, rule):
        if name not in matches:
            matches[name] = {
                "matched_rule": rule,
                "token_index": int(max(0, index)),
                "local_opcode_snippet": _snippet(units, max(0, index)),
                "noisy": name in NOISY_PATTERNS,
            }

    for index, opcode in enumerate(opcodes):
        if opcode in CALL_OPS:
            mark("has_external_call", index, "CALL-like opcode is present")
        if opcode == "DELEGATECALL":
            mark("has_delegatecall", index, "DELEGATECALL is present")
        if opcode == "SLOAD":
            mark("has_state_read", index, "SLOAD is present")
        if opcode == "SSTORE":
            mark("has_state_write", index, "SSTORE is present")
        if opcode in REVERT_OPS:
            mark("revert_or_invalid_present", index, "REVERT or INVALID is present")
        if opcode in {"SELFDESTRUCT", "SUICIDE"}:
            mark("selfdestruct_present", index, "SELFDESTRUCT or SUICIDE is present")
        if opcode in {"CREATE", "CREATE2"}:
            mark("create_contract_present", index, "CREATE or CREATE2 is present")
        if _has_panic_selector(units[index]):
            mark("panic_selector_present", index, "Solidity panic selector 0x4e487b71")
        if _is_address_operand(units[index]):
            mark("has_hardcoded_address", index, "PUSH20 or normalized address operand")

    index = _near_pair(
        opcodes, CALL_OPS, {"SSTORE"}, params["call_state_window"]
    )
    if index is not None:
        mark("call_before_state_write", index, "CALL-like followed by SSTORE")
    index = _near_pair(
        opcodes, {"SSTORE"}, CALL_OPS, params["call_state_window"]
    )
    if index is not None:
        mark("state_write_before_call", index, "SSTORE followed by CALL-like")

    for index, opcode in enumerate(opcodes):
        if opcode in CALL_OPS:
            checked = _window_has(
                opcodes,
                index + 1,
                index + 1 + params["return_check_window"],
                RETURN_CHECK_OPS,
            )
            if checked:
                mark("call_with_return_check", index, "post-call check opcode in window")
            else:
                mark(
                    "call_without_return_check",
                    index,
                    "no post-call check opcode in the weak-rule window",
                )
            if _window_has(
                opcodes,
                index - params["condition_window"],
                index + params["condition_window"] + 1,
                GAS_VALUE_OPS,
            ):
                mark(
                    "value_or_gas_sensitive_call",
                    index,
                    "CALL-like near gas/value opcode",
                )

    for index, opcode in enumerate(opcodes):
        if opcode == "SSTORE":
            name = (
                "state_write_with_auth_guard"
                if _auth_guard_before(opcodes, index, params["auth_window"])
                else "state_write_without_auth_guard"
            )
            mark(name, index, "SSTORE with/without preceding auth-source guard")
        if opcode in SENSITIVE_OPS:
            name = (
                "sensitive_call_with_auth_guard"
                if _auth_guard_before(opcodes, index, params["auth_window"])
                else "sensitive_call_without_auth_guard"
            )
            mark(name, index, "sensitive opcode with/without preceding auth guard")

    for index, opcode in enumerate(opcodes):
        if opcode in CONDITION_ENV_OPS:
            if _window_has(
                opcodes,
                index + 1,
                index + 1 + params["condition_window"],
                GUARD_OPS,
            ):
                mark("env_used_in_condition", index, "environment opcode followed by guard")
            if _window_has(
                opcodes,
                index + 1,
                index + 1 + params["condition_window"],
                ARITHMETIC_OPS | HASH_OPS,
            ):
                mark(
                    "env_used_in_arithmetic_or_hash",
                    index,
                    "environment opcode followed by arithmetic/hash",
                )

    index = _dense_window(
        opcodes,
        ARITHMETIC_OPS,
        params["arithmetic_window"],
        params["arithmetic_min_count"],
    )
    if index is not None:
        mark("arithmetic_chain", index, "arithmetic opcode count reaches threshold")
    index = _near_pair(
        opcodes, {"DIV", "SDIV"}, {"MUL"}, params["arithmetic_window"]
    )
    if index is not None:
        mark("div_before_mul", index, "DIV/SDIV followed by MUL")
    index = _near_pair(
        opcodes,
        ARITHMETIC_OPS,
        GUARD_OPS | REVERT_OPS,
        params["arithmetic_window"],
    )
    if index is not None:
        mark(
            "arithmetic_followed_by_guard_or_revert",
            index,
            "arithmetic followed by guard/revert",
        )

    for index, unit in enumerate(units):
        if _is_address_operand(unit) and _window_has(
            opcodes,
            index - params["address_call_window"],
            index + params["address_call_window"] + 1,
            ADDRESS_NEAR_OPS,
        ):
            mark(
                "hardcoded_address_near_call",
                index,
                "address literal near call/address-sensitive opcode",
            )

    index = _dense_window(
        opcodes,
        {"SLOAD", "SSTORE"} | CALL_OPS,
        params["dense_ops_window"],
        params["dense_ops_min_count"],
    )
    if index is not None:
        mark("dense_storage_or_call_ops", index, "dense storage/call weak rule")

    for index, opcode in enumerate(opcodes):
        if opcode not in CALL_OPS | {"SSTORE"}:
            continue
        start = max(0, index - params["loop_candidate_window"])
        end = min(len(opcodes), index + params["loop_candidate_window"] + 1)
        window = opcodes[start:end]
        if "JUMPDEST" in window and "JUMPI" in window:
            name = (
                "external_call_near_loop_candidate"
                if opcode in CALL_OPS
                else "storage_write_near_loop_candidate"
            )
            mark(name, index, "weak JUMPDEST+JUMPI loop-candidate heuristic")

    labels = [int(name in matches) for name in EFPP_PATTERNS]
    return labels, matches


def annotate_effect_relations(units, params=None):
    params = {**DEFAULT_PATTERN_PARAMS, **(params or {})}
    opcodes = [unit.opcode for unit in units]
    matches = {}

    def mark(name, index, rule):
        if name not in matches:
            matches[name] = {
                "matched_rule": rule,
                "token_index": int(max(0, index)),
                "local_opcode_snippet": _snippet(units, max(0, index)),
            }

    index = _near_pair(opcodes, CALL_OPS, {"SSTORE"}, params["call_state_window"])
    if index is not None:
        mark("call_then_state_write", index, "CALL-like followed by SSTORE")
    index = _near_pair(opcodes, {"SSTORE"}, CALL_OPS, params["call_state_window"])
    if index is not None:
        mark("state_write_then_call", index, "SSTORE followed by CALL-like")

    for index, opcode in enumerate(opcodes):
        if opcode in CALL_OPS and _window_has(
            opcodes,
            index + 1,
            index + 1 + params["return_check_window"],
            RETURN_CHECK_OPS,
        ):
            mark("call_then_return_check", index, "CALL-like followed by weak return-check window")
        if opcode == "SSTORE" and _auth_guard_before(opcodes, index, params["auth_window"]):
            mark("state_write_then_auth_guard", index, "SSTORE has preceding auth-source guard")
        if opcode in CONDITION_ENV_OPS and _window_has(
            opcodes,
            index + 1,
            index + 1 + params["condition_window"],
            GUARD_OPS,
        ):
            mark("env_then_branch", index, "environment opcode followed by branch/guard")
        if opcode in {"DIV", "SDIV"} and _window_has(
            opcodes,
            index + 1,
            index + 1 + params["arithmetic_window"],
            {"MUL"},
        ):
            mark("div_then_mul", index, "DIV/SDIV followed by MUL")

    labels = [int(name in matches) for name in RELATION_TYPES]
    primary_name = next((name for name in RELATION_TYPES if name in matches), None)
    primary_id = RELATION_TYPES.index(primary_name) if primary_name is not None else -100
    return labels, matches, primary_id


def effect_events(units, effects, offset=0):
    events = []
    for index, (unit, effect) in enumerate(zip(units, effects)):
        if not effect["loss_mask"] or effect["primary_name"] == "Normal":
            continue
        events.append(
            {
                "token_index": index + offset,
                "opcode": unit.raw,
                "effect_type": effect["primary_name"],
                "effect_type_id": effect["primary_id"],
            }
        )
    return events


def effect_type_token_names(effects):
    names = []
    for effect in effects:
        names.append(effect["primary_name"] if effect["loss_mask"] else None)
    return names


def effect_type_chunk_histogram(effects):
    histogram = [0] * len(EFFECT_TYPES)
    active = 0
    for effect in effects:
        if not effect["loss_mask"]:
            continue
        active += 1
        histogram[effect["primary_id"]] += 1
    if active == 0:
        return [0.0] * len(EFFECT_TYPES)
    return [count / active for count in histogram]


def relation_events(units, relation_matches, offset=0):
    events = []
    for relation_name, match in relation_matches.items():
        events.append(
            {
                "token_index": int(match["token_index"]) + offset,
                "relation_type": relation_name,
                "relation_type_id": RELATION_TYPES.index(relation_name),
                "opcode_snippet": match["local_opcode_snippet"],
            }
        )
    return events


def pattern_metadata():
    return [
        {
            "id": index,
            "name": name,
            "noise_status": "noisy_but_candidate" if name in NOISY_PATTERNS else "rule_based",
        }
        for index, name in enumerate(EFPP_PATTERNS)
    ]
