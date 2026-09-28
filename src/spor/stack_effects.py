"""Verified EVM stack effects used by the bounded SPOR analysis.

Counts follow the Ethereum Yellow Paper opcode table and go-ethereum's jump
table. They describe stack height only, not an instruction's value semantics.
"""

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class StackEffect:
    pop_count: int
    push_count: int
    min_stack: int = 0


EFFECTS = {}


def _add(names, pops, pushes):
    for name in names.split():
        EFFECTS[name] = StackEffect(pops, pushes, pops)


_add("STOP JUMPDEST INVALID UNKNOWN_0xFE", 0, 0)
_add("ADD MUL SUB DIV SDIV MOD SMOD EXP SIGNEXTEND LT GT SLT SGT EQ AND OR XOR BYTE SHL SHR SAR", 2, 1)
_add("ISZERO NOT", 1, 1)
_add("ADDMOD MULMOD", 3, 1)
_add("SHA3 KECCAK256", 2, 1)
_add("ADDRESS ORIGIN CALLER CALLVALUE CALLDATASIZE CODESIZE GASPRICE RETURNDATASIZE COINBASE TIMESTAMP NUMBER DIFFICULTY PREVRANDAO GASLIMIT CHAINID SELFBALANCE BASEFEE BLOBBASEFEE PC MSIZE GAS", 0, 1)
_add("BALANCE CALLDATALOAD EXTCODESIZE EXTCODEHASH BLOCKHASH BLOBHASH MLOAD SLOAD TLOAD", 1, 1)
_add("CALLDATACOPY CODECOPY RETURNDATACOPY MCOPY", 3, 0)
_add("EXTCODECOPY", 4, 0)
_add("MSTORE MSTORE8 SSTORE TSTORE", 2, 0)
_add("POP JUMP SELFDESTRUCT", 1, 0)
_add("JUMPI RETURN REVERT", 2, 0)
_add("CREATE", 3, 1)
_add("CREATE2", 4, 1)
_add("CALL CALLCODE", 7, 1)
_add("DELEGATECALL STATICCALL", 6, 1)
for count in range(5):
    EFFECTS[f"LOG{count}"] = StackEffect(count + 2, 0, count + 2)
for count in range(17):
    EFFECTS[f"PUSH{count}"] = StackEffect(0, 1)
for count in range(17, 33):
    EFFECTS[f"PUSH{count}"] = StackEffect(0, 1)
for count in range(1, 17):
    EFFECTS[f"DUP{count}"] = StackEffect(0, 1, count)
    EFFECTS[f"SWAP{count}"] = StackEffect(0, 0, count + 1)


def stack_effect(opcode):
    return EFFECTS.get(opcode)
