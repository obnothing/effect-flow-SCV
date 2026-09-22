# SPOR Phase 2 Feature Quality

DIVE_8 train/valid only; test_checked=false.

{
  "dataset": "data\\processed\\DIVE_8_opcode_random_split",
  "label_names": [
    "Reentrancy",
    "Access Control",
    "Arithmetic",
    "Unchecked Return Values",
    "DoS",
    "Bad Randomness",
    "Front Running",
    "Time manipulation"
  ],
  "provenance_categories": [
    "CONSTANT",
    "CALLER",
    "CALLVALUE",
    "CALLDATA",
    "TIMESTAMP",
    "BLOCK_ENV",
    "STORAGE",
    "MEMORY",
    "CALL_RESULT",
    "ARITHMETIC",
    "COMPARISON",
    "OTHER",
    "UNKNOWN"
  ],
  "role_names": [
    "OTHER",
    "CONSTANT",
    "STACK_POP",
    "STACK_DUP",
    "STACK_SWAP",
    "ARITHMETIC",
    "COMPARISON",
    "CALLER",
    "CALLVALUE",
    "CALLDATA",
    "BLOCK_ENV",
    "MEMORY",
    "STORAGE",
    "CALL",
    "CONTROL_FLOW",
    "PUSH_IMMEDIATE"
  ],
  "status_names": [
    "known",
    "unknown_entry",
    "analysis_failure",
    "analysis_failure_parse_status",
    "operand_alias",
    "parse_error"
  ],
  "splits": {
    "train": {
      "contracts": 17864,
      "instruction_count": {
        "min": 10,
        "mean": 4343.770096283028,
        "max": 18619
      },
      "model_token_count": {
        "min": 14,
        "mean": 5523.913849081952,
        "max": 30982
      },
      "basic_block_count": {
        "min": 2,
        "mean": 402.90623600537396,
        "max": 2050
      },
      "counters": {
        "unknown_opcode_annotation": 159359,
        "known": 65578226,
        "unknown_entry": 2577785,
        "analysis_failure_unsupported_stack_effect": 2554413,
        "analysis_failure_stack_underflow": 5294317,
        "analysis_failure": 1592055,
        "CONSTANT": 21324739,
        "MEMORY": 4083226,
        "CALLVALUE": 405846,
        "STACK_DUP": 11892043,
        "COMPARISON": 3170792,
        "CONTROL_FLOW": 11624024,
        "STACK_POP": 4419909,
        "OTHER": 2554413,
        "CALLDATA": 276492,
        "ARITHMETIC": 7990613,
        "STACK_SWAP": 7964676,
        "CALLER": 240503,
        "STORAGE": 1508305,
        "BLOCK_ENV": 59954,
        "target_JUMPI": 2185113,
        "target_SSTORE": 360150,
        "target_SLOAD": 1148155,
        "CALL": 81574,
        "target_CALL": 57236,
        "push_immediate_width_mismatch": 313,
        "analysis_failure_parse_status": 313
      },
      "mean_seconds_per_contract": 0.04188038624055524,
      "total_seconds": 766.7923832999659
    },
    "valid": {
      "contracts": 2233,
      "instruction_count": {
        "min": 18,
        "mean": 4332.105687416032,
        "max": 19232
      },
      "model_token_count": {
        "min": 27,
        "mean": 5509.637259292432,
        "max": 24615
      },
      "basic_block_count": {
        "min": 2,
        "mean": 401.44379758172863,
        "max": 1611
      },
      "counters": {
        "unknown_opcode_annotation": 21295,
        "push_immediate_width_mismatch": 33,
        "known": 8174835,
        "analysis_failure_unsupported_stack_effect": 327585,
        "unknown_entry": 319644,
        "analysis_failure_stack_underflow": 655136,
        "analysis_failure": 196359,
        "analysis_failure_parse_status": 33,
        "CONSTANT": 2659914,
        "MEMORY": 507426,
        "OTHER": 327585,
        "COMPARISON": 395453,
        "CONTROL_FLOW": 1446689,
        "CALLDATA": 35373,
        "ARITHMETIC": 992680,
        "STACK_DUP": 1478592,
        "CALLVALUE": 52056,
        "STACK_POP": 547169,
        "STACK_SWAP": 990380,
        "STORAGE": 190852,
        "CALLER": 30851,
        "BLOCK_ENV": 8670,
        "CALL": 9902,
        "target_JUMPI": 273478,
        "target_SLOAD": 145103,
        "target_SSTORE": 45749,
        "target_CALL": 7007
      },
      "mean_seconds_per_contract": 0.0324865879528873,
      "total_seconds": 74.34522670001024
    }
  },
  "runtime_bytecode_available": false,
  "test_checked": false,
  "scope": "disassembled-opcode-based basic-block-local reconstruction; no cross-block execution claim"
}
