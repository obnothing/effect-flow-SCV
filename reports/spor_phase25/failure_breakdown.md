# SPOR Phase 2.5 Failure Breakdown

Dataset: `DIVE_8_opcode_random_split`; train/valid only. `test_checked=false`.
Percentages below are shown both against all model tokens and against tokens failing the strict known-source criterion.

## train

Total tokens: 98,679,197; invalid tokens: 47,817,787 (48.46%).

| Cause | Absolute count | % all tokens | % invalid tokens |
|---|---:|---:|---:|
| operand_alias | 21,484,098 | 21.77% | 44.93% |
| dynamic_jump_or_unresolved_cfg | 8,236,594 | 8.35% | 17.22% |
| abstract_stack_underflow | 5,617,875 | 5.69% | 11.75% |
| not_applicable | 5,114,741 | 5.18% | 10.70% |
| known_stack_effect_but_unknown_provenance | 4,244,179 | 4.30% | 8.88% |
| predecessor_stack_height_conflict | 2,880,342 | 2.92% | 6.02% |
| previous_analysis_failure | 138,519 | 0.14% | 0.29% |
| unsupported_opcode_stack_effect | 95,803 | 0.10% | 0.20% |
| other | 5,323 | 0.01% | 0.01% |
| parser_failure | 313 | 0.00% | 0.00% |
| basic_block_entry_unknown | 0 | 0.00% | 0.00% |
| token_alignment_failure | 0 | 0.00% | 0.00% |

Target instruction causes:

| Opcode | Cause | Count | % of this opcode |
|---|---|---:|---:|
| JUMPI | known | 1,281,129 | 58.63% |
| JUMPI | dynamic_jump_or_unresolved_cfg | 276,235 | 12.64% |
| JUMPI | abstract_stack_underflow | 222,601 | 10.19% |
| JUMPI | known_stack_effect_but_unknown_provenance | 205,048 | 9.38% |
| JUMPI | predecessor_stack_height_conflict | 199,327 | 9.12% |
| JUMPI | previous_analysis_failure | 552 | 0.03% |
| JUMPI | other | 221 | 0.01% |
| CALL | dynamic_jump_or_unresolved_cfg | 23,003 | 40.19% |
| CALL | known | 15,817 | 27.63% |
| CALL | known_stack_effect_but_unknown_provenance | 8,485 | 14.82% |
| CALL | abstract_stack_underflow | 6,371 | 11.13% |
| CALL | predecessor_stack_height_conflict | 2,574 | 4.50% |
| CALL | previous_analysis_failure | 986 | 1.72% |
| SSTORE | known_stack_effect_but_unknown_provenance | 140,865 | 39.11% |
| SSTORE | known | 102,216 | 28.38% |
| SSTORE | dynamic_jump_or_unresolved_cfg | 59,896 | 16.63% |
| SSTORE | abstract_stack_underflow | 53,687 | 14.91% |
| SSTORE | predecessor_stack_height_conflict | 2,694 | 0.75% |
| SSTORE | previous_analysis_failure | 789 | 0.22% |
| SSTORE | other | 3 | 0.00% |
| SLOAD | known | 752,757 | 65.56% |
| SLOAD | known_stack_effect_but_unknown_provenance | 333,704 | 29.06% |
| SLOAD | abstract_stack_underflow | 22,023 | 1.92% |
| SLOAD | predecessor_stack_height_conflict | 21,526 | 1.87% |
| SLOAD | dynamic_jump_or_unresolved_cfg | 17,192 | 1.50% |
| SLOAD | previous_analysis_failure | 932 | 0.08% |
| SLOAD | other | 21 | 0.00% |

## valid

Total tokens: 12,303,020; invalid tokens: 5,954,228 (48.40%).

| Cause | Absolute count | % all tokens | % invalid tokens |
|---|---:|---:|---:|
| operand_alias | 2,681,209 | 21.79% | 45.03% |
| dynamic_jump_or_unresolved_cfg | 1,018,788 | 8.28% | 17.11% |
| abstract_stack_underflow | 697,785 | 5.67% | 11.72% |
| not_applicable | 635,520 | 5.17% | 10.67% |
| known_stack_effect_but_unknown_provenance | 522,996 | 4.25% | 8.78% |
| predecessor_stack_height_conflict | 355,910 | 2.89% | 5.98% |
| previous_analysis_failure | 28,022 | 0.23% | 0.47% |
| unsupported_opcode_stack_effect | 13,486 | 0.11% | 0.23% |
| other | 398 | 0.00% | 0.01% |
| basic_block_entry_unknown | 81 | 0.00% | 0.00% |
| parser_failure | 33 | 0.00% | 0.00% |
| token_alignment_failure | 0 | 0.00% | 0.00% |

Target instruction causes:

| Opcode | Cause | Count | % of this opcode |
|---|---|---:|---:|
| JUMPI | known | 161,618 | 59.10% |
| JUMPI | dynamic_jump_or_unresolved_cfg | 33,992 | 12.43% |
| JUMPI | abstract_stack_underflow | 27,338 | 10.00% |
| JUMPI | known_stack_effect_but_unknown_provenance | 25,614 | 9.37% |
| JUMPI | predecessor_stack_height_conflict | 24,789 | 9.06% |
| JUMPI | previous_analysis_failure | 102 | 0.04% |
| JUMPI | other | 17 | 0.01% |
| JUMPI | basic_block_entry_unknown | 8 | 0.00% |
| CALL | dynamic_jump_or_unresolved_cfg | 2,794 | 39.87% |
| CALL | known | 1,921 | 27.42% |
| CALL | known_stack_effect_but_unknown_provenance | 1,071 | 15.28% |
| CALL | abstract_stack_underflow | 770 | 10.99% |
| CALL | predecessor_stack_height_conflict | 320 | 4.57% |
| CALL | previous_analysis_failure | 131 | 1.87% |
| SSTORE | known_stack_effect_but_unknown_provenance | 17,617 | 38.51% |
| SSTORE | known | 13,187 | 28.82% |
| SSTORE | dynamic_jump_or_unresolved_cfg | 7,853 | 17.17% |
| SSTORE | abstract_stack_underflow | 6,661 | 14.56% |
| SSTORE | predecessor_stack_height_conflict | 286 | 0.63% |
| SSTORE | previous_analysis_failure | 144 | 0.31% |
| SSTORE | basic_block_entry_unknown | 1 | 0.00% |
| SLOAD | known | 95,805 | 66.03% |
| SLOAD | known_stack_effect_but_unknown_provenance | 41,508 | 28.61% |
| SLOAD | predecessor_stack_height_conflict | 2,801 | 1.93% |
| SLOAD | abstract_stack_underflow | 2,575 | 1.77% |
| SLOAD | dynamic_jump_or_unresolved_cfg | 2,288 | 1.58% |
| SLOAD | previous_analysis_failure | 122 | 0.08% |
| SLOAD | basic_block_entry_unknown | 4 | 0.00% |
