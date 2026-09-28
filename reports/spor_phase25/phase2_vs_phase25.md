# SPOR Phase 2 vs Phase 2.5

Same strict criterion: a token is valid only when its instruction has a supported and known provenance state; PUSH operand aliases are excluded. Train and valid only; test locked.

| Split | Phase | Strict valid tokens | Total tokens | Coverage |
|---|---|---:|---:|---:|
| train | Phase 2 | 37,848,938 | 98,679,197 | 38.36% |
| train | Phase 2.5 | 50,861,410 | 98,679,197 | 51.54% |
| valid | Phase 2 | 4,706,788 | 12,303,020 | 38.26% |
| valid | Phase 2.5 | 6,348,792 | 12,303,020 | 51.60% |

| Split | Opcode | Phase 2 known/total | Phase 2.5 known/total | Phase 2 known % | Phase 2.5 known % |
|---|---|---:|---:|---:|---:|
| train | JUMPI | 613,612/2,185,113 | 1,281,129/2,185,113 | 28.08% | 58.63% |
| train | CALL | 803/57,236 | 15,817/57,236 | 1.40% | 27.63% |
| train | SSTORE | 67,342/360,150 | 102,216/360,150 | 18.70% | 28.38% |
| train | SLOAD | 666,705/1,148,155 | 752,757/1,148,155 | 58.07% | 65.56% |
| valid | JUMPI | 77,081/273,478 | 161,618/273,478 | 28.19% | 59.10% |
| valid | CALL | 98/7,007 | 1,921/7,007 | 1.40% | 27.42% |
| valid | SSTORE | 8,683/45,749 | 13,187/45,749 | 18.98% | 28.82% |
| valid | SLOAD | 84,707/145,103 | 95,805/145,103 | 58.38% | 66.03% |

## CFG and convergence

| Split | Static jump edges | Fall-through edges | Unresolved jumps / branch ops | Height-conflict blocks | Mean worklist visits per block | Nonconverged blocks |
|---|---:|---:|---:|---:|---:|---:|
| train | 4,735,862 | 2,667,583 | 754,726/5,490,444 (13.75%) | 497,995 | 1.1834 | 621 |
| valid | 590,178 | 333,358 | 93,428/683,589 (13.67%) | 61,394 | 1.1768 | 50 |

| Split | Instructions | Block-entry unknown count / rate | Analysis-failure count / rate | Parser failures | Token-alignment failures |
|---|---:|---:|---:|---:|---:|
| train | 77,195,099 | 0 (0.00%) | 12,982,354 (16.82%) | 313 | 0 |
| valid | 9,621,811 | 81 (0.00%) | 1,618,630 (16.82%) | 33 | 0 |