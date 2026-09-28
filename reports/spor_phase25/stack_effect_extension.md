# SPOR Phase 2.5 Stack-Effect Extension

The registry records only verified pop/push counts. An opcode with known stack effect but unmodeled value semantics consumes its inputs and pushes an explicit unknown value. Unknown byte annotations without a verified effect remain unsupported.

Stack effects were checked against the [Ethereum opcode reference](https://ethereum.org/developers/docs/evm/opcodes), [go-ethereum jump table](https://github.com/ethereum/go-ethereum/blob/master/core/vm/jump_table.go), and [Yellow Paper](https://ethereum.github.io/yellowpaper/paper.pdf).

| Opcode | Train count | Valid count | Pop | Push | Minimum stack |
|---|---:|---:|---:|---:|---:|
| ADD | 3,026,456 | 375,669 | 2 | 1 | 2 |
| ADDMOD | 4,913 | 618 | 3 | 1 | 3 |
| ADDRESS | 84,246 | 11,496 | 0 | 1 | 0 |
| AND | 2,040,973 | 254,912 | 2 | 1 | 2 |
| BALANCE | 5,863 | 2,048 | 1 | 1 | 1 |
| BASEFEE | 718 | 100 | 0 | 1 | 0 |
| BLOCKHASH | 927 | 112 | 1 | 1 | 1 |
| BYTE | 2,661 | 361 | 2 | 1 | 2 |
| CALL | 57,236 | 7,007 | 7 | 1 | 7 |
| CALLCODE | 1,111 | 117 | 7 | 1 | 7 |
| CALLDATACOPY | 21,346 | 3,482 | 3 | 0 | 3 |
| CALLDATALOAD | 276,492 | 35,373 | 1 | 1 | 1 |
| CALLDATASIZE | 211,721 | 26,889 | 0 | 1 | 0 |
| CALLER | 240,503 | 30,851 | 0 | 1 | 0 |
| CALLVALUE | 405,846 | 52,056 | 0 | 1 | 0 |
| CHAINID | 1,341 | 189 | 0 | 1 | 0 |
| CODECOPY | 22,452 | 3,374 | 3 | 0 | 3 |
| CODESIZE | 2,283 | 829 | 0 | 1 | 0 |
| COINBASE | 1,685 | 1,064 | 0 | 1 | 0 |
| CREATE | 766 | 93 | 3 | 1 | 3 |
| CREATE2 | 1,662 | 206 | 4 | 1 | 4 |
| DELEGATECALL | 1,281 | 187 | 6 | 1 | 6 |
| DIFFICULTY | 1,084 | 172 | 0 | 1 | 0 |
| DIV | 289,773 | 36,071 | 2 | 1 | 2 |
| DUP1 | 3,724,522 | 466,050 | 0 | 1 | 1 |
| DUP10 | 46,553 | 5,653 | 0 | 1 | 10 |
| DUP11 | 28,059 | 3,330 | 0 | 1 | 11 |
| DUP12 | 18,301 | 2,105 | 0 | 1 | 12 |
| DUP13 | 10,455 | 1,174 | 0 | 1 | 13 |
| DUP14 | 6,961 | 876 | 0 | 1 | 14 |
| DUP15 | 5,396 | 726 | 0 | 1 | 15 |
| DUP16 | 3,528 | 422 | 0 | 1 | 16 |
| DUP2 | 3,311,802 | 412,820 | 0 | 1 | 2 |
| DUP3 | 1,951,056 | 241,764 | 0 | 1 | 3 |
| DUP4 | 1,139,882 | 141,622 | 0 | 1 | 4 |
| DUP5 | 661,806 | 81,554 | 0 | 1 | 5 |
| DUP6 | 442,230 | 54,261 | 0 | 1 | 6 |
| DUP7 | 256,412 | 31,530 | 0 | 1 | 7 |
| DUP8 | 181,006 | 22,114 | 0 | 1 | 8 |
| DUP9 | 104,074 | 12,591 | 0 | 1 | 9 |
| EQ | 724,953 | 90,649 | 2 | 1 | 2 |
| EXP | 356,371 | 44,923 | 2 | 1 | 2 |
| EXTCODECOPY | 638 | 84 | 4 | 0 | 4 |
| EXTCODEHASH | 720 | 81 | 1 | 1 | 1 |
| EXTCODESIZE | 37,200 | 4,489 | 1 | 1 | 1 |
| GAS | 66,397 | 8,115 | 0 | 1 | 0 |
| GASLIMIT | 6,469 | 862 | 0 | 1 | 0 |
| GASPRICE | 6,813 | 895 | 0 | 1 | 0 |
| GT | 287,984 | 36,424 | 2 | 1 | 2 |
| ISZERO | 1,782,230 | 221,528 | 1 | 1 | 1 |
| JUMP | 3,305,331 | 410,111 | 1 | 0 | 1 |
| JUMPDEST | 4,972,259 | 617,894 | 0 | 0 | 0 |
| JUMPI | 2,185,113 | 273,478 | 2 | 0 | 2 |
| LOG0 | 699 | 69 | 2 | 0 | 2 |
| LOG1 | 16,571 | 2,134 | 3 | 0 | 3 |
| LOG2 | 16,402 | 2,006 | 4 | 0 | 4 |
| LOG3 | 61,968 | 7,920 | 5 | 0 | 5 |
| LOG4 | 6,796 | 772 | 6 | 0 | 6 |
| LT | 266,120 | 33,360 | 2 | 1 | 2 |
| MLOAD | 1,413,520 | 175,063 | 1 | 1 | 1 |
| MOD | 3,765 | 431 | 2 | 1 | 2 |
| MSIZE | 1,881 | 256 | 0 | 1 | 0 |
| MSTORE | 2,669,706 | 332,363 | 2 | 0 | 2 |
| MSTORE8 | 3,007 | 419 | 2 | 0 | 2 |
| MUL | 278,064 | 34,245 | 2 | 1 | 2 |
| MULMOD | 1,045 | 148 | 3 | 1 | 3 |
| NOT | 235,639 | 28,672 | 1 | 1 | 1 |
| NUMBER | 16,865 | 2,225 | 0 | 1 | 0 |
| OR | 120,611 | 14,754 | 2 | 1 | 2 |
| ORIGIN | 18,523 | 2,368 | 0 | 1 | 0 |
| PC | 1,419 | 174 | 0 | 1 | 0 |
| POP | 4,419,909 | 547,169 | 1 | 0 | 1 |
| PUSH0 | 402,010 | 51,781 | 0 | 1 | 0 |
| PUSH1 | 11,569,676 | 1,441,803 | 0 | 1 | 0 |
| PUSH10 | 6,378 | 812 | 0 | 1 | 0 |
| PUSH11 | 2,929 | 372 | 0 | 1 | 0 |
| PUSH12 | 4,338 | 650 | 0 | 1 | 0 |
| PUSH13 | 3,596 | 509 | 0 | 1 | 0 |
| PUSH14 | 4,981 | 620 | 0 | 1 | 0 |
| PUSH15 | 4,169 | 592 | 0 | 1 | 0 |
| PUSH16 | 6,003 | 778 | 0 | 1 | 0 |
| PUSH17 | 3,664 | 364 | 0 | 1 | 0 |
| PUSH18 | 1,659 | 228 | 0 | 1 | 0 |
| PUSH19 | 6,315 | 892 | 0 | 1 | 0 |
| PUSH2 | 6,997,279 | 871,301 | 0 | 1 | 0 |
| PUSH20 | 774,761 | 96,848 | 0 | 1 | 0 |
| PUSH21 | 11,157 | 1,521 | 0 | 1 | 0 |
| PUSH22 | 4,663 | 657 | 0 | 1 | 0 |
| PUSH23 | 1,059 | 169 | 0 | 1 | 0 |
| PUSH24 | 1,495 | 276 | 0 | 1 | 0 |
| PUSH25 | 1,238 | 199 | 0 | 1 | 0 |
| PUSH26 | 636 | 144 | 0 | 1 | 0 |
| PUSH27 | 1,360 | 249 | 0 | 1 | 0 |
| PUSH28 | 5,334 | 585 | 0 | 1 | 0 |
| PUSH29 | 14,510 | 1,736 | 0 | 1 | 0 |
| PUSH3 | 208,516 | 25,492 | 0 | 1 | 0 |
| PUSH30 | 371 | 59 | 0 | 1 | 0 |
| PUSH31 | 1,251 | 181 | 0 | 1 | 0 |
| PUSH32 | 411,069 | 51,501 | 0 | 1 | 0 |
| PUSH4 | 765,372 | 96,130 | 0 | 1 | 0 |
| PUSH5 | 25,854 | 3,320 | 0 | 1 | 0 |
| PUSH6 | 11,577 | 1,513 | 0 | 1 | 0 |
| PUSH7 | 9,016 | 1,277 | 0 | 1 | 0 |
| PUSH8 | 50,302 | 5,972 | 0 | 1 | 0 |
| PUSH9 | 12,201 | 1,383 | 0 | 1 | 0 |
| RETURN | 142,854 | 17,798 | 2 | 0 | 2 |
| RETURNDATACOPY | 68,767 | 8,285 | 3 | 0 | 3 |
| RETURNDATASIZE | 188,924 | 22,751 | 0 | 1 | 0 |
| REVERT | 930,680 | 116,515 | 2 | 0 | 2 |
| SAR | 588 | 85 | 2 | 1 | 2 |
| SDIV | 1,205 | 135 | 2 | 1 | 2 |
| SELFBALANCE | 22,874 | 2,948 | 0 | 1 | 0 |
| SELFDESTRUCT | 1,551 | 182 | 1 | 0 | 1 |
| SGT | 3,033 | 340 | 2 | 1 | 2 |
| SHA3 | 526,595 | 68,210 | 2 | 1 | 2 |
| SHL | 825,491 | 101,540 | 2 | 1 | 2 |
| SHR | 25,035 | 3,014 | 2 | 1 | 2 |
| SIGNEXTEND | 1,840 | 160 | 2 | 1 | 2 |
| SLOAD | 1,148,155 | 145,103 | 1 | 1 | 1 |
| SLT | 106,472 | 13,152 | 2 | 1 | 2 |
| SMOD | 994 | 123 | 2 | 1 | 2 |
| SSTORE | 360,150 | 45,749 | 2 | 0 | 2 |
| STATICCALL | 21,946 | 2,591 | 6 | 1 | 6 |
| STOP | 86,236 | 10,711 | 0 | 0 | 0 |
| SUB | 1,370,225 | 170,741 | 2 | 1 | 2 |
| SWAP1 | 4,635,534 | 577,803 | 0 | 0 | 2 |
| SWAP10 | 7,223 | 848 | 0 | 0 | 11 |
| SWAP11 | 5,542 | 615 | 0 | 0 | 12 |
| SWAP12 | 5,319 | 628 | 0 | 0 | 13 |
| SWAP13 | 6,223 | 777 | 0 | 0 | 14 |
| SWAP14 | 1,324 | 110 | 0 | 0 | 15 |
| SWAP15 | 2,318 | 275 | 0 | 0 | 16 |
| SWAP16 | 672 | 74 | 0 | 0 | 17 |
| SWAP2 | 2,056,206 | 256,586 | 0 | 0 | 3 |
| SWAP3 | 715,572 | 88,449 | 0 | 0 | 4 |
| SWAP4 | 268,732 | 32,860 | 0 | 0 | 5 |
| SWAP5 | 124,231 | 15,091 | 0 | 0 | 6 |
| SWAP6 | 65,676 | 8,053 | 0 | 0 | 7 |
| SWAP7 | 38,049 | 4,524 | 0 | 0 | 8 |
| SWAP8 | 18,537 | 2,080 | 0 | 0 | 9 |
| SWAP9 | 13,518 | 1,607 | 0 | 0 | 10 |
| TIMESTAMP | 30,865 | 3,946 | 0 | 1 | 0 |
| UNKNOWN_0x0C | 767 | 81 | unsupported | unsupported | unknown |
| UNKNOWN_0x0D | 612 | 71 | unsupported | unsupported | unknown |
| UNKNOWN_0x0E | 600 | 68 | unsupported | unsupported | unknown |
| UNKNOWN_0x0F | 668 | 85 | unsupported | unsupported | unknown |
| UNKNOWN_0x1E | 532 | 65 | unsupported | unsupported | unknown |
| UNKNOWN_0x1F | 617 | 73 | unsupported | unsupported | unknown |
| UNKNOWN_0x21 | 551 | 69 | unsupported | unsupported | unknown |
| UNKNOWN_0x22 | 6,388 | 792 | unsupported | unsupported | unknown |
| UNKNOWN_0x23 | 2,219 | 126 | unsupported | unsupported | unknown |
| UNKNOWN_0x24 | 566 | 65 | unsupported | unsupported | unknown |
| UNKNOWN_0x25 | 3,176 | 66 | unsupported | unsupported | unknown |
| UNKNOWN_0x26 | 516 | 59 | unsupported | unsupported | unknown |
| UNKNOWN_0x27 | 515 | 68 | unsupported | unsupported | unknown |
| UNKNOWN_0x28 | 619 | 77 | unsupported | unsupported | unknown |
| UNKNOWN_0x29 | 534 | 77 | unsupported | unsupported | unknown |
| UNKNOWN_0x2A | 554 | 60 | unsupported | unsupported | unknown |
| UNKNOWN_0x2B | 887 | 103 | unsupported | unsupported | unknown |
| UNKNOWN_0x2C | 582 | 80 | unsupported | unsupported | unknown |
| UNKNOWN_0x2D | 2,169 | 1,052 | unsupported | unsupported | unknown |
| UNKNOWN_0x2E | 12,529 | 2,822 | unsupported | unsupported | unknown |
| UNKNOWN_0x2F | 1,039 | 129 | unsupported | unsupported | unknown |
| UNKNOWN_0x49 | 1,051 | 174 | unsupported | unsupported | unknown |
| UNKNOWN_0x4A | 624 | 145 | unsupported | unsupported | unknown |
| UNKNOWN_0x4B | 666 | 153 | unsupported | unsupported | unknown |
| UNKNOWN_0x4C | 787 | 139 | unsupported | unsupported | unknown |
| UNKNOWN_0x4D | 1,416 | 246 | unsupported | unsupported | unknown |
| UNKNOWN_0x4E | 932 | 134 | unsupported | unsupported | unknown |
| UNKNOWN_0x4F | 938 | 143 | unsupported | unsupported | unknown |
| UNKNOWN_0x5C | 543 | 69 | unsupported | unsupported | unknown |
| UNKNOWN_0x5D | 576 | 75 | unsupported | unsupported | unknown |
| UNKNOWN_0x5E | 554 | 81 | unsupported | unsupported | unknown |
| UNKNOWN_0xA5 | 621 | 72 | unsupported | unsupported | unknown |
| UNKNOWN_0xA6 | 605 | 70 | unsupported | unsupported | unknown |
| UNKNOWN_0xA7 | 602 | 69 | unsupported | unsupported | unknown |
| UNKNOWN_0xA8 | 618 | 89 | unsupported | unsupported | unknown |
| UNKNOWN_0xA9 | 562 | 63 | unsupported | unsupported | unknown |
| UNKNOWN_0xAA | 527 | 68 | unsupported | unsupported | unknown |
| UNKNOWN_0xAB | 583 | 69 | unsupported | unsupported | unknown |
| UNKNOWN_0xAC | 631 | 70 | unsupported | unsupported | unknown |
| UNKNOWN_0xAD | 1,031 | 130 | unsupported | unsupported | unknown |
| UNKNOWN_0xAE | 507 | 68 | unsupported | unsupported | unknown |
| UNKNOWN_0xAF | 564 | 58 | unsupported | unsupported | unknown |
| UNKNOWN_0xB0 | 735 | 69 | unsupported | unsupported | unknown |
| UNKNOWN_0xB1 | 596 | 60 | unsupported | unsupported | unknown |
| UNKNOWN_0xB2 | 633 | 80 | unsupported | unsupported | unknown |
| UNKNOWN_0xB3 | 1,171 | 135 | unsupported | unsupported | unknown |
| UNKNOWN_0xB4 | 644 | 73 | unsupported | unsupported | unknown |
| UNKNOWN_0xB5 | 619 | 59 | unsupported | unsupported | unknown |
| UNKNOWN_0xB6 | 596 | 82 | unsupported | unsupported | unknown |
| UNKNOWN_0xB7 | 571 | 76 | unsupported | unsupported | unknown |
| UNKNOWN_0xB8 | 665 | 75 | unsupported | unsupported | unknown |
| UNKNOWN_0xB9 | 607 | 83 | unsupported | unsupported | unknown |
| UNKNOWN_0xBA | 486 | 70 | unsupported | unsupported | unknown |
| UNKNOWN_0xBB | 567 | 75 | unsupported | unsupported | unknown |
| UNKNOWN_0xBC | 643 | 87 | unsupported | unsupported | unknown |
| UNKNOWN_0xBD | 579 | 65 | unsupported | unsupported | unknown |
| UNKNOWN_0xBE | 553 | 66 | unsupported | unsupported | unknown |
| UNKNOWN_0xBF | 552 | 85 | unsupported | unsupported | unknown |
| UNKNOWN_0xC0 | 504 | 64 | unsupported | unsupported | unknown |
| UNKNOWN_0xC1 | 547 | 74 | unsupported | unsupported | unknown |
| UNKNOWN_0xC2 | 605 | 88 | unsupported | unsupported | unknown |
| UNKNOWN_0xC3 | 574 | 52 | unsupported | unsupported | unknown |
| UNKNOWN_0xC4 | 597 | 80 | unsupported | unsupported | unknown |
| UNKNOWN_0xC5 | 492 | 58 | unsupported | unsupported | unknown |
| UNKNOWN_0xC6 | 535 | 60 | unsupported | unsupported | unknown |
| UNKNOWN_0xC7 | 548 | 65 | unsupported | unsupported | unknown |
| UNKNOWN_0xC8 | 1,032 | 132 | unsupported | unsupported | unknown |
| UNKNOWN_0xC9 | 574 | 57 | unsupported | unsupported | unknown |
| UNKNOWN_0xCA | 527 | 65 | unsupported | unsupported | unknown |
| UNKNOWN_0xCB | 565 | 80 | unsupported | unsupported | unknown |
| UNKNOWN_0xCC | 518 | 74 | unsupported | unsupported | unknown |
| UNKNOWN_0xCD | 593 | 58 | unsupported | unsupported | unknown |
| UNKNOWN_0xCE | 525 | 75 | unsupported | unsupported | unknown |
| UNKNOWN_0xCF | 552 | 59 | unsupported | unsupported | unknown |
| UNKNOWN_0xD0 | 604 | 78 | unsupported | unsupported | unknown |
| UNKNOWN_0xD1 | 539 | 63 | unsupported | unsupported | unknown |
| UNKNOWN_0xD2 | 572 | 72 | unsupported | unsupported | unknown |
| UNKNOWN_0xD3 | 549 | 86 | unsupported | unsupported | unknown |
| UNKNOWN_0xD4 | 556 | 57 | unsupported | unsupported | unknown |
| UNKNOWN_0xD5 | 589 | 67 | unsupported | unsupported | unknown |
| UNKNOWN_0xD6 | 564 | 57 | unsupported | unsupported | unknown |
| UNKNOWN_0xD7 | 563 | 65 | unsupported | unsupported | unknown |
| UNKNOWN_0xD8 | 554 | 59 | unsupported | unsupported | unknown |
| UNKNOWN_0xD9 | 558 | 83 | unsupported | unsupported | unknown |
| UNKNOWN_0xDA | 509 | 69 | unsupported | unsupported | unknown |
| UNKNOWN_0xDB | 559 | 82 | unsupported | unsupported | unknown |
| UNKNOWN_0xDC | 585 | 53 | unsupported | unsupported | unknown |
| UNKNOWN_0xDD | 1,050 | 129 | unsupported | unsupported | unknown |
| UNKNOWN_0xDE | 583 | 84 | unsupported | unsupported | unknown |
| UNKNOWN_0xDF | 557 | 58 | unsupported | unsupported | unknown |
| UNKNOWN_0xE0 | 586 | 75 | unsupported | unsupported | unknown |
| UNKNOWN_0xE1 | 579 | 70 | unsupported | unsupported | unknown |
| UNKNOWN_0xE2 | 1,019 | 132 | unsupported | unsupported | unknown |
| UNKNOWN_0xE3 | 589 | 89 | unsupported | unsupported | unknown |
| UNKNOWN_0xE4 | 561 | 64 | unsupported | unsupported | unknown |
| UNKNOWN_0xE5 | 576 | 79 | unsupported | unsupported | unknown |
| UNKNOWN_0xE6 | 530 | 64 | unsupported | unsupported | unknown |
| UNKNOWN_0xE7 | 531 | 71 | unsupported | unsupported | unknown |
| UNKNOWN_0xE8 | 555 | 70 | unsupported | unsupported | unknown |
| UNKNOWN_0xE9 | 569 | 57 | unsupported | unsupported | unknown |
| UNKNOWN_0xEA | 831 | 77 | unsupported | unsupported | unknown |
| UNKNOWN_0xEB | 1,107 | 76 | unsupported | unsupported | unknown |
| UNKNOWN_0xEC | 1,377 | 56 | unsupported | unsupported | unknown |
| UNKNOWN_0xED | 862 | 68 | unsupported | unsupported | unknown |
| UNKNOWN_0xEE | 612 | 91 | unsupported | unsupported | unknown |
| UNKNOWN_0xEF | 1,102 | 145 | unsupported | unsupported | unknown |
| UNKNOWN_0xF6 | 568 | 59 | unsupported | unsupported | unknown |
| UNKNOWN_0xF7 | 517 | 65 | unsupported | unsupported | unknown |
| UNKNOWN_0xF8 | 529 | 62 | unsupported | unsupported | unknown |
| UNKNOWN_0xF9 | 550 | 72 | unsupported | unsupported | unknown |
| UNKNOWN_0xFB | 540 | 60 | unsupported | unsupported | unknown |
| UNKNOWN_0xFC | 539 | 68 | unsupported | unsupported | unknown |
| UNKNOWN_0xFE | 63,556 | 7,809 | 0 | 0 | 0 |
| XOR | 1,475 | 194 | 2 | 1 | 2 |

Known stack effects: 86,707,621/86,816,910 instructions (99.874%). Unsupported opcode types: 112.

Unsupported observed opcodes (with total count):

- `UNKNOWN_0x2E`: 15,351
- `UNKNOWN_0x22`: 7,180
- `UNKNOWN_0x25`: 3,242
- `UNKNOWN_0x2D`: 3,221
- `UNKNOWN_0x23`: 2,345
- `UNKNOWN_0x4D`: 1,662
- `UNKNOWN_0xEC`: 1,433
- `UNKNOWN_0xB3`: 1,306
- `UNKNOWN_0xEF`: 1,247
- `UNKNOWN_0x49`: 1,225
- `UNKNOWN_0xEB`: 1,183
- `UNKNOWN_0xDD`: 1,179
- `UNKNOWN_0x2F`: 1,168
- `UNKNOWN_0xC8`: 1,164
- `UNKNOWN_0xAD`: 1,161
- `UNKNOWN_0xE2`: 1,151
- `UNKNOWN_0x4F`: 1,081
- `UNKNOWN_0x4E`: 1,066
- `UNKNOWN_0x2B`: 990
- `UNKNOWN_0xED`: 930
- `UNKNOWN_0x4C`: 926
- `UNKNOWN_0xEA`: 908
- `UNKNOWN_0x0C`: 848
- `UNKNOWN_0x4B`: 819
- `UNKNOWN_0xB0`: 804
- `UNKNOWN_0x4A`: 769
- `UNKNOWN_0x0F`: 753
- `UNKNOWN_0xB8`: 740
- `UNKNOWN_0xBC`: 730
- `UNKNOWN_0xB4`: 717
- `UNKNOWN_0xB2`: 713
- `UNKNOWN_0xA8`: 707
- `UNKNOWN_0xEE`: 703
- `UNKNOWN_0xAC`: 701
- `UNKNOWN_0x28`: 696
- `UNKNOWN_0xA5`: 693
- `UNKNOWN_0xC2`: 693
- `UNKNOWN_0x1F`: 690
- `UNKNOWN_0xB9`: 690
- `UNKNOWN_0x0D`: 683
- `UNKNOWN_0xD0`: 682
- `UNKNOWN_0xB5`: 678
- `UNKNOWN_0xB6`: 678
- `UNKNOWN_0xE3`: 678
- `UNKNOWN_0xC4`: 677
- `UNKNOWN_0xA6`: 675
- `UNKNOWN_0xA7`: 671
- `UNKNOWN_0x0E`: 668
- `UNKNOWN_0xDE`: 667
- `UNKNOWN_0x2C`: 662
- `UNKNOWN_0xE0`: 661
- `UNKNOWN_0xB1`: 656
- `UNKNOWN_0xD5`: 656
- `UNKNOWN_0xE5`: 655
- `UNKNOWN_0xAB`: 652
- `UNKNOWN_0x5D`: 651
- `UNKNOWN_0xCD`: 651
- `UNKNOWN_0xE1`: 649
- `UNKNOWN_0xB7`: 647
- `UNKNOWN_0xCB`: 645
- `UNKNOWN_0xBD`: 644
- `UNKNOWN_0xD2`: 644
- `UNKNOWN_0xBB`: 642
- `UNKNOWN_0xD9`: 641
- `UNKNOWN_0xDB`: 641
- `UNKNOWN_0xDC`: 638
- `UNKNOWN_0xBF`: 637
- `UNKNOWN_0x5E`: 635
- `UNKNOWN_0xD3`: 635
- `UNKNOWN_0x24`: 631
- `UNKNOWN_0xC9`: 631
- `UNKNOWN_0xD7`: 628
- `UNKNOWN_0xF6`: 627
- `UNKNOWN_0xC3`: 626
- `UNKNOWN_0xE9`: 626
- `UNKNOWN_0xA9`: 625
- `UNKNOWN_0xE4`: 625
- `UNKNOWN_0xE8`: 625
- `UNKNOWN_0xAF`: 622
- `UNKNOWN_0xF9`: 622
- `UNKNOWN_0xC1`: 621
- `UNKNOWN_0xD6`: 621
- `UNKNOWN_0x21`: 620
- `UNKNOWN_0xBE`: 619
- `UNKNOWN_0xDF`: 615
- `UNKNOWN_0x2A`: 614
- `UNKNOWN_0xC7`: 613
- `UNKNOWN_0xD4`: 613
- `UNKNOWN_0xD8`: 613
- `UNKNOWN_0x5C`: 612
- `UNKNOWN_0x29`: 611
- `UNKNOWN_0xCF`: 611
- `UNKNOWN_0xFC`: 607
- `UNKNOWN_0xD1`: 602
- `UNKNOWN_0xE7`: 602
- `UNKNOWN_0xCE`: 600
- `UNKNOWN_0xFB`: 600
- `UNKNOWN_0x1E`: 597
- `UNKNOWN_0xAA`: 595
- `UNKNOWN_0xC6`: 595
- `UNKNOWN_0xE6`: 594
- `UNKNOWN_0xCA`: 592
- `UNKNOWN_0xCC`: 592
- `UNKNOWN_0xF8`: 591
- `UNKNOWN_0x27`: 583
- `UNKNOWN_0xF7`: 582
- `UNKNOWN_0xDA`: 578
- `UNKNOWN_0x26`: 575
- `UNKNOWN_0xAE`: 575
- `UNKNOWN_0xC0`: 568
- `UNKNOWN_0xBA`: 556
- `UNKNOWN_0xC5`: 550

Version-sensitive opcodes in the registry: `BASEFEE` (London), `PREVRANDAO` (Paris transition), `PUSH0` (Shanghai), and `BLOBHASH`, `BLOBBASEFEE`, `TLOAD`, `TSTORE`, `MCOPY` (Cancun). A mnemonic's documented stack effect is stable when that opcode is valid for the executing fork; unknown byte annotations are not reinterpreted through a compiler-version guess.

The parser's `UNKNOWN_0xFE` annotation is treated as `INVALID` for stack effect/control termination because FE is the documented INVALID opcode. Other unknown bytes are not assigned guessed stack effects.
