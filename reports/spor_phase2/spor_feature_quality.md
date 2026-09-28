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
    "not_applicable"
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
        "known": 42900506,
        "unknown_entry": 23524021,
        "analysis_failure_unsupported_stack_effect": 6827742,
        "analysis_failure": 4185168,
        "analysis_failure_parse_status": 159672,
        "CONSTANT": 21324426,
        "MEMORY": 4086233,
        "CALLVALUE": 405846,
        "STACK_DUP": 11892043,
        "COMPARISON": 3170792,
        "CONTROL_FLOW": 11624024,
        "STACK_POP": 4419909,
        "CALLDATA": 488213,
        "ARITHMETIC": 8587124,
        "STACK_SWAP": 7964676,
        "CALLER": 240503,
        "OTHER": 1743487,
        "STORAGE": 1508305,
        "BLOCK_ENV": 59954,
        "push_instructions": 21324739,
        "push_operand_tokens": 20922729,
        "feature_valid_tokens": 37848938,
        "unknown_source_tokens": 37536550,
        "not_applicable_tokens": 5051568,
        "CALL": 81574,
        "push_immediate_width_mismatch": 313
      },
      "target_status": {
        "JUMPI": {
          "known": 613612,
          "unknown_entry": 1332281,
          "analysis_failure": 228405,
          "analysis_failure_unsupported_stack_effect": 10815
        },
        "CALL": {
          "unknown_entry": 8445,
          "analysis_failure": 47711,
          "known": 803,
          "analysis_failure_unsupported_stack_effect": 277
        },
        "SLOAD": {
          "known": 666705,
          "analysis_failure": 340839,
          "analysis_failure_unsupported_stack_effect": 46091,
          "unknown_entry": 94520
        },
        "SSTORE": {
          "analysis_failure": 150689,
          "analysis_failure_unsupported_stack_effect": 8458,
          "unknown_entry": 133661,
          "known": 67342
        }
      },
      "instruction_status_ratio": {
        "known": 0.5528621691305535,
        "unknown_entry": 0.3031558946351983,
        "analysis_failure": 0.05393458666095408,
        "analysis_failure_unsupported_stack_effect": 0.08798964404717707,
        "analysis_failure_parse_status": 0.002057705526116959
      },
      "token_valid_ratio": 0.38355539111247533,
      "token_unknown_ratio": 0.3803896985501412,
      "push_operand_alignment": {
        "mapped_operand_tokens": 20922729,
        "mapped_not_applicable_tokens": 5051568,
        "unmapped_model_tokens": 0
      },
      "compiler_coverage": {
        "v0.6.0+commit.26b70077": {
          "contracts": 7,
          "instructions": 18109,
          "known_instructions": 8889,
          "unknown_instructions": 5864,
          "failed_instructions": 3356
        },
        "v0.8.19+commit.7dd6d404": {
          "contracts": 1338,
          "instructions": 6264076,
          "known_instructions": 3644129,
          "unknown_instructions": 1951244,
          "failed_instructions": 668703
        },
        "v0.8.7+commit.e28d00a7": {
          "contracts": 660,
          "instructions": 3689724,
          "known_instructions": 2083665,
          "unknown_instructions": 1259005,
          "failed_instructions": 347054
        },
        "v0.8.10+commit.fc410830": {
          "contracts": 739,
          "instructions": 5354498,
          "known_instructions": 3248624,
          "unknown_instructions": 1599658,
          "failed_instructions": 506216
        },
        "v0.8.9+commit.e5eed63a": {
          "contracts": 922,
          "instructions": 5570503,
          "known_instructions": 3315391,
          "unknown_instructions": 1767198,
          "failed_instructions": 487914
        },
        "v0.4.18+commit.9cf6e910": {
          "contracts": 692,
          "instructions": 1865820,
          "known_instructions": 1070590,
          "unknown_instructions": 487327,
          "failed_instructions": 307903
        },
        "v0.4.17+commit.bdeb9e52": {
          "contracts": 104,
          "instructions": 309271,
          "known_instructions": 187858,
          "unknown_instructions": 72544,
          "failed_instructions": 48869
        },
        "v0.8.15+commit.e14f2714": {
          "contracts": 199,
          "instructions": 1256583,
          "known_instructions": 761836,
          "unknown_instructions": 374362,
          "failed_instructions": 120385
        },
        "v0.8.12+commit.f00d7308": {
          "contracts": 92,
          "instructions": 567054,
          "known_instructions": 342347,
          "unknown_instructions": 174694,
          "failed_instructions": 50013
        },
        "v0.4.13+commit.fb4cb1a": {
          "contracts": 107,
          "instructions": 258167,
          "known_instructions": 153151,
          "unknown_instructions": 71441,
          "failed_instructions": 33575
        },
        "v0.8.24+commit.e11b9ed9": {
          "contracts": 696,
          "instructions": 3227564,
          "known_instructions": 1280886,
          "unknown_instructions": 1065787,
          "failed_instructions": 880891
        },
        "v0.8.17+commit.8df45f5f": {
          "contracts": 1483,
          "instructions": 8123719,
          "known_instructions": 4705149,
          "unknown_instructions": 2498223,
          "failed_instructions": 920347
        },
        "v0.8.20+commit.a1b79de6": {
          "contracts": 688,
          "instructions": 3354226,
          "known_instructions": 1383492,
          "unknown_instructions": 1008118,
          "failed_instructions": 962616
        },
        "v0.4.24+commit.e67f0147": {
          "contracts": 2021,
          "instructions": 6483081,
          "known_instructions": 3621445,
          "unknown_instructions": 1741377,
          "failed_instructions": 1120259
        },
        "v0.4.22-nightly.2018.3.8+commit.fbc29f6d": {
          "contracts": 6,
          "instructions": 19708,
          "known_instructions": 11239,
          "unknown_instructions": 5143,
          "failed_instructions": 3326
        },
        "v0.4.15+commit.8b45bddb": {
          "contracts": 1,
          "instructions": 4765,
          "known_instructions": 2284,
          "unknown_instructions": 1479,
          "failed_instructions": 1002
        },
        "v0.7.1+commit.f4a555be": {
          "contracts": 11,
          "instructions": 19715,
          "known_instructions": 11070,
          "unknown_instructions": 6034,
          "failed_instructions": 2611
        },
        "v0.8.23+commit.f704f362": {
          "contracts": 594,
          "instructions": 3086761,
          "known_instructions": 1295029,
          "unknown_instructions": 895324,
          "failed_instructions": 896408
        },
        "v0.4.25+commit.59dbf8f1": {
          "contracts": 1171,
          "instructions": 3267608,
          "known_instructions": 1832247,
          "unknown_instructions": 895870,
          "failed_instructions": 539491
        },
        "v0.4.21+commit.dfe3193c": {
          "contracts": 610,
          "instructions": 1624524,
          "known_instructions": 935425,
          "unknown_instructions": 420754,
          "failed_instructions": 268345
        },
        "v0.6.12+commit.27d51765": {
          "contracts": 319,
          "instructions": 1538235,
          "known_instructions": 842951,
          "unknown_instructions": 456805,
          "failed_instructions": 238479
        },
        "v0.4.26+commit.4563c3fc": {
          "contracts": 132,
          "instructions": 322263,
          "known_instructions": 178862,
          "unknown_instructions": 88596,
          "failed_instructions": 54805
        },
        "v0.4.11+commit.68ef5810": {
          "contracts": 186,
          "instructions": 483510,
          "known_instructions": 278622,
          "unknown_instructions": 140835,
          "failed_instructions": 64053
        },
        "v0.4.19+commit.c4cbbb05": {
          "contracts": 710,
          "instructions": 1770686,
          "known_instructions": 1018235,
          "unknown_instructions": 463746,
          "failed_instructions": 288705
        },
        "v0.7.6+commit.7338295f": {
          "contracts": 91,
          "instructions": 468010,
          "known_instructions": 249886,
          "unknown_instructions": 155705,
          "failed_instructions": 62419
        },
        "v0.8.18+commit.87f61d96": {
          "contracts": 923,
          "instructions": 5217670,
          "known_instructions": 2964441,
          "unknown_instructions": 1774039,
          "failed_instructions": 479190
        },
        "v0.4.14+commit.c2215d46": {
          "contracts": 20,
          "instructions": 45820,
          "known_instructions": 26881,
          "unknown_instructions": 11407,
          "failed_instructions": 7532
        },
        "v0.8.13+commit.abaa5c0e": {
          "contracts": 155,
          "instructions": 905854,
          "known_instructions": 523871,
          "unknown_instructions": 291261,
          "failed_instructions": 90722
        },
        "v0.8.14+commit.80d49f37": {
          "contracts": 97,
          "instructions": 562035,
          "known_instructions": 331125,
          "unknown_instructions": 185521,
          "failed_instructions": 45389
        },
        "v0.4.23+commit.124ca40d": {
          "contracts": 398,
          "instructions": 1256895,
          "known_instructions": 707688,
          "unknown_instructions": 336620,
          "failed_instructions": 212587
        },
        "v0.8.4+commit.c7e474f2": {
          "contracts": 283,
          "instructions": 1354459,
          "known_instructions": 760123,
          "unknown_instructions": 449890,
          "failed_instructions": 144446
        },
        "v0.8.21+commit.d9974bed": {
          "contracts": 107,
          "instructions": 454203,
          "known_instructions": 195020,
          "unknown_instructions": 133761,
          "failed_instructions": 125422
        },
        "v0.8.5+commit.a4f2e591": {
          "contracts": 78,
          "instructions": 302494,
          "known_instructions": 166291,
          "unknown_instructions": 102859,
          "failed_instructions": 33344
        },
        "v0.4.24-nightly.2018.5.9+commit.1e953355": {
          "contracts": 2,
          "instructions": 4858,
          "known_instructions": 2910,
          "unknown_instructions": 1141,
          "failed_instructions": 807
        },
        "v0.8.0+commit.c7dfd78e": {
          "contracts": 165,
          "instructions": 749964,
          "known_instructions": 398564,
          "unknown_instructions": 275450,
          "failed_instructions": 75950
        },
        "v0.4.20+commit.3155dd80": {
          "contracts": 227,
          "instructions": 638628,
          "known_instructions": 360991,
          "unknown_instructions": 179391,
          "failed_instructions": 98246
        },
        "v0.4.21-nightly.2018.3.1+commit.cf6720ea": {
          "contracts": 4,
          "instructions": 16559,
          "known_instructions": 8329,
          "unknown_instructions": 6164,
          "failed_instructions": 2066
        },
        "v0.8.6+commit.11564f7e": {
          "contracts": 44,
          "instructions": 182848,
          "known_instructions": 101990,
          "unknown_instructions": 60422,
          "failed_instructions": 20436
        },
        "v0.4.22+commit.4cb486ee": {
          "contracts": 57,
          "instructions": 167582,
          "known_instructions": 93042,
          "unknown_instructions": 43152,
          "failed_instructions": 31388
        },
        "v0.4.16-nightly.2017.8.24+commit.78c2dcac": {
          "contracts": 2,
          "instructions": 4130,
          "known_instructions": 2586,
          "unknown_instructions": 1154,
          "failed_instructions": 390
        },
        "v0.5.16+commit.9c3226ce": {
          "contracts": 27,
          "instructions": 112644,
          "known_instructions": 60390,
          "unknown_instructions": 35470,
          "failed_instructions": 16784
        },
        "v0.4.16+commit.d7661dd9": {
          "contracts": 221,
          "instructions": 557208,
          "known_instructions": 318014,
          "unknown_instructions": 152806,
          "failed_instructions": 86388
        },
        "v0.5.17+commit.d19bba13": {
          "contracts": 81,
          "instructions": 256916,
          "known_instructions": 138430,
          "unknown_instructions": 81602,
          "failed_instructions": 36884
        },
        "v0.8.11+commit.d7f03943": {
          "contracts": 156,
          "instructions": 950043,
          "known_instructions": 560463,
          "unknown_instructions": 286650,
          "failed_instructions": 102930
        },
        "v0.8.16+commit.07a7930e": {
          "contracts": 351,
          "instructions": 1893696,
          "known_instructions": 1121428,
          "unknown_instructions": 594414,
          "failed_instructions": 177854
        },
        "v0.4.15+commit.bbb8e64f": {
          "contracts": 114,
          "instructions": 358097,
          "known_instructions": 213906,
          "unknown_instructions": 99260,
          "failed_instructions": 44931
        },
        "v0.7.0+commit.9e61f92b": {
          "contracts": 14,
          "instructions": 65103,
          "known_instructions": 34662,
          "unknown_instructions": 21765,
          "failed_instructions": 8676
        },
        "v0.7.4+commit.3f05b770": {
          "contracts": 12,
          "instructions": 47216,
          "known_instructions": 26714,
          "unknown_instructions": 12651,
          "failed_instructions": 7851
        },
        "v0.4.12+commit.194ff033": {
          "contracts": 28,
          "instructions": 62033,
          "known_instructions": 37629,
          "unknown_instructions": 16629,
          "failed_instructions": 7775
        },
        "v0.4.19-nightly.2017.10.29+commit.eb140bc6": {
          "contracts": 4,
          "instructions": 8160,
          "known_instructions": 5049,
          "unknown_instructions": 1997,
          "failed_instructions": 1114
        },
        "v0.8.1+commit.df193b15": {
          "contracts": 22,
          "instructions": 113173,
          "known_instructions": 58772,
          "unknown_instructions": 44525,
          "failed_instructions": 9876
        },
        "v0.5.0+commit.1d4f565a": {
          "contracts": 58,
          "instructions": 130986,
          "known_instructions": 71027,
          "unknown_instructions": 44065,
          "failed_instructions": 15894
        },
        "v0.4.18-nightly.2017.9.22+commit.a2a58789": {
          "contracts": 4,
          "instructions": 15032,
          "known_instructions": 9116,
          "unknown_instructions": 3925,
          "failed_instructions": 1991
        },
        "v0.6.6+commit.6c089d02": {
          "contracts": 24,
          "instructions": 135456,
          "known_instructions": 70786,
          "unknown_instructions": 45809,
          "failed_instructions": 18861
        },
        "v0.8.22+commit.4fc1097e": {
          "contracts": 113,
          "instructions": 529953,
          "known_instructions": 221066,
          "unknown_instructions": 167449,
          "failed_instructions": 141438
        },
        "v0.7.5+commit.eb77ed08": {
          "contracts": 14,
          "instructions": 66904,
          "known_instructions": 38809,
          "unknown_instructions": 18318,
          "failed_instructions": 9777
        },
        "v0.4.20-nightly.2018.1.6+commit.2548228b": {
          "contracts": 37,
          "instructions": 90508,
          "known_instructions": 51525,
          "unknown_instructions": 24166,
          "failed_instructions": 14817
        },
        "v0.4.24-nightly.2018.4.23+commit.c7ee2ca0": {
          "contracts": 2,
          "instructions": 4928,
          "known_instructions": 2804,
          "unknown_instructions": 1215,
          "failed_instructions": 909
        },
        "v0.4.25-nightly.2018.5.17+commit.4aa2f036": {
          "contracts": 1,
          "instructions": 3385,
          "known_instructions": 1792,
          "unknown_instructions": 1017,
          "failed_instructions": 576
        },
        "v0.4.19-nightly.2017.11.11+commit.284c3839": {
          "contracts": 12,
          "instructions": 29504,
          "known_instructions": 16940,
          "unknown_instructions": 7274,
          "failed_instructions": 5290
        },
        "v0.8.8+commit.dddeac2f": {
          "contracts": 10,
          "instructions": 52405,
          "known_instructions": 28754,
          "unknown_instructions": 18469,
          "failed_instructions": 5182
        },
        "v0.4.20-nightly.2017.12.8+commit.226bfe5b": {
          "contracts": 1,
          "instructions": 2265,
          "known_instructions": 1155,
          "unknown_instructions": 745,
          "failed_instructions": 365
        },
        "v0.4.9+commit.364da425": {
          "contracts": 9,
          "instructions": 42166,
          "known_instructions": 22407,
          "unknown_instructions": 11336,
          "failed_instructions": 8423
        },
        "v0.5.12+commit.7709ece9": {
          "contracts": 19,
          "instructions": 79308,
          "known_instructions": 43511,
          "unknown_instructions": 21129,
          "failed_instructions": 14668
        },
        "v0.5.7+commit.6da8b019": {
          "contracts": 1,
          "instructions": 237,
          "known_instructions": 157,
          "unknown_instructions": 51,
          "failed_instructions": 29
        },
        "v0.4.4+commit.4633f3de": {
          "contracts": 4,
          "instructions": 5426,
          "known_instructions": 2994,
          "unknown_instructions": 1329,
          "failed_instructions": 1103
        },
        "v0.4.25-nightly.2018.6.12+commit.56a965ea": {
          "contracts": 3,
          "instructions": 8462,
          "known_instructions": 4449,
          "unknown_instructions": 2428,
          "failed_instructions": 1585
        },
        "v0.4.20-nightly.2018.1.26+commit.bbad48bb": {
          "contracts": 2,
          "instructions": 6181,
          "known_instructions": 3721,
          "unknown_instructions": 1744,
          "failed_instructions": 716
        },
        "v0.8.2+commit.661d1103": {
          "contracts": 35,
          "instructions": 127915,
          "known_instructions": 68516,
          "unknown_instructions": 45922,
          "failed_instructions": 13477
        },
        "v0.4.25-nightly.2018.6.3+commit.ef8fb63b": {
          "contracts": 4,
          "instructions": 9176,
          "known_instructions": 5215,
          "unknown_instructions": 2614,
          "failed_instructions": 1347
        },
        "v0.4.20-nightly.2017.12.20+commit.efc198d5": {
          "contracts": 5,
          "instructions": 17686,
          "known_instructions": 8864,
          "unknown_instructions": 4525,
          "failed_instructions": 4297
        },
        "v0.4.20-nightly.2018.1.29+commit.a668b9de": {
          "contracts": 6,
          "instructions": 12625,
          "known_instructions": 7021,
          "unknown_instructions": 3553,
          "failed_instructions": 2051
        },
        "v0.4.25-nightly.2018.7.31+commit.75c1a9bd": {
          "contracts": 1,
          "instructions": 5429,
          "known_instructions": 2455,
          "unknown_instructions": 2361,
          "failed_instructions": 613
        },
        "v0.4.24-nightly.2018.5.4+commit.81d61ca0": {
          "contracts": 1,
          "instructions": 10716,
          "known_instructions": 4901,
          "unknown_instructions": 2903,
          "failed_instructions": 2912
        },
        "v0.4.25-nightly.2018.5.21+commit.e97f9b6b": {
          "contracts": 6,
          "instructions": 22739,
          "known_instructions": 13861,
          "unknown_instructions": 5724,
          "failed_instructions": 3154
        },
        "v0.4.22-nightly.2018.4.12+commit.c3dc67d0": {
          "contracts": 2,
          "instructions": 4676,
          "known_instructions": 2576,
          "unknown_instructions": 1481,
          "failed_instructions": 619
        },
        "v0.4.25-nightly.2018.6.14+commit.baeabe1c": {
          "contracts": 6,
          "instructions": 14110,
          "known_instructions": 8094,
          "unknown_instructions": 3829,
          "failed_instructions": 2187
        },
        "v0.5.11+commit.c082d0b4": {
          "contracts": 2,
          "instructions": 3721,
          "known_instructions": 1770,
          "unknown_instructions": 1265,
          "failed_instructions": 686
        },
        "v0.7.3+commit.9bfce1f6": {
          "contracts": 10,
          "instructions": 34179,
          "known_instructions": 18701,
          "unknown_instructions": 10154,
          "failed_instructions": 5324
        },
        "v0.4.19-nightly.2017.11.22+commit.f22ac8fc": {
          "contracts": 1,
          "instructions": 1453,
          "known_instructions": 795,
          "unknown_instructions": 414,
          "failed_instructions": 244
        },
        "v0.4.25-nightly.2018.5.18+commit.4d7b092c": {
          "contracts": 7,
          "instructions": 31658,
          "known_instructions": 18171,
          "unknown_instructions": 8120,
          "failed_instructions": 5367
        },
        "v0.4.22-nightly.2018.3.30+commit.326d656a": {
          "contracts": 2,
          "instructions": 5283,
          "known_instructions": 2944,
          "unknown_instructions": 1422,
          "failed_instructions": 917
        },
        "v0.6.10+commit.00c0fcaf": {
          "contracts": 6,
          "instructions": 33655,
          "known_instructions": 18819,
          "unknown_instructions": 10397,
          "failed_instructions": 4439
        },
        "v0.4.8+commit.60cc1668": {
          "contracts": 18,
          "instructions": 39551,
          "known_instructions": 22533,
          "unknown_instructions": 11673,
          "failed_instructions": 5345
        },
        "v0.7.2+commit.51b20bc0": {
          "contracts": 6,
          "instructions": 40651,
          "known_instructions": 23042,
          "unknown_instructions": 12631,
          "failed_instructions": 4978
        },
        "v0.4.25-nightly.2018.5.23+commit.18c651b7": {
          "contracts": 9,
          "instructions": 22526,
          "known_instructions": 12759,
          "unknown_instructions": 5889,
          "failed_instructions": 3878
        },
        "v0.6.7+commit.b8d736ae": {
          "contracts": 5,
          "instructions": 7157,
          "known_instructions": 4530,
          "unknown_instructions": 1782,
          "failed_instructions": 845
        },
        "v0.8.25+commit.b61c2a91": {
          "contracts": 27,
          "instructions": 86788,
          "known_instructions": 33421,
          "unknown_instructions": 31667,
          "failed_instructions": 21700
        },
        "v0.4.22-nightly.2018.3.13+commit.f2614be9": {
          "contracts": 4,
          "instructions": 8614,
          "known_instructions": 4799,
          "unknown_instructions": 2238,
          "failed_instructions": 1577
        },
        "v0.4.22-nightly.2018.3.21+commit.8fd53c1c": {
          "contracts": 7,
          "instructions": 13322,
          "known_instructions": 7793,
          "unknown_instructions": 3523,
          "failed_instructions": 2006
        },
        "v0.4.24-nightly.2018.5.10+commit.85d417a8": {
          "contracts": 6,
          "instructions": 6444,
          "known_instructions": 4518,
          "unknown_instructions": 1190,
          "failed_instructions": 736
        },
        "v0.4.2+commit.af6afb04": {
          "contracts": 3,
          "instructions": 4413,
          "known_instructions": 2351,
          "unknown_instructions": 1409,
          "failed_instructions": 653
        },
        "v0.4.25-nightly.2018.6.8+commit.81c5a6e4": {
          "contracts": 7,
          "instructions": 19758,
          "known_instructions": 11829,
          "unknown_instructions": 4785,
          "failed_instructions": 3144
        },
        "v0.4.6+commit.2dabbdf0": {
          "contracts": 6,
          "instructions": 11058,
          "known_instructions": 6072,
          "unknown_instructions": 3116,
          "failed_instructions": 1870
        },
        "v0.5.15+commit.6a57276f": {
          "contracts": 7,
          "instructions": 22244,
          "known_instructions": 11141,
          "unknown_instructions": 6678,
          "failed_instructions": 4425
        },
        "v0.4.22-nightly.2018.3.14+commit.c3f07b52": {
          "contracts": 2,
          "instructions": 4812,
          "known_instructions": 2691,
          "unknown_instructions": 1267,
          "failed_instructions": 854
        },
        "v0.4.21-nightly.2018.2.23+commit.cae6cc2c": {
          "contracts": 3,
          "instructions": 4511,
          "known_instructions": 2522,
          "unknown_instructions": 1272,
          "failed_instructions": 717
        },
        "v0.6.9+commit.3e3065ac": {
          "contracts": 2,
          "instructions": 15663,
          "known_instructions": 9578,
          "unknown_instructions": 4494,
          "failed_instructions": 1591
        },
        "v0.4.11-nightly.2017.3.16+commit.a2eb2c0a": {
          "contracts": 1,
          "instructions": 1814,
          "known_instructions": 1061,
          "unknown_instructions": 476,
          "failed_instructions": 277
        },
        "v0.5.10+commit.5a6ea5b1": {
          "contracts": 12,
          "instructions": 35828,
          "known_instructions": 20043,
          "unknown_instructions": 9017,
          "failed_instructions": 6768
        },
        "v0.5.14+commit.01f1aaa4": {
          "contracts": 2,
          "instructions": 507,
          "known_instructions": 262,
          "unknown_instructions": 167,
          "failed_instructions": 78
        },
        "v0.4.10+commit.f0d539ae": {
          "contracts": 9,
          "instructions": 22396,
          "known_instructions": 12243,
          "unknown_instructions": 6761,
          "failed_instructions": 3392
        },
        "v0.4.25-nightly.2018.6.22+commit.9b67bdb3": {
          "contracts": 2,
          "instructions": 5313,
          "known_instructions": 3089,
          "unknown_instructions": 1311,
          "failed_instructions": 913
        },
        "v0.4.5+commit.b318366e": {
          "contracts": 1,
          "instructions": 393,
          "known_instructions": 304,
          "unknown_instructions": 68,
          "failed_instructions": 21
        },
        "v0.8.3+commit.8d00100c": {
          "contracts": 9,
          "instructions": 35155,
          "known_instructions": 18909,
          "unknown_instructions": 11904,
          "failed_instructions": 4342
        },
        "v0.4.22-nightly.2018.3.16+commit.2b2527f3": {
          "contracts": 3,
          "instructions": 8627,
          "known_instructions": 4811,
          "unknown_instructions": 2139,
          "failed_instructions": 1677
        },
        "v0.5.1+commit.c8a2cb62": {
          "contracts": 1,
          "instructions": 2823,
          "known_instructions": 1614,
          "unknown_instructions": 959,
          "failed_instructions": 250
        },
        "v0.4.21-nightly.2018.2.15+commit.f4aa05f3": {
          "contracts": 1,
          "instructions": 2219,
          "known_instructions": 1259,
          "unknown_instructions": 580,
          "failed_instructions": 380
        },
        "v0.4.7-nightly.2016.11.24+commit.851f8576": {
          "contracts": 1,
          "instructions": 622,
          "known_instructions": 361,
          "unknown_instructions": 69,
          "failed_instructions": 192
        },
        "v0.4.23-nightly.2018.4.19+commit.ae834e3d": {
          "contracts": 3,
          "instructions": 14098,
          "known_instructions": 6728,
          "unknown_instructions": 5712,
          "failed_instructions": 1658
        },
        "v0.4.20-nightly.2018.1.23+commit.31aaf433": {
          "contracts": 1,
          "instructions": 2051,
          "known_instructions": 1036,
          "unknown_instructions": 648,
          "failed_instructions": 367
        },
        "v0.4.22-nightly.2018.4.6+commit.9bd49516": {
          "contracts": 1,
          "instructions": 2106,
          "known_instructions": 1088,
          "unknown_instructions": 652,
          "failed_instructions": 366
        },
        "v0.4.11-nightly.2017.5.3+commit.1aa0f77a": {
          "contracts": 2,
          "instructions": 3474,
          "known_instructions": 1899,
          "unknown_instructions": 1042,
          "failed_instructions": 533
        },
        "v0.4.21-nightly.2018.2.27+commit.415ac2ae": {
          "contracts": 1,
          "instructions": 2170,
          "known_instructions": 1375,
          "unknown_instructions": 500,
          "failed_instructions": 295
        },
        "v0.4.24-nightly.2018.4.30+commit.9e61b25d": {
          "contracts": 2,
          "instructions": 6176,
          "known_instructions": 3480,
          "unknown_instructions": 1505,
          "failed_instructions": 1191
        },
        "v0.4.25-nightly.2018.5.28+commit.c223b03": {
          "contracts": 1,
          "instructions": 3095,
          "known_instructions": 1847,
          "unknown_instructions": 788,
          "failed_instructions": 460
        },
        "v0.4.25-nightly.2018.5.30+commit.3f3d6df2": {
          "contracts": 2,
          "instructions": 9648,
          "known_instructions": 4932,
          "unknown_instructions": 2334,
          "failed_instructions": 2382
        },
        "v0.4.19-nightly.2017.11.30+commit.f5a2508e": {
          "contracts": 2,
          "instructions": 5346,
          "known_instructions": 2978,
          "unknown_instructions": 1413,
          "failed_instructions": 955
        },
        "v0.4.24-nightly.2018.5.15+commit.b8b46099": {
          "contracts": 1,
          "instructions": 1559,
          "known_instructions": 817,
          "unknown_instructions": 509,
          "failed_instructions": 233
        },
        "v0.4.24-nightly.2018.4.27+commit.1604a996": {
          "contracts": 3,
          "instructions": 8929,
          "known_instructions": 5013,
          "unknown_instructions": 2472,
          "failed_instructions": 1444
        },
        "v0.6.4+commit.1dca32f3": {
          "contracts": 4,
          "instructions": 19760,
          "known_instructions": 10223,
          "unknown_instructions": 6741,
          "failed_instructions": 2796
        },
        "v0.4.20-nightly.2017.12.14+commit.3d1830f3": {
          "contracts": 2,
          "instructions": 3606,
          "known_instructions": 1865,
          "unknown_instructions": 1095,
          "failed_instructions": 646
        },
        "v0.4.19-nightly.2017.10.19+commit.c58d9d2c": {
          "contracts": 1,
          "instructions": 3234,
          "known_instructions": 1667,
          "unknown_instructions": 825,
          "failed_instructions": 742
        },
        "v0.4.7+commit.822622cf": {
          "contracts": 1,
          "instructions": 2714,
          "known_instructions": 1885,
          "unknown_instructions": 619,
          "failed_instructions": 210
        },
        "v0.4.25-nightly.2018.6.7+commit.ddd256a6": {
          "contracts": 1,
          "instructions": 2763,
          "known_instructions": 1477,
          "unknown_instructions": 776,
          "failed_instructions": 510
        },
        "v0.5.5+commit.47a71e8f": {
          "contracts": 2,
          "instructions": 6143,
          "known_instructions": 3782,
          "unknown_instructions": 1720,
          "failed_instructions": 641
        },
        "v0.4.24-nightly.2018.5.16+commit.7f965c86": {
          "contracts": 10,
          "instructions": 27452,
          "known_instructions": 15742,
          "unknown_instructions": 7532,
          "failed_instructions": 4178
        },
        "v0.4.25-nightly.2018.8.1+commit.21888e24": {
          "contracts": 1,
          "instructions": 7972,
          "known_instructions": 2328,
          "unknown_instructions": 4248,
          "failed_instructions": 1396
        },
        "v0.6.11+commit.5ef660b1": {
          "contracts": 3,
          "instructions": 15233,
          "known_instructions": 8963,
          "unknown_instructions": 3959,
          "failed_instructions": 2311
        },
        "v0.5.8+commit.23d335f2": {
          "contracts": 6,
          "instructions": 26832,
          "known_instructions": 13246,
          "unknown_instructions": 7844,
          "failed_instructions": 5742
        },
        "v0.4.20-nightly.2018.2.13+commit.27ef9794": {
          "contracts": 2,
          "instructions": 5073,
          "known_instructions": 2735,
          "unknown_instructions": 1636,
          "failed_instructions": 702
        },
        "v0.4.22-nightly.2018.4.16+commit.d8030c9b": {
          "contracts": 2,
          "instructions": 5078,
          "known_instructions": 2879,
          "unknown_instructions": 1320,
          "failed_instructions": 879
        },
        "v0.4.25-nightly.2018.6.4+commit.a074d84": {
          "contracts": 1,
          "instructions": 7680,
          "known_instructions": 3608,
          "unknown_instructions": 1751,
          "failed_instructions": 2321
        },
        "v0.6.8+commit.0bbfe453": {
          "contracts": 3,
          "instructions": 11817,
          "known_instructions": 6064,
          "unknown_instructions": 3857,
          "failed_instructions": 1896
        },
        "v0.4.20-nightly.2018.1.4+commit.a0771691": {
          "contracts": 2,
          "instructions": 2960,
          "known_instructions": 1662,
          "unknown_instructions": 778,
          "failed_instructions": 520
        },
        "v0.4.19-nightly.2017.10.18+commit.f7ca2421": {
          "contracts": 3,
          "instructions": 4917,
          "known_instructions": 2776,
          "unknown_instructions": 1284,
          "failed_instructions": 857
        },
        "v0.4.22-nightly.2018.3.12+commit.c6e9dd13": {
          "contracts": 1,
          "instructions": 1558,
          "known_instructions": 895,
          "unknown_instructions": 407,
          "failed_instructions": 256
        },
        "v0.4.17-nightly.2017.8.24+commit.12d9f79": {
          "contracts": 2,
          "instructions": 4291,
          "known_instructions": 2424,
          "unknown_instructions": 1226,
          "failed_instructions": 641
        },
        "v0.5.2+commit.1df8f40c": {
          "contracts": 1,
          "instructions": 3047,
          "known_instructions": 1672,
          "unknown_instructions": 1088,
          "failed_instructions": 287
        },
        "v0.4.0+commit.acd334c9": {
          "contracts": 3,
          "instructions": 9722,
          "known_instructions": 5787,
          "unknown_instructions": 2956,
          "failed_instructions": 979
        },
        "v0.4.23-nightly.2018.4.17+commit.5499db01": {
          "contracts": 2,
          "instructions": 4166,
          "known_instructions": 2474,
          "unknown_instructions": 1028,
          "failed_instructions": 664
        },
        "v0.4.21-nightly.2018.3.6+commit.a9e02acc": {
          "contracts": 2,
          "instructions": 4104,
          "known_instructions": 2650,
          "unknown_instructions": 928,
          "failed_instructions": 526
        },
        "v0.4.25-nightly.2018.6.26+commit.24f124f8": {
          "contracts": 2,
          "instructions": 6119,
          "known_instructions": 3054,
          "unknown_instructions": 1730,
          "failed_instructions": 1335
        },
        "v0.4.24-nightly.2018.5.7+commit.6db7e09a": {
          "contracts": 1,
          "instructions": 5335,
          "known_instructions": 2629,
          "unknown_instructions": 1593,
          "failed_instructions": 1113
        },
        "v0.6.2+commit.bacdbe57": {
          "contracts": 8,
          "instructions": 30764,
          "known_instructions": 16841,
          "unknown_instructions": 9458,
          "failed_instructions": 4465
        },
        "v0.4.25-nightly.2018.5.16+commit.3897c367": {
          "contracts": 3,
          "instructions": 8464,
          "known_instructions": 5102,
          "unknown_instructions": 2116,
          "failed_instructions": 1246
        },
        "v0.4.22-nightly.2018.4.13+commit.2001cc6b": {
          "contracts": 2,
          "instructions": 3837,
          "known_instructions": 2050,
          "unknown_instructions": 1014,
          "failed_instructions": 773
        },
        "v0.4.20-nightly.2018.1.22+commit.e5def2da": {
          "contracts": 2,
          "instructions": 4262,
          "known_instructions": 2170,
          "unknown_instructions": 1223,
          "failed_instructions": 869
        },
        "v0.4.14-nightly.2017.7.24+commit.cfb11ff7": {
          "contracts": 1,
          "instructions": 1359,
          "known_instructions": 772,
          "unknown_instructions": 449,
          "failed_instructions": 138
        },
        "v0.4.25-nightly.2018.8.16+commit.a9e7ae29": {
          "contracts": 1,
          "instructions": 2186,
          "known_instructions": 1177,
          "unknown_instructions": 665,
          "failed_instructions": 344
        },
        "v0.4.17-nightly.2017.9.21+commit.725b4fc2": {
          "contracts": 3,
          "instructions": 9884,
          "known_instructions": 6697,
          "unknown_instructions": 2308,
          "failed_instructions": 879
        },
        "v0.4.21-nightly.2018.2.21+commit.16c7eabc": {
          "contracts": 1,
          "instructions": 3673,
          "known_instructions": 2872,
          "unknown_instructions": 624,
          "failed_instructions": 177
        },
        "v0.4.15-nightly.2017.7.31+commit.93f90eb2": {
          "contracts": 1,
          "instructions": 2652,
          "known_instructions": 1873,
          "unknown_instructions": 448,
          "failed_instructions": 331
        },
        "v0.4.20-nightly.2018.1.5+commit.bca01f8f": {
          "contracts": 1,
          "instructions": 2187,
          "known_instructions": 1258,
          "unknown_instructions": 562,
          "failed_instructions": 367
        },
        "v0.4.23-nightly.2018.4.18+commit.85687a37": {
          "contracts": 1,
          "instructions": 2219,
          "known_instructions": 1399,
          "unknown_instructions": 515,
          "failed_instructions": 305
        },
        "v0.4.19-nightly.2017.10.27+commit.1e085f85": {
          "contracts": 2,
          "instructions": 3085,
          "known_instructions": 1866,
          "unknown_instructions": 716,
          "failed_instructions": 503
        },
        "v0.4.21-nightly.2018.3.5+commit.cd6ffbdf": {
          "contracts": 2,
          "instructions": 5299,
          "known_instructions": 3110,
          "unknown_instructions": 1362,
          "failed_instructions": 827
        },
        "v0.4.24-nightly.2018.5.2+commit.dc18cde6": {
          "contracts": 1,
          "instructions": 1059,
          "known_instructions": 612,
          "unknown_instructions": 314,
          "failed_instructions": 133
        },
        "v0.4.24-nightly.2018.5.14+commit.7a669b39": {
          "contracts": 1,
          "instructions": 2592,
          "known_instructions": 1468,
          "unknown_instructions": 691,
          "failed_instructions": 433
        },
        "v0.4.21-nightly.2018.3.7+commit.bd7bc7c4": {
          "contracts": 2,
          "instructions": 11614,
          "known_instructions": 6153,
          "unknown_instructions": 3752,
          "failed_instructions": 1709
        },
        "v0.4.24-nightly.2018.4.25+commit.81cca26f": {
          "contracts": 1,
          "instructions": 2284,
          "known_instructions": 1325,
          "unknown_instructions": 560,
          "failed_instructions": 399
        },
        "v0.4.14-nightly.2017.7.28+commit.7e40def6": {
          "contracts": 1,
          "instructions": 2228,
          "known_instructions": 1275,
          "unknown_instructions": 520,
          "failed_instructions": 433
        },
        "v0.4.16-nightly.2017.8.15+commit.dca1f45c": {
          "contracts": 1,
          "instructions": 1525,
          "known_instructions": 1040,
          "unknown_instructions": 418,
          "failed_instructions": 67
        },
        "v0.4.22-nightly.2018.3.27+commit.af262281": {
          "contracts": 1,
          "instructions": 3120,
          "known_instructions": 1788,
          "unknown_instructions": 997,
          "failed_instructions": 335
        },
        "v0.4.19-nightly.2017.10.23+commit.dc6b1f02": {
          "contracts": 1,
          "instructions": 1036,
          "known_instructions": 652,
          "unknown_instructions": 203,
          "failed_instructions": 181
        },
        "v0.4.18-nightly.2017.10.16+commit.dbc8655b": {
          "contracts": 1,
          "instructions": 4091,
          "known_instructions": 2788,
          "unknown_instructions": 894,
          "failed_instructions": 409
        },
        "v0.4.24-nightly.2018.5.11+commit.43803b1a": {
          "contracts": 1,
          "instructions": 4911,
          "known_instructions": 2760,
          "unknown_instructions": 1387,
          "failed_instructions": 764
        },
        "v0.4.19-nightly.2017.10.26+commit.59d4dfbd": {
          "contracts": 1,
          "instructions": 4245,
          "known_instructions": 2669,
          "unknown_instructions": 961,
          "failed_instructions": 615
        },
        "v0.4.3-nightly.2016.9.30+commit.d5cfb17b": {
          "contracts": 2,
          "instructions": 5313,
          "known_instructions": 2677,
          "unknown_instructions": 1438,
          "failed_instructions": 1198
        },
        "v0.4.20-nightly.2018.1.15+commit.14fcbd65": {
          "contracts": 1,
          "instructions": 2284,
          "known_instructions": 1340,
          "unknown_instructions": 577,
          "failed_instructions": 367
        },
        "v0.4.19-nightly.2017.10.28+commit.f9b24009": {
          "contracts": 1,
          "instructions": 619,
          "known_instructions": 366,
          "unknown_instructions": 196,
          "failed_instructions": 57
        },
        "v0.4.13-nightly.2017.7.3+commit.6e4e627b": {
          "contracts": 1,
          "instructions": 3513,
          "known_instructions": 2104,
          "unknown_instructions": 951,
          "failed_instructions": 458
        },
        "v0.4.16-nightly.2017.8.11+commit.c84de7fa": {
          "contracts": 1,
          "instructions": 1947,
          "known_instructions": 1056,
          "unknown_instructions": 575,
          "failed_instructions": 316
        },
        "v0.4.17-nightly.2017.8.28+commit.d15cde2a": {
          "contracts": 1,
          "instructions": 3560,
          "known_instructions": 2214,
          "unknown_instructions": 708,
          "failed_instructions": 638
        },
        "v0.4.25-nightly.2018.6.29+commit.c9cab803": {
          "contracts": 1,
          "instructions": 1537,
          "known_instructions": 818,
          "unknown_instructions": 488,
          "failed_instructions": 231
        },
        "v0.4.22-nightly.2018.3.7+commit.b5e804b8": {
          "contracts": 1,
          "instructions": 2131,
          "known_instructions": 1165,
          "unknown_instructions": 513,
          "failed_instructions": 453
        },
        "v0.4.25-nightly.2018.6.6+commit.59b35fa5": {
          "contracts": 1,
          "instructions": 435,
          "known_instructions": 297,
          "unknown_instructions": 67,
          "failed_instructions": 71
        },
        "v0.4.24-nightly.2018.4.24+commit.258ae892": {
          "contracts": 1,
          "instructions": 3227,
          "known_instructions": 1905,
          "unknown_instructions": 782,
          "failed_instructions": 540
        },
        "v0.4.20-nightly.2018.1.19+commit.eba46a65": {
          "contracts": 1,
          "instructions": 2834,
          "known_instructions": 1525,
          "unknown_instructions": 821,
          "failed_instructions": 488
        },
        "v0.4.15-nightly.2017.8.2+commit.4166ce1": {
          "contracts": 1,
          "instructions": 1546,
          "known_instructions": 951,
          "unknown_instructions": 433,
          "failed_instructions": 162
        },
        "v0.4.21-nightly.2018.2.20+commit.dcc4083b": {
          "contracts": 1,
          "instructions": 1555,
          "known_instructions": 829,
          "unknown_instructions": 447,
          "failed_instructions": 279
        }
      },
      "failure_examples": [
        {
          "contract_id": "13386",
          "errors": {
            "unknown_opcode_annotation": 12
          }
        },
        {
          "contract_id": "15474",
          "errors": {
            "unknown_opcode_annotation": 3
          }
        },
        {
          "contract_id": "8117",
          "errors": {
            "unknown_opcode_annotation": 2
          }
        },
        {
          "contract_id": "4315",
          "errors": {
            "unknown_opcode_annotation": 7
          }
        },
        {
          "contract_id": "13864",
          "errors": {
            "unknown_opcode_annotation": 7
          }
        },
        {
          "contract_id": "15060",
          "errors": {
            "unknown_opcode_annotation": 8
          }
        },
        {
          "contract_id": "21077",
          "errors": {
            "unknown_opcode_annotation": 7,
            "push_immediate_width_mismatch": 1
          }
        },
        {
          "contract_id": "968",
          "errors": {
            "unknown_opcode_annotation": 14
          }
        },
        {
          "contract_id": "14012",
          "errors": {
            "unknown_opcode_annotation": 10
          }
        },
        {
          "contract_id": "19539",
          "errors": {
            "unknown_opcode_annotation": 2
          }
        },
        {
          "contract_id": "13216",
          "errors": {
            "unknown_opcode_annotation": 12
          }
        },
        {
          "contract_id": "18939",
          "errors": {
            "unknown_opcode_annotation": 14
          }
        },
        {
          "contract_id": "19123",
          "errors": {
            "unknown_opcode_annotation": 2
          }
        },
        {
          "contract_id": "11566",
          "errors": {
            "unknown_opcode_annotation": 3
          }
        },
        {
          "contract_id": "17998",
          "errors": {
            "unknown_opcode_annotation": 4
          }
        },
        {
          "contract_id": "17342",
          "errors": {
            "unknown_opcode_annotation": 3
          }
        },
        {
          "contract_id": "15306",
          "errors": {
            "unknown_opcode_annotation": 54
          }
        },
        {
          "contract_id": "19262",
          "errors": {
            "unknown_opcode_annotation": 16
          }
        },
        {
          "contract_id": "8006",
          "errors": {
            "unknown_opcode_annotation": 8
          }
        },
        {
          "contract_id": "20727",
          "errors": {
            "unknown_opcode_annotation": 1
          }
        },
        {
          "contract_id": "7373",
          "errors": {
            "unknown_opcode_annotation": 6
          }
        },
        {
          "contract_id": "3538",
          "errors": {
            "unknown_opcode_annotation": 4
          }
        },
        {
          "contract_id": "7161",
          "errors": {
            "unknown_opcode_annotation": 2
          }
        },
        {
          "contract_id": "6174",
          "errors": {
            "unknown_opcode_annotation": 15
          }
        },
        {
          "contract_id": "9890",
          "errors": {
            "unknown_opcode_annotation": 1
          }
        },
        {
          "contract_id": "17167",
          "errors": {
            "unknown_opcode_annotation": 1
          }
        },
        {
          "contract_id": "10780",
          "errors": {
            "unknown_opcode_annotation": 10
          }
        },
        {
          "contract_id": "1302",
          "errors": {
            "unknown_opcode_annotation": 7
          }
        },
        {
          "contract_id": "7570",
          "errors": {
            "unknown_opcode_annotation": 6
          }
        },
        {
          "contract_id": "6959",
          "errors": {
            "unknown_opcode_annotation": 2
          }
        },
        {
          "contract_id": "7658",
          "errors": {
            "unknown_opcode_annotation": 2
          }
        },
        {
          "contract_id": "14750",
          "errors": {
            "unknown_opcode_annotation": 8
          }
        },
        {
          "contract_id": "21496",
          "errors": {
            "unknown_opcode_annotation": 9
          }
        },
        {
          "contract_id": "11723",
          "errors": {
            "unknown_opcode_annotation": 8
          }
        },
        {
          "contract_id": "15286",
          "errors": {
            "unknown_opcode_annotation": 4
          }
        },
        {
          "contract_id": "2054",
          "errors": {
            "unknown_opcode_annotation": 7
          }
        },
        {
          "contract_id": "10388",
          "errors": {
            "unknown_opcode_annotation": 8
          }
        },
        {
          "contract_id": "7976",
          "errors": {
            "unknown_opcode_annotation": 8
          }
        },
        {
          "contract_id": "15780",
          "errors": {
            "unknown_opcode_annotation": 2
          }
        },
        {
          "contract_id": "760",
          "errors": {
            "unknown_opcode_annotation": 28
          }
        },
        {
          "contract_id": "20480",
          "errors": {
            "unknown_opcode_annotation": 2
          }
        },
        {
          "contract_id": "17770",
          "errors": {
            "unknown_opcode_annotation": 5
          }
        },
        {
          "contract_id": "21506",
          "errors": {
            "unknown_opcode_annotation": 3
          }
        },
        {
          "contract_id": "15",
          "errors": {
            "unknown_opcode_annotation": 15
          }
        },
        {
          "contract_id": "13129",
          "errors": {
            "unknown_opcode_annotation": 2
          }
        },
        {
          "contract_id": "20786",
          "errors": {
            "unknown_opcode_annotation": 10
          }
        },
        {
          "contract_id": "6669",
          "errors": {
            "unknown_opcode_annotation": 8
          }
        },
        {
          "contract_id": "5718",
          "errors": {
            "unknown_opcode_annotation": 2
          }
        },
        {
          "contract_id": "11082",
          "errors": {
            "unknown_opcode_annotation": 1
          }
        },
        {
          "contract_id": "7643",
          "errors": {
            "unknown_opcode_annotation": 2
          }
        }
      ],
      "processing_seconds": {
        "mean": 0.05886800824001551,
        "p50": 0.050002500000118744,
        "p95": 0.14505390499252815
      },
      "cache_bytes": 3256979303,
      "total_seconds": 1069.1994016000099
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
        "known": 5334522,
        "unknown_entry": 2926779,
        "analysis_failure_unsupported_stack_effect": 869128,
        "analysis_failure": 521835,
        "analysis_failure_parse_status": 21328,
        "CONSTANT": 2659881,
        "MEMORY": 507845,
        "CALLDATA": 62262,
        "COMPARISON": 395453,
        "CONTROL_FLOW": 1446689,
        "ARITHMETIC": 1066796,
        "STACK_DUP": 1478592,
        "CALLVALUE": 52056,
        "STACK_POP": 547169,
        "STACK_SWAP": 990380,
        "OTHER": 226194,
        "STORAGE": 190852,
        "CALLER": 30851,
        "BLOCK_ENV": 8670,
        "CALL": 9902,
        "push_instructions": 2659914,
        "push_operand_tokens": 2608133,
        "feature_valid_tokens": 4706788,
        "unknown_source_tokens": 4698147,
        "not_applicable_tokens": 627734
      },
      "target_status": {
        "JUMPI": {
          "known": 77081,
          "unknown_entry": 166669,
          "analysis_failure": 28380,
          "analysis_failure_unsupported_stack_effect": 1348
        },
        "CALL": {
          "analysis_failure": 5837,
          "unknown_entry": 1032,
          "analysis_failure_unsupported_stack_effect": 40,
          "known": 98
        },
        "SLOAD": {
          "analysis_failure": 42958,
          "known": 84707,
          "unknown_entry": 11547,
          "analysis_failure_unsupported_stack_effect": 5891
        },
        "SSTORE": {
          "analysis_failure": 19075,
          "unknown_entry": 16856,
          "known": 8683,
          "analysis_failure_unsupported_stack_effect": 1135
        }
      },
      "instruction_status_ratio": {
        "known": 0.5514520356037343,
        "unknown_entry": 0.30255348788743625,
        "analysis_failure": 0.05394428460493269,
        "analysis_failure_unsupported_stack_effect": 0.08984542660058435,
        "analysis_failure_parse_status": 0.002204765303312358
      },
      "token_valid_ratio": 0.38257175880393596,
      "token_unknown_ratio": 0.38186941092512244,
      "push_operand_alignment": {
        "mapped_operand_tokens": 2608133,
        "mapped_not_applicable_tokens": 627734,
        "unmapped_model_tokens": 0
      },
      "compiler_coverage": {
        "v0.8.10+commit.fc410830": {
          "contracts": 84,
          "instructions": 608029,
          "known_instructions": 371159,
          "unknown_instructions": 180464,
          "failed_instructions": 56406
        },
        "v0.8.11+commit.d7f03943": {
          "contracts": 14,
          "instructions": 93672,
          "known_instructions": 55661,
          "unknown_instructions": 29275,
          "failed_instructions": 8736
        },
        "v0.8.23+commit.f704f362": {
          "contracts": 77,
          "instructions": 417521,
          "known_instructions": 170900,
          "unknown_instructions": 123541,
          "failed_instructions": 123080
        },
        "v0.8.19+commit.7dd6d404": {
          "contracts": 163,
          "instructions": 795151,
          "known_instructions": 462137,
          "unknown_instructions": 251081,
          "failed_instructions": 81933
        },
        "v0.8.9+commit.e5eed63a": {
          "contracts": 133,
          "instructions": 808752,
          "known_instructions": 482305,
          "unknown_instructions": 254588,
          "failed_instructions": 71859
        },
        "v0.8.7+commit.e28d00a7": {
          "contracts": 65,
          "instructions": 341564,
          "known_instructions": 194428,
          "unknown_instructions": 113995,
          "failed_instructions": 33141
        },
        "v0.8.24+commit.e11b9ed9": {
          "contracts": 95,
          "instructions": 421176,
          "known_instructions": 167749,
          "unknown_instructions": 138522,
          "failed_instructions": 114905
        },
        "v0.6.12+commit.27d51765": {
          "contracts": 43,
          "instructions": 182375,
          "known_instructions": 96460,
          "unknown_instructions": 56284,
          "failed_instructions": 29631
        },
        "v0.4.16-nightly.2017.8.16+commit.83561e13": {
          "contracts": 1,
          "instructions": 1117,
          "known_instructions": 636,
          "unknown_instructions": 344,
          "failed_instructions": 137
        },
        "v0.4.22+commit.4cb486ee": {
          "contracts": 11,
          "instructions": 26248,
          "known_instructions": 15046,
          "unknown_instructions": 6721,
          "failed_instructions": 4481
        },
        "v0.4.13+commit.fb4cb1a": {
          "contracts": 19,
          "instructions": 40333,
          "known_instructions": 24231,
          "unknown_instructions": 11396,
          "failed_instructions": 4706
        },
        "v0.4.11+commit.68ef5810": {
          "contracts": 20,
          "instructions": 55022,
          "known_instructions": 32926,
          "unknown_instructions": 15020,
          "failed_instructions": 7076
        },
        "v0.4.21+commit.dfe3193c": {
          "contracts": 86,
          "instructions": 246089,
          "known_instructions": 137422,
          "unknown_instructions": 65589,
          "failed_instructions": 43078
        },
        "v0.4.18+commit.9cf6e910": {
          "contracts": 102,
          "instructions": 294773,
          "known_instructions": 170142,
          "unknown_instructions": 77648,
          "failed_instructions": 46983
        },
        "v0.7.6+commit.7338295f": {
          "contracts": 17,
          "instructions": 103886,
          "known_instructions": 53779,
          "unknown_instructions": 37184,
          "failed_instructions": 12923
        },
        "v0.8.16+commit.07a7930e": {
          "contracts": 52,
          "instructions": 270955,
          "known_instructions": 161481,
          "unknown_instructions": 85033,
          "failed_instructions": 24441
        },
        "v0.8.18+commit.87f61d96": {
          "contracts": 100,
          "instructions": 573207,
          "known_instructions": 322847,
          "unknown_instructions": 199605,
          "failed_instructions": 50755
        },
        "v0.4.25+commit.59dbf8f1": {
          "contracts": 137,
          "instructions": 386447,
          "known_instructions": 215045,
          "unknown_instructions": 103427,
          "failed_instructions": 67975
        },
        "v0.4.19+commit.c4cbbb05": {
          "contracts": 92,
          "instructions": 275046,
          "known_instructions": 156244,
          "unknown_instructions": 74745,
          "failed_instructions": 44057
        },
        "v0.8.17+commit.8df45f5f": {
          "contracts": 184,
          "instructions": 1032787,
          "known_instructions": 603918,
          "unknown_instructions": 316750,
          "failed_instructions": 112119
        },
        "v0.4.24+commit.e67f0147": {
          "contracts": 236,
          "instructions": 730581,
          "known_instructions": 411952,
          "unknown_instructions": 195312,
          "failed_instructions": 123317
        },
        "v0.8.0+commit.c7dfd78e": {
          "contracts": 15,
          "instructions": 76224,
          "known_instructions": 40015,
          "unknown_instructions": 27436,
          "failed_instructions": 8773
        },
        "v0.8.20+commit.a1b79de6": {
          "contracts": 92,
          "instructions": 390237,
          "known_instructions": 165868,
          "unknown_instructions": 113303,
          "failed_instructions": 111066
        },
        "v0.8.2+commit.661d1103": {
          "contracts": 7,
          "instructions": 17635,
          "known_instructions": 9042,
          "unknown_instructions": 7322,
          "failed_instructions": 1271
        },
        "v0.8.4+commit.c7e474f2": {
          "contracts": 28,
          "instructions": 117057,
          "known_instructions": 66123,
          "unknown_instructions": 39100,
          "failed_instructions": 11834
        },
        "v0.4.23+commit.124ca40d": {
          "contracts": 31,
          "instructions": 83906,
          "known_instructions": 47644,
          "unknown_instructions": 21658,
          "failed_instructions": 14604
        },
        "v0.4.20-nightly.2018.1.4+commit.a0771691": {
          "contracts": 2,
          "instructions": 2468,
          "known_instructions": 1401,
          "unknown_instructions": 682,
          "failed_instructions": 385
        },
        "v0.4.16+commit.d7661dd9": {
          "contracts": 26,
          "instructions": 63157,
          "known_instructions": 36178,
          "unknown_instructions": 16709,
          "failed_instructions": 10270
        },
        "v0.8.25+commit.b61c2a91": {
          "contracts": 5,
          "instructions": 28070,
          "known_instructions": 13323,
          "unknown_instructions": 8920,
          "failed_instructions": 5827
        },
        "v0.4.26+commit.4563c3fc": {
          "contracts": 20,
          "instructions": 42259,
          "known_instructions": 23251,
          "unknown_instructions": 12065,
          "failed_instructions": 6943
        },
        "v0.8.15+commit.e14f2714": {
          "contracts": 31,
          "instructions": 224916,
          "known_instructions": 126538,
          "unknown_instructions": 60647,
          "failed_instructions": 37731
        },
        "v0.4.25-nightly.2018.5.16+commit.3897c367": {
          "contracts": 1,
          "instructions": 1916,
          "known_instructions": 1017,
          "unknown_instructions": 524,
          "failed_instructions": 375
        },
        "v0.4.20+commit.3155dd80": {
          "contracts": 35,
          "instructions": 107052,
          "known_instructions": 57533,
          "unknown_instructions": 31310,
          "failed_instructions": 18209
        },
        "v0.4.15+commit.bbb8e64f": {
          "contracts": 12,
          "instructions": 35152,
          "known_instructions": 22426,
          "unknown_instructions": 8284,
          "failed_instructions": 4442
        },
        "v0.8.21+commit.d9974bed": {
          "contracts": 12,
          "instructions": 52591,
          "known_instructions": 21994,
          "unknown_instructions": 14703,
          "failed_instructions": 15894
        },
        "v0.8.5+commit.a4f2e591": {
          "contracts": 9,
          "instructions": 44076,
          "known_instructions": 24223,
          "unknown_instructions": 14776,
          "failed_instructions": 5077
        },
        "v0.6.6+commit.6c089d02": {
          "contracts": 4,
          "instructions": 18500,
          "known_instructions": 8700,
          "unknown_instructions": 7388,
          "failed_instructions": 2412
        },
        "v0.8.12+commit.f00d7308": {
          "contracts": 11,
          "instructions": 78748,
          "known_instructions": 46642,
          "unknown_instructions": 25223,
          "failed_instructions": 6883
        },
        "v0.4.3-nightly.2016.10.15+commit.482807f6": {
          "contracts": 1,
          "instructions": 1249,
          "known_instructions": 653,
          "unknown_instructions": 348,
          "failed_instructions": 248
        },
        "v0.5.0+commit.1d4f565a": {
          "contracts": 8,
          "instructions": 18000,
          "known_instructions": 10169,
          "unknown_instructions": 5918,
          "failed_instructions": 1913
        },
        "v0.4.24-nightly.2018.5.10+commit.85d417a8": {
          "contracts": 1,
          "instructions": 1188,
          "known_instructions": 833,
          "unknown_instructions": 225,
          "failed_instructions": 130
        },
        "v0.5.17+commit.d19bba13": {
          "contracts": 12,
          "instructions": 31108,
          "known_instructions": 15629,
          "unknown_instructions": 10627,
          "failed_instructions": 4852
        },
        "v0.4.10+commit.f0d539ae": {
          "contracts": 2,
          "instructions": 4440,
          "known_instructions": 2175,
          "unknown_instructions": 1264,
          "failed_instructions": 1001
        },
        "v0.4.19-nightly.2017.10.19+commit.c58d9d2c": {
          "contracts": 1,
          "instructions": 2683,
          "known_instructions": 1424,
          "unknown_instructions": 930,
          "failed_instructions": 329
        },
        "v0.6.0+commit.26b70077": {
          "contracts": 2,
          "instructions": 6261,
          "known_instructions": 3321,
          "unknown_instructions": 2182,
          "failed_instructions": 758
        },
        "v0.8.13+commit.abaa5c0e": {
          "contracts": 22,
          "instructions": 128715,
          "known_instructions": 71204,
          "unknown_instructions": 41418,
          "failed_instructions": 16093
        },
        "v0.8.8+commit.dddeac2f": {
          "contracts": 2,
          "instructions": 11349,
          "known_instructions": 7273,
          "unknown_instructions": 3010,
          "failed_instructions": 1066
        },
        "v0.4.8+commit.60cc1668": {
          "contracts": 1,
          "instructions": 2143,
          "known_instructions": 1382,
          "unknown_instructions": 617,
          "failed_instructions": 144
        },
        "v0.4.17+commit.bdeb9e52": {
          "contracts": 9,
          "instructions": 19039,
          "known_instructions": 10402,
          "unknown_instructions": 5748,
          "failed_instructions": 2889
        },
        "v0.5.8+commit.23d335f2": {
          "contracts": 1,
          "instructions": 7729,
          "known_instructions": 4096,
          "unknown_instructions": 2037,
          "failed_instructions": 1596
        },
        "v0.4.14+commit.c2215d46": {
          "contracts": 5,
          "instructions": 10907,
          "known_instructions": 6288,
          "unknown_instructions": 3022,
          "failed_instructions": 1597
        },
        "v0.4.24-nightly.2018.5.11+commit.43803b1a": {
          "contracts": 2,
          "instructions": 8430,
          "known_instructions": 5045,
          "unknown_instructions": 2157,
          "failed_instructions": 1228
        },
        "v0.6.9+commit.3e3065ac": {
          "contracts": 2,
          "instructions": 3053,
          "known_instructions": 1346,
          "unknown_instructions": 1287,
          "failed_instructions": 420
        },
        "v0.8.22+commit.4fc1097e": {
          "contracts": 15,
          "instructions": 69980,
          "known_instructions": 27939,
          "unknown_instructions": 21719,
          "failed_instructions": 20322
        },
        "v0.4.25-nightly.2018.5.18+commit.4d7b092c": {
          "contracts": 1,
          "instructions": 4442,
          "known_instructions": 3117,
          "unknown_instructions": 849,
          "failed_instructions": 476
        },
        "v0.7.5+commit.eb77ed08": {
          "contracts": 5,
          "instructions": 11793,
          "known_instructions": 6059,
          "unknown_instructions": 4015,
          "failed_instructions": 1719
        },
        "v0.4.21-nightly.2018.3.1+commit.cf6720ea": {
          "contracts": 1,
          "instructions": 2131,
          "known_instructions": 1165,
          "unknown_instructions": 513,
          "failed_instructions": 453
        },
        "v0.7.0+commit.9e61f92b": {
          "contracts": 3,
          "instructions": 21080,
          "known_instructions": 11009,
          "unknown_instructions": 7246,
          "failed_instructions": 2825
        },
        "v0.7.1+commit.f4a555be": {
          "contracts": 1,
          "instructions": 7119,
          "known_instructions": 4125,
          "unknown_instructions": 2053,
          "failed_instructions": 941
        },
        "v0.4.9-nightly.2017.1.17+commit.6ecb4aa3": {
          "contracts": 1,
          "instructions": 702,
          "known_instructions": 449,
          "unknown_instructions": 138,
          "failed_instructions": 115
        },
        "v0.4.12+commit.194ff033": {
          "contracts": 4,
          "instructions": 7385,
          "known_instructions": 4482,
          "unknown_instructions": 1910,
          "failed_instructions": 993
        },
        "v0.8.14+commit.80d49f37": {
          "contracts": 8,
          "instructions": 52532,
          "known_instructions": 31002,
          "unknown_instructions": 17256,
          "failed_instructions": 4274
        },
        "v0.4.19-nightly.2017.11.22+commit.f22ac8fc": {
          "contracts": 1,
          "instructions": 3115,
          "known_instructions": 1831,
          "unknown_instructions": 720,
          "failed_instructions": 564
        },
        "v0.4.19-nightly.2017.11.11+commit.284c3839": {
          "contracts": 3,
          "instructions": 6396,
          "known_instructions": 3462,
          "unknown_instructions": 1871,
          "failed_instructions": 1063
        },
        "v0.5.16+commit.9c3226ce": {
          "contracts": 4,
          "instructions": 10201,
          "known_instructions": 5882,
          "unknown_instructions": 2905,
          "failed_instructions": 1414
        },
        "v0.4.24-nightly.2018.4.25+commit.81cca26f": {
          "contracts": 1,
          "instructions": 2318,
          "known_instructions": 1356,
          "unknown_instructions": 563,
          "failed_instructions": 399
        },
        "v0.4.22-nightly.2018.3.21+commit.8fd53c1c": {
          "contracts": 1,
          "instructions": 3109,
          "known_instructions": 1577,
          "unknown_instructions": 910,
          "failed_instructions": 622
        },
        "v0.8.6+commit.11564f7e": {
          "contracts": 4,
          "instructions": 19052,
          "known_instructions": 11344,
          "unknown_instructions": 5754,
          "failed_instructions": 1954
        },
        "v0.4.22-nightly.2018.3.13+commit.f2614be9": {
          "contracts": 2,
          "instructions": 4969,
          "known_instructions": 3046,
          "unknown_instructions": 1206,
          "failed_instructions": 717
        },
        "v0.4.12-nightly.2017.5.30+commit.254b5572": {
          "contracts": 1,
          "instructions": 486,
          "known_instructions": 421,
          "unknown_instructions": 41,
          "failed_instructions": 24
        },
        "v0.4.24-nightly.2018.5.16+commit.7f965c86": {
          "contracts": 1,
          "instructions": 1439,
          "known_instructions": 761,
          "unknown_instructions": 444,
          "failed_instructions": 234
        },
        "v0.4.15-nightly.2017.8.7+commit.212454a9": {
          "contracts": 1,
          "instructions": 528,
          "known_instructions": 302,
          "unknown_instructions": 211,
          "failed_instructions": 15
        },
        "v0.4.20-nightly.2018.1.6+commit.2548228b": {
          "contracts": 2,
          "instructions": 4236,
          "known_instructions": 2221,
          "unknown_instructions": 1391,
          "failed_instructions": 624
        },
        "v0.4.25-nightly.2018.7.2+commit.a5608b31": {
          "contracts": 1,
          "instructions": 2034,
          "known_instructions": 1230,
          "unknown_instructions": 491,
          "failed_instructions": 313
        },
        "v0.8.3+commit.8d00100c": {
          "contracts": 2,
          "instructions": 14765,
          "known_instructions": 8605,
          "unknown_instructions": 4768,
          "failed_instructions": 1392
        },
        "v0.4.19-nightly.2017.10.18+commit.f7ca2421": {
          "contracts": 1,
          "instructions": 1586,
          "known_instructions": 939,
          "unknown_instructions": 472,
          "failed_instructions": 175
        },
        "v0.4.20-nightly.2018.1.19+commit.eba46a65": {
          "contracts": 1,
          "instructions": 2932,
          "known_instructions": 1616,
          "unknown_instructions": 771,
          "failed_instructions": 545
        },
        "v0.4.21-nightly.2018.3.6+commit.a9e02acc": {
          "contracts": 1,
          "instructions": 2973,
          "known_instructions": 1575,
          "unknown_instructions": 760,
          "failed_instructions": 638
        },
        "v0.4.25-nightly.2018.6.12+commit.56a965ea": {
          "contracts": 1,
          "instructions": 2895,
          "known_instructions": 1458,
          "unknown_instructions": 875,
          "failed_instructions": 562
        },
        "v0.7.4+commit.3f05b770": {
          "contracts": 1,
          "instructions": 5743,
          "known_instructions": 2716,
          "unknown_instructions": 2148,
          "failed_instructions": 879
        },
        "v0.4.25-nightly.2018.6.14+commit.baeabe1c": {
          "contracts": 1,
          "instructions": 2047,
          "known_instructions": 1040,
          "unknown_instructions": 615,
          "failed_instructions": 392
        },
        "v0.4.24-nightly.2018.4.27+commit.1604a996": {
          "contracts": 1,
          "instructions": 5503,
          "known_instructions": 2604,
          "unknown_instructions": 1479,
          "failed_instructions": 1420
        },
        "v0.6.7+commit.b8d736ae": {
          "contracts": 2,
          "instructions": 4512,
          "known_instructions": 2661,
          "unknown_instructions": 1256,
          "failed_instructions": 595
        },
        "v0.4.20-nightly.2018.1.24+commit.b177352a": {
          "contracts": 1,
          "instructions": 2051,
          "known_instructions": 1036,
          "unknown_instructions": 648,
          "failed_instructions": 367
        },
        "v0.4.9+commit.364da425": {
          "contracts": 1,
          "instructions": 681,
          "known_instructions": 373,
          "unknown_instructions": 83,
          "failed_instructions": 225
        },
        "v0.4.6+commit.2dabbdf0": {
          "contracts": 2,
          "instructions": 2023,
          "known_instructions": 1345,
          "unknown_instructions": 465,
          "failed_instructions": 213
        },
        "v0.5.11+commit.c082d0b4": {
          "contracts": 2,
          "instructions": 5579,
          "known_instructions": 2931,
          "unknown_instructions": 1748,
          "failed_instructions": 900
        },
        "v0.7.3+commit.9bfce1f6": {
          "contracts": 1,
          "instructions": 5762,
          "known_instructions": 3160,
          "unknown_instructions": 1638,
          "failed_instructions": 964
        },
        "v0.5.12+commit.7709ece9": {
          "contracts": 1,
          "instructions": 15456,
          "known_instructions": 9119,
          "unknown_instructions": 4166,
          "failed_instructions": 2171
        },
        "v0.8.1+commit.df193b15": {
          "contracts": 1,
          "instructions": 1029,
          "known_instructions": 525,
          "unknown_instructions": 394,
          "failed_instructions": 110
        },
        "v0.4.25-nightly.2018.6.22+commit.9b67bdb3": {
          "contracts": 1,
          "instructions": 1891,
          "known_instructions": 990,
          "unknown_instructions": 501,
          "failed_instructions": 400
        },
        "v0.4.24-nightly.2018.4.24+commit.258ae892": {
          "contracts": 1,
          "instructions": 2130,
          "known_instructions": 1100,
          "unknown_instructions": 664,
          "failed_instructions": 366
        },
        "v0.4.19-nightly.2017.10.29+commit.eb140bc6": {
          "contracts": 1,
          "instructions": 1453,
          "known_instructions": 795,
          "unknown_instructions": 414,
          "failed_instructions": 244
        },
        "v0.4.16-nightly.2017.8.15+commit.dca1f45c": {
          "contracts": 1,
          "instructions": 1562,
          "known_instructions": 817,
          "unknown_instructions": 507,
          "failed_instructions": 238
        },
        "v0.4.17-nightly.2017.8.25+commit.e945f458": {
          "contracts": 1,
          "instructions": 4206,
          "known_instructions": 2833,
          "unknown_instructions": 1005,
          "failed_instructions": 368
        },
        "v0.4.21-nightly.2018.2.23+commit.cae6cc2c": {
          "contracts": 2,
          "instructions": 2632,
          "known_instructions": 1653,
          "unknown_instructions": 609,
          "failed_instructions": 370
        },
        "v0.6.8+commit.0bbfe453": {
          "contracts": 1,
          "instructions": 2094,
          "known_instructions": 1189,
          "unknown_instructions": 575,
          "failed_instructions": 330
        },
        "v0.4.20-nightly.2018.2.13+commit.27ef9794": {
          "contracts": 1,
          "instructions": 2051,
          "known_instructions": 1036,
          "unknown_instructions": 648,
          "failed_instructions": 367
        }
      },
      "failure_examples": [
        {
          "contract_id": "10108",
          "errors": {
            "unknown_opcode_annotation": 18,
            "push_immediate_width_mismatch": 1
          }
        },
        {
          "contract_id": "11654",
          "errors": {
            "unknown_opcode_annotation": 8
          }
        },
        {
          "contract_id": "13834",
          "errors": {
            "unknown_opcode_annotation": 4
          }
        },
        {
          "contract_id": "11642",
          "errors": {
            "unknown_opcode_annotation": 3
          }
        },
        {
          "contract_id": "1064",
          "errors": {
            "unknown_opcode_annotation": 11
          }
        },
        {
          "contract_id": "16570",
          "errors": {
            "unknown_opcode_annotation": 6
          }
        },
        {
          "contract_id": "19861",
          "errors": {
            "unknown_opcode_annotation": 8
          }
        },
        {
          "contract_id": "9645",
          "errors": {
            "unknown_opcode_annotation": 14
          }
        },
        {
          "contract_id": "16127",
          "errors": {
            "unknown_opcode_annotation": 2
          }
        },
        {
          "contract_id": "14766",
          "errors": {
            "unknown_opcode_annotation": 17
          }
        },
        {
          "contract_id": "1906",
          "errors": {
            "unknown_opcode_annotation": 14
          }
        },
        {
          "contract_id": "14621",
          "errors": {
            "unknown_opcode_annotation": 9
          }
        },
        {
          "contract_id": "2936",
          "errors": {
            "unknown_opcode_annotation": 11
          }
        },
        {
          "contract_id": "19280",
          "errors": {
            "unknown_opcode_annotation": 8
          }
        },
        {
          "contract_id": "12369",
          "errors": {
            "unknown_opcode_annotation": 5
          }
        },
        {
          "contract_id": "16569",
          "errors": {
            "unknown_opcode_annotation": 9
          }
        },
        {
          "contract_id": "15165",
          "errors": {
            "unknown_opcode_annotation": 6
          }
        },
        {
          "contract_id": "11156",
          "errors": {
            "unknown_opcode_annotation": 40
          }
        },
        {
          "contract_id": "3119",
          "errors": {
            "unknown_opcode_annotation": 19
          }
        },
        {
          "contract_id": "10507",
          "errors": {
            "unknown_opcode_annotation": 3
          }
        },
        {
          "contract_id": "13710",
          "errors": {
            "unknown_opcode_annotation": 18
          }
        },
        {
          "contract_id": "3781",
          "errors": {
            "unknown_opcode_annotation": 14
          }
        },
        {
          "contract_id": "16568",
          "errors": {
            "unknown_opcode_annotation": 10
          }
        },
        {
          "contract_id": "21149",
          "errors": {
            "unknown_opcode_annotation": 4
          }
        },
        {
          "contract_id": "5793",
          "errors": {
            "unknown_opcode_annotation": 9
          }
        },
        {
          "contract_id": "3244",
          "errors": {
            "unknown_opcode_annotation": 12
          }
        },
        {
          "contract_id": "12371",
          "errors": {
            "unknown_opcode_annotation": 8
          }
        },
        {
          "contract_id": "3722",
          "errors": {
            "unknown_opcode_annotation": 1
          }
        },
        {
          "contract_id": "9391",
          "errors": {
            "unknown_opcode_annotation": 7
          }
        },
        {
          "contract_id": "1740",
          "errors": {
            "unknown_opcode_annotation": 5,
            "push_immediate_width_mismatch": 1
          }
        },
        {
          "contract_id": "3733",
          "errors": {
            "unknown_opcode_annotation": 2
          }
        },
        {
          "contract_id": "4863",
          "errors": {
            "unknown_opcode_annotation": 5
          }
        },
        {
          "contract_id": "19670",
          "errors": {
            "unknown_opcode_annotation": 10
          }
        },
        {
          "contract_id": "6080",
          "errors": {
            "unknown_opcode_annotation": 5
          }
        },
        {
          "contract_id": "14502",
          "errors": {
            "unknown_opcode_annotation": 16
          }
        },
        {
          "contract_id": "939",
          "errors": {
            "unknown_opcode_annotation": 6
          }
        },
        {
          "contract_id": "4613",
          "errors": {
            "unknown_opcode_annotation": 9
          }
        },
        {
          "contract_id": "17075",
          "errors": {
            "unknown_opcode_annotation": 9
          }
        },
        {
          "contract_id": "1759",
          "errors": {
            "unknown_opcode_annotation": 13
          }
        },
        {
          "contract_id": "11218",
          "errors": {
            "unknown_opcode_annotation": 28
          }
        },
        {
          "contract_id": "18281",
          "errors": {
            "unknown_opcode_annotation": 2
          }
        },
        {
          "contract_id": "7793",
          "errors": {
            "unknown_opcode_annotation": 2
          }
        },
        {
          "contract_id": "3221",
          "errors": {
            "unknown_opcode_annotation": 14
          }
        },
        {
          "contract_id": "11100",
          "errors": {
            "unknown_opcode_annotation": 9
          }
        },
        {
          "contract_id": "7646",
          "errors": {
            "unknown_opcode_annotation": 9
          }
        },
        {
          "contract_id": "11352",
          "errors": {
            "unknown_opcode_annotation": 6
          }
        },
        {
          "contract_id": "907",
          "errors": {
            "unknown_opcode_annotation": 7
          }
        },
        {
          "contract_id": "14299",
          "errors": {
            "unknown_opcode_annotation": 6
          }
        },
        {
          "contract_id": "892",
          "errors": {
            "unknown_opcode_annotation": 9
          }
        },
        {
          "contract_id": "16827",
          "errors": {
            "unknown_opcode_annotation": 13
          }
        }
      ],
      "processing_seconds": {
        "mean": 0.05391609046129984,
        "p50": 0.04755849999492057,
        "p95": 0.1291449400014244
      },
      "cache_bytes": 406073063,
      "total_seconds": 123.57943169999635
    }
  },
  "runtime_bytecode_available": false,
  "test_checked": false,
  "scope": "disassembled-opcode-based basic-block-local reconstruction; no cross-block execution claim"
}
