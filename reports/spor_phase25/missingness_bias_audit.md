# SPOR Missingness Bias Audit

Classifier inputs contain only per-contract availability ratios and optional sequence length. No opcode identity, provenance category, source code, or model embedding is used. ROC-AUC and PR-AUC are measured on valid; the comparison dummy has ROC-AUC 0.5 and PR-AUC equal to label prevalence.

| Label | Valid prevalence | ROC-AUC no length | PR-AUC no length | ROC-AUC + length | PR-AUC + length | Dummy PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Reentrancy | 0.5101 | 0.7116 | 0.6515 | 0.7482 | 0.7016 | 0.5101 |
| Access Control | 0.7497 | 0.6287 | 0.8218 | 0.6470 | 0.8381 | 0.7497 |
| Arithmetic | 0.4254 | 0.6690 | 0.5736 | 0.7006 | 0.6340 | 0.4254 |
| Unchecked Return Values | 0.2651 | 0.7218 | 0.4422 | 0.7954 | 0.5814 | 0.2651 |
| DoS | 0.1679 | 0.6802 | 0.3047 | 0.7109 | 0.3321 | 0.1679 |
| Bad Randomness | 0.0282 | 0.6364 | 0.0481 | 0.6323 | 0.0730 | 0.0282 |
| Front Running | 0.0278 | 0.8005 | 0.1149 | 0.8003 | 0.1237 | 0.0278 |
| Time manipulation | 0.2853 | 0.6939 | 0.4680 | 0.7281 | 0.5477 | 0.2853 |

## Positive vs negative availability

Coverage vectors use the feature order shown in the JSON: overall valid ratio, UNKNOWN ratio, and known ratios for JUMPI/CALL/SLOAD/SSTORE.

| Split | Label | Positive n | Negative n | Positive availability vector | Negative availability vector |
|---|---|---:|---:|---|---|
| train | Reentrancy | 9,119 | 8,745 | `[0.5397, 0.1955, 0.616, 0.2369, 0.6908, 0.3259]` | `[0.5287, 0.2132, 0.6054, 0.098, 0.5261, 0.193]` |
| valid | Reentrancy | 1,139 | 1,094 | `[0.5406, 0.1947, 0.6189, 0.2302, 0.6887, 0.3232]` | `[0.5292, 0.2128, 0.6068, 0.1072, 0.5388, 0.2017]` |
| train | Access Control | 13,375 | 4,489 | `[0.5391, 0.1996, 0.6215, 0.1917, 0.6244, 0.2654]` | `[0.5201, 0.2177, 0.5789, 0.1011, 0.5676, 0.2473]` |
| valid | Access Control | 1,674 | 559 | `[0.5403, 0.1984, 0.6245, 0.1937, 0.6251, 0.2663]` | `[0.5194, 0.2193, 0.5785, 0.0989, 0.5859, 0.2558]` |
| train | Arithmetic | 7,634 | 10,230 | `[0.519, 0.2128, 0.5926, 0.1502, 0.6282, 0.2598]` | `[0.5457, 0.1977, 0.6244, 0.1829, 0.5967, 0.2617]` |
| valid | Arithmetic | 950 | 1,283 | `[0.5188, 0.2125, 0.5933, 0.1498, 0.6306, 0.2605]` | `[0.5471, 0.197, 0.6275, 0.1849, 0.6039, 0.266]` |
| train | Unchecked Return Values | 4,729 | 13,135 | `[0.5249, 0.2081, 0.5891, 0.1748, 0.6244, 0.2147]` | `[0.5377, 0.2027, 0.6186, 0.1668, 0.6051, 0.2775]` |
| valid | Unchecked Return Values | 592 | 1,641 | `[0.5267, 0.2068, 0.5945, 0.162, 0.6288, 0.2017]` | `[0.5381, 0.2025, 0.6196, 0.1728, 0.6104, 0.2861]` |
| train | DoS | 3,026 | 14,838 | `[0.5571, 0.1852, 0.6354, 0.337, 0.6325, 0.2899]` | `[0.5297, 0.208, 0.6058, 0.1346, 0.6056, 0.255]` |
| valid | DoS | 375 | 1,858 | `[0.5584, 0.1831, 0.6346, 0.3381, 0.6396, 0.2877]` | `[0.5303, 0.2077, 0.6086, 0.136, 0.6103, 0.2588]` |
| train | Bad Randomness | 509 | 17,355 | `[0.5425, 0.196, 0.6141, 0.2987, 0.6539, 0.3078]` | `[0.5341, 0.2044, 0.6107, 0.1651, 0.6089, 0.2595]` |
| valid | Bad Randomness | 63 | 2,170 | `[0.5323, 0.2032, 0.6044, 0.3015, 0.6533, 0.2721]` | `[0.5351, 0.2036, 0.6132, 0.1661, 0.6141, 0.2634]` |
| train | Front Running | 484 | 17,380 | `[0.5904, 0.1586, 0.7164, 0.3431, 0.6783, 0.3866]` | `[0.5327, 0.2054, 0.6079, 0.164, 0.6083, 0.2574]` |
| valid | Front Running | 62 | 2,171 | `[0.5996, 0.1497, 0.7266, 0.3572, 0.7117, 0.4149]` | `[0.5332, 0.2051, 0.6097, 0.1646, 0.6125, 0.2594]` |
| train | Time manipulation | 5,057 | 12,807 | `[0.5322, 0.1995, 0.5995, 0.2412, 0.651, 0.2952]` | `[0.5351, 0.206, 0.6153, 0.1404, 0.594, 0.2473]` |
| valid | Time manipulation | 637 | 1,596 | `[0.5342, 0.197, 0.608, 0.2527, 0.6684, 0.2962]` | `[0.5354, 0.2062, 0.615, 0.1369, 0.594, 0.2507]` |

The group audit includes compiler release and target EVM version separately, as well as length bins; train group counts: {'compiler': 181, 'evm_version': 10, 'length': 4}; valid group counts: {'compiler': 98, 'evm_version': 6, 'length': 4}.
Full compiler/EVM-version × length × split × label × polarity results are in `results/spor_phase25/coverage_by_label_compiler_length.json`.
