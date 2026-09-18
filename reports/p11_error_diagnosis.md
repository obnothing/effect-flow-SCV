# P11 Error Diagnosis

Validation-only analysis; frozen checkpoint and thresholds; test_checked=false.

## Per-label confusion matrices

| Label | TN | FP | FN | TP | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|
| Reentrancy | 958 | 244 | 56 | 774 | 0.2030 | 0.0675 |
| Access Control | 631 | 261 | 59 | 1081 | 0.2926 | 0.0518 |
| Arithmetic | 1105 | 214 | 74 | 639 | 0.1622 | 0.1038 |
| Unchecked Return Values | 1511 | 87 | 45 | 389 | 0.0544 | 0.1037 |
| DoS | 1696 | 72 | 61 | 203 | 0.0407 | 0.2311 |
| Time manipulation | 1393 | 88 | 61 | 490 | 0.0594 | 0.1107 |

## Error-polarity means

| Label | TP e+ | TP e- | FN e+ | FN e- | TN e+ | TN e- | FP e+ | FP e- |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Reentrancy | 2.045 | -1.618 | -0.157 | 2.724 | -6.755 | 6.979 | 1.068 | -0.235 |
| Access Control | 2.334 | -2.451 | -0.455 | 1.821 | -7.086 | 8.004 | 1.132 | -0.834 |
| Arithmetic | 1.577 | -1.770 | -1.961 | 0.676 | -7.174 | 7.894 | 0.864 | -0.918 |
| Unchecked Return Values | 3.298 | -1.445 | -2.068 | 1.653 | -7.312 | 4.455 | 2.284 | -0.625 |
| DoS | 2.918 | -1.867 | -2.825 | 0.555 | -5.226 | 2.198 | 1.557 | -1.487 |
| Time manipulation | 2.477 | -2.659 | -1.226 | 3.215 | -5.921 | 10.363 | 0.756 | -0.664 |

## Calibration

| Label | Brier | ECE | Adaptive ECE |
|---|---:|---:|---:|
| Reentrancy | 0.1040 | 0.0547 | 0.0470 |
| Access Control | 0.1175 | 0.0672 | 0.0666 |
| Arithmetic | 0.1074 | 0.0643 | 0.0601 |
| Unchecked Return Values | 0.0575 | 0.0455 | 0.0415 |
| DoS | 0.0574 | 0.0450 | 0.0426 |
| Time manipulation | 0.0574 | 0.0366 | 0.0284 |

## Selective risk

| Coverage | Error rate | Mean decision confidence |
|---:|---:|---:|
| 1.0 | 0.1084 | 0.8817 |
| 0.9 | 0.0763 | 0.9468 |
| 0.8 | 0.0512 | 0.9782 |
| 0.7 | 0.0350 | 0.9919 |
| 0.5 | 0.0180 | 0.9992 |
| 0.3 | 0.0055 | 1.0000 |

## Counterfactual attention evidence

Positive Top-5 masking Macro-F1 drop: `-0.075362`
Negative Top-5 masking Macro-F1 drop: `-0.095656`
Matched random-5 masking drop: `-0.004241`

## Interpretation boundary

Low label-quality scores identify contracts for manual review; they do not prove label errors. Attention statistics are interpreted only together with counterfactual masking, not as standalone explanations.

## Method basis and findings

The diagnosis combines five suitable methods for a frozen long-sequence multi-label model.

1. **Multi-label confusion analysis.** Each vulnerability is treated as an independent binary decision, while cross-label conditional FP/FN matrices identify interactions with the other true labels.
2. **Polarity decomposition.** TP, FP, FN, and TN decisions are grouped by `e+`, `e-`, and `s=e+-e-`. This directly tests whether an error is a weak threshold crossing or a reversal of both polarity branches.
3. **Probability calibration and selective risk.** Per-label Brier score, fixed-bin ECE, adaptive ECE, and risk-coverage curves evaluate whether confidence distinguishes reliable from unreliable decisions. These follow calibration and selective-classification practice [Guo et al., ICML 2017; Geifman and El-Yaniv, NeurIPS 2017].
4. **Label-review candidate screening.** The score `p` for a positive label and `1-p` for a negative label ranks low self-confidence contract-label pairs. This is a confident-learning-inspired review aid [Northcutt et al., JAIR 2021], not proof of annotation error.
5. **Attention faithfulness by perturbation.** Attention is not treated as explanation by itself [Jain and Wallace, NAACL 2019]. The companion causal diagnosis therefore masks Top-5 positive/negative attention tokens and compares against matched random masking.

### Main error mechanisms

- **Polarity reversal dominates confident errors.** For false negatives, the fractions in which both branches disagree with the true positive polarity are 0.518 (Reentrancy), 0.525 (Access Control), 0.676 (Arithmetic), 0.689 (Unchecked Return Values), 0.508 (DoS), and 0.770 (Time manipulation). For false positives, the corresponding active false-vulnerability pattern occurs in 0.459, 0.625, 0.519, 0.655, 0.583, and 0.534 of cases. Thus most errors are not marginal threshold mistakes.
- **Error type depends on label.** DoS has the highest FNR (0.2311), so missed DoS evidence is the primary recall problem. Access Control has the highest FPR (0.2926), followed by Reentrancy (0.2030), so those labels are more prone to false vulnerability evidence.
- **Cross-label confusion is material.** Access Control false-positive rates exceed 0.50 when Time manipulation or DoS is truly present. Reentrancy false positives are highest when Time manipulation is present (0.3755). DoS false negatives are highest when Access Control (0.2361) or Unchecked Return Values (0.2077) co-occur.
- **Long and single-label contracts are harder.** Contracts of length 4097–8192 have Hamming error 0.1349 and exact-match rate 0.5845, compared with 0.0677 and 0.7352 for lengths 1025–2048. Single-label contracts have the highest Hamming error (0.1453), whereas contracts with four or more labels have 0.0812. This is consistent with the model benefiting from recurrent co-occurrence patterns; it does not establish a causal source-code mechanism.
- **Calibration is imperfect but not the sole cause.** ECE is highest for Access Control (0.0672) and Arithmetic (0.0643). Selective risk falls from 0.1084 at full coverage to 0.0180 at 50% coverage, but 521 high-confidence errors remain. Threshold adjustment alone would therefore not resolve the dominant failures.
- **Attention is decision-relevant but not sufficient to explain every error.** Matched random Top-5 masking reduces Macro-F1 by only 0.0042, versus 0.0754 and 0.0957 for positive/negative Top-5 masking. Nevertheless, attention entropy and positive-negative overlap do not separate every TP/FP/FN/TN category consistently; token-level causal masking and manual review remain necessary.

### Review candidates

`label_review_candidates.csv` contains 100 lowest self-confidence contract-label pairs, including their true/predicted label, probability, `e+`, `e-`, margin, contract length, and positive/negative Top-5 decision-associated opcode tokens. The largest candidate groups are Access Control (21), Time manipulation (20), DoS (20), and Unchecked Return Values (19). These should be manually audited before any claim of dataset-label noise.
