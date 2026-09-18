# P11 Causal Support Diagnosis

Inference-only; process01 validation; seed 42; frozen P11 checkpoint and thresholds; test_checked=false.

| Condition | Macro-F1 | Delta vs normal | Micro-F1 |
|---|---:|---:|---:|
| normal | 0.833507 | +0.000000 | 0.843993 |
| average_support | 0.374155 | -0.459352 | 0.536318 |
| swapped_support | 0.105615 | -0.727892 | 0.106071 |
| mask_positive_top5 | 0.758146 | -0.075362 | 0.775127 |
| mask_negative_top5 | 0.737852 | -0.095656 | 0.763054 |
| mask_union | 0.669634 | -0.163873 | 0.693383 |
| mask_random5 | 0.829266 | -0.004241 | 0.841409 |
| mask_random_union | 0.826489 | -0.007018 | 0.838824 |

## Per-label branch complementarity

| Label | Positive Top-5 drop | Negative Top-5 drop | Union drop | Union excess | Dominant branch |
|---|---:|---:|---:|---:|---|
| Reentrancy | +0.063110 | +0.078041 | +0.159292 | +0.081251 | negative |
| Access Control | +0.053566 | +0.034425 | +0.092417 | +0.038851 | positive |
| Arithmetic | +0.112529 | +0.041589 | +0.140993 | +0.028464 | positive |
| Unchecked Return Values | +0.078105 | +0.093499 | +0.157028 | +0.063529 | negative |
| DoS | +0.111223 | +0.066568 | +0.150134 | +0.038911 | positive |
| Time manipulation | +0.033636 | +0.259812 | +0.283375 | +0.023563 | negative |

## Findings

1. Positive and negative supports are causally non-interchangeable. Swapping them reverses the sign of every label's mean margin separation and reduces Macro-F1 by 0.727892.
2. Replacing both supports with their average collapses all sample-dependent margin separation to approximately zero. Because P11 shares one scorer between polarities, identical evidence representations cancel in `e+ - e-`, leaving almost only branch bias.
3. Positive Top-5 masking reduces Macro-F1 by 0.075362 and negative Top-5 masking by 0.095656, while matched random masking reduces it by only 0.004241.
4. Masking the union reduces Macro-F1 by 0.163873, compared with 0.007018 for a random union-size mask. For all labels, the union drop exceeds the stronger individual-branch drop, demonstrating complementary evidence support.
5. Positive support is more influential for Access Control, Arithmetic, and DoS. Negative support is more influential for Reentrancy, Unchecked Return Values, and especially Time manipulation.
6. Margin separation alone is not sufficient to assess a masked model. Some masking conditions make negative examples more negative while simultaneously destroying positive calibration; frozen-threshold F1 captures this failure whereas the difference of class-conditional means may increase.

## Conclusion

P11 needs two independently localized evidence regions. The attention difference is functional specialization rather than harmful misalignment. A future architecture may add a shared relevance prior, but it must preserve private positive and negative token retrieval; forcing a single support is structurally incompatible with P11's shared-scorer competition.
