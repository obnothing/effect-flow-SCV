# Relevance–Polarity Final Report

Dataset: `DIVE_main6_opcode_process01`; seed 42; train/validation only; `test_checked=false`.

## Main results

| Variant | Localization | Polarity support | Tuned Macro-F1 | Fixed Macro-F1 | Micro-F1 | Detection-F1 | Params |
|---|---|---|---:|---:|---:|---:|---:|
| A0 P11 | q+ / q- independently | independent | 0.833507 | 0.827086 | 0.843993 | 0.931488 | 3,705,996 |
| B1 | q+ | shared aligned | 0.820963 | 0.815315 | 0.835221 | 0.927322 | 3,705,996 |
| B2 | dedicated q_rel | shared aligned | 0.826641 | 0.818253 | 0.837605 | 0.929777 | 3,710,604 |

B2 improves over B1 by 0.005678 Macro-F1, but remains 0.006866 below A0. B2 adds exactly 4,608 parameters, while B1 and A0 have identical parameter counts.

## Mechanism findings

1. **Does P11 exhibit evidence-support misalignment?** Yes descriptively: positive/negative attention cosine ranges from 0.249 to 0.545 and Top-5 overlap from 0.126 to 0.207. However, this separation is not shown to be harmful because A0 remains the strongest model.
2. **Is shared aligned support better than independent support?** No. Both aligned variants are below A0 in tuned and fixed Macro-F1. The shared-support constraint removes useful positional specialization.
3. **Is the positive query sufficient as a locator?** No. B2 exceeds B1 by 0.005678 Macro-F1, with the largest per-label recovery on DoS (+0.033180).
4. **Does dedicated q_rel contribute independently?** Yes relative to B1, but not enough to surpass A0. Its identity is meaningful: cyclic label shuffling reduces B2 Macro-F1 from 0.826641 to 0.665973 (-0.160668).
5. **Is q_rel label-specific?** Yes. Shuffling harms five labels strongly, and pairwise relevance distributions are not identical. The highest off-diagonal cosine is 0.964 for Access Control–Arithmetic, consistent with substantial shared code patterns.
6. **Do q+ and q- learn different polarity behavior?** Yes, although the gates remain highly similar (cosine 0.993–0.996). Query swapping collapses Macro-F1 to 0.116331, while positive/negative evidence representations have lower cosine (0.383–0.637). The branches are functional but polarity discrimination is expressed through relatively subtle gates.
7. **Is evidence competition better than one branch?** Yes. Full competition reaches 0.826641, compared with 0.811806 for positive-only and 0.813593 for negative-only under frozen thresholds.
8. **Can the B2 effect be explained by parameter count?** No. B2 adds only 4,608 parameters and remains below A0. B1 has exactly the same parameter count as A0 but also performs worse.
9. **Which labels benefit?** Relative to A0, B2 improves only Time manipulation (+0.006866). Reentrancy, Access Control, Arithmetic, Unchecked Return Values, and DoS decrease. Relative to B1, B2 mainly recovers DoS (+0.033180), Access Control (+0.003335), and Time manipulation (+0.003140).
10. **Should the study enter multi-seed confirmation?** No. B2 is 0.006866 below A0, exceeding the predefined negative gate of 0.003.

## Why aligned support failed

The intervention diagnostics show that B2 successfully localizes decision-relevant tokens: masking its Top-5 relevance tokens reduces Macro-F1 by 0.139048, whereas masking five matched random tokens reduces it by only 0.001131. The failure therefore does not arise from an unused relevance locator. Instead, aligned support narrows the evidence available to both polarity branches. A0 allows positive evidence and counter-evidence to reside in different code regions, which is plausible for long smart contracts containing distributed and co-occurring vulnerability patterns.

This restriction weakens polarity separation. Mean positive-versus-negative margin separation is smaller in B2 than A0 for every label: Reentrancy 8.862 vs 13.903, Access Control 9.916 vs 14.519, Arithmetic 9.982 vs 15.060, Unchecked Return Values 14.432 vs 14.833, DoS 7.682 vs 9.896, and Time manipulation 14.016 vs 19.308.

## Verdict

**NOT SUPPORTED**

The experiments do not support replacing independent positive/negative retrieval with a single shared token support. Dedicated relevance localization is meaningful and improves over the positive-locator control, but the aligned-support constraint reduces the evidence margin and overall validation performance.

## Next step

**Keep P11. Do not run seeds 43/44 for B2 and do not add losses to rescue the architecture.**

The defensible mechanism conclusion is: independent positive and negative retrieval in P11 appears to provide useful positional specialization. Relevance localization and polarity discrimination are both learnable, but forcing both polarities onto one shared support is overly restrictive for the current multi-label smart-contract setting.
