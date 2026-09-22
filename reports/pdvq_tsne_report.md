# PDVQ vulnerability-aware representation t-SNE

Input features are validation-contract `Z+ - Z-` evidence differences for all six labels, flattened from `[6,768]` to 4608 dimensions, reduced by PCA to 50 dimensions and t-SNE to two dimensions (perplexity 30, random state 42). Colors encode the ten most frequent multilabel combinations; rarer combinations are grouped as Other. The visualization suggests representation structure but does not represent a true classification boundary or prove separability.

Validation contracts: 2032. Top-combination counts: {"000000": 615, "110000": 149, "010000": 126, "111000": 121, "000001": 120, "111101": 104, "011000": 81, "111100": 80, "001000": 55, "110001": 43}. Test was not read.
