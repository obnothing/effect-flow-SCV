"""Run P10-P13 as isolated single-factor polarity-query experiments."""

import sys

import run_polarity_queries as base


ROOT = base.ROOT
FOLLOWUPS = (
    ("P10", ROOT / "configs/light_label/polarity_queries_followup_p10.yaml"),
    ("P11", ROOT / "configs/light_label/polarity_queries_followup_p11.yaml"),
    ("P12", ROOT / "configs/light_label/polarity_queries_followup_p12.yaml"),
    ("P13", ROOT / "configs/light_label/polarity_queries_followup_p13.yaml"),
)


def main():
    for mode, config in FOLLOWUPS:
        base.CONFIG = config
        base.VARIANTS = (mode,)
        sys.argv = [sys.argv[0]]
        base.main()


if __name__ == "__main__":
    main()
