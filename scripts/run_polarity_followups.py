"""Run P6-P9 polarity follow-ups sequentially with isolated artifacts."""

import sys
from pathlib import Path

import run_polarity_queries as base

ROOT = Path(__file__).resolve().parents[1]
FOLLOWUPS = (
    ("P6", ROOT / "configs/light_label/polarity_queries_followup_p6.yaml"),
    ("P7", ROOT / "configs/light_label/polarity_queries_followup_p7.yaml"),
    ("P8", ROOT / "configs/light_label/polarity_queries_followup_p8.yaml"),
    ("P9", ROOT / "configs/light_label/polarity_queries_followup_p9.yaml"),
)


def main():
    for mode, config in FOLLOWUPS:
        base.CONFIG = config
        base.VARIANTS = (mode,)
        base.PROTOCOL_VERSION = f"polarity_queries_{mode}_single_factor"
        sys.argv = [sys.argv[0]]
        base.main()


if __name__ == "__main__":
    main()
