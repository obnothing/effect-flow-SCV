"""Single-factor P5 follow-up: increase only the polarity auxiliary weight."""

import run_polarity_queries as base

base.CONFIG = base.ROOT / "configs/light_label/polarity_queries_p5.yaml"
base.VARIANTS = ("P5",)
base.PROTOCOL_VERSION = "polarity_queries_p5_auxiliary_weight_0.2"


if __name__ == "__main__":
    base.main()
