"""Generate isolated process01 B2 baseline sweep configurations."""

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def main():
    base = yaml.safe_load((ROOT / "configs/light_label/b2_label_attention.yaml").read_text(encoding="utf-8"))
    experiments = {
        "pw_none": {"weighted_bce": False},
        "pw_sqrt_cap3": {"weighted_bce": True, "pos_weight_mode": "sqrt_ratio", "max_pos_weight": 3.0},
        "pw_ratio_cap5": {"weighted_bce": True, "pos_weight_mode": "ratio", "max_pos_weight": 5.0},
        "hidden192": {"gru_hidden_size": 192},
        "hidden256": {"gru_hidden_size": 256},
        "hidden384": {"gru_hidden_size": 384},
        "embed256": {"embedding_dim": 256},
        "embed256_hidden256": {"embedding_dim": 256, "gru_hidden_size": 256},
        "lr3e4": {"learning_rate": 3e-4},
        "lr5e4": {"learning_rate": 5e-4},
        "wd3e4": {"weight_decay": 3e-4},
        "wd1e3": {"weight_decay": 1e-3},
    }
    output = ROOT / "configs/light_label/baseline_sweep"
    output.mkdir(parents=True, exist_ok=True)
    for name, changes in experiments.items():
        config = dict(base)
        config.update(changes)
        config.update({
            "variant": "b2_label_attention",
            "max_len": 8192,
            "runtime_path": f"results/light_label/baseline_sweep_runtime/{name}.json",
            "result_dir": f"results/light_label/baseline_sweep/{name}",
            "checkpoint_dir": f"checkpoints/light_label/baseline_sweep/{name}",
            "report_dir": "reports/light_label_model/baseline_sweep",
            "allow_test": False,
        })
        (output / f"{name}.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"generated {len(experiments)} baseline sweep configs")


if __name__ == "__main__":
    main()

