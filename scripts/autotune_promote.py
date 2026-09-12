"""Promote the best completed AutoTune configuration to 10 or 30 epochs."""

import argparse
import json
import sys
from pathlib import Path

import optuna

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autotune_state import append_ledger, update_best  # noqa: E402
from autotune_campaign import load_base, train_trial  # noqa: E402


def best_trial_config(architecture):
    study = optuna.load_study(study_name=f"scvd_main6_autotune_{architecture}", storage="sqlite:///autotune/optuna.db")
    completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None]
    if not completed:
        raise RuntimeError(f"no completed trials for {architecture}")
    trial = max(completed, key=lambda item: item.value)
    base = load_base()
    config = {
        "aggregation": architecture,
        "embedding_dim": int(trial.params["embedding_dim"]),
        "encoder": trial.params["encoder"],
        "hidden": int(trial.params["hidden"]),
        "layers": int(trial.params["layers"]),
        "embedding_mlp": bool(trial.params["embedding_mlp"]),
        "self_attention": bool(trial.params["self_attention"]),
        "residual": bool(trial.params["residual"]),
        "dropout": float(trial.params["dropout"]),
        "learning_rate": float(trial.params["learning_rate"]),
        "weight_decay": float(trial.params["weight_decay"]),
        "gradient_clip": float(trial.params["gradient_clip"]),
        "optimizer": trial.params["optimizer"],
        "scheduler": trial.params["scheduler"],
        "max_len": int(trial.params["max_len"]),
        "batch_size": int(base.get("batch_size", 4)),
        "gradient_accumulation_steps": int(base.get("gradient_accumulation_steps", 4)),
        "num_labels": 6,
        "bidirectional": str(trial.params["encoder"]).startswith("Bi"),
        "amp": bool(base.get("amp", True)),
        "weighted_bce": bool(base.get("weighted_bce", True)),
        "pos_weight_mode": base.get("pos_weight_mode", "sqrt_ratio"),
        "max_pos_weight": float(base.get("max_pos_weight", 5.0)),
    }
    if config["self_attention"]:
        config["heads"] = int(trial.params["heads"])
    return config, trial.number


class FixedTrial:
    def __init__(self):
        self.values = []

    def report(self, value, step):
        self.values.append((step, value))

    def should_prune(self):
        return False


def promote(architecture, fidelity, device):
    result_dir = ROOT / f"results/autotune/promoted/{architecture}"
    result_dir.mkdir(parents=True, exist_ok=True)
    target = result_dir / f"fidelity{fidelity}.json"
    if target.exists():
        return json.loads(target.read_text(encoding="utf-8"))
    config, trial_number = best_trial_config(architecture)
    if fidelity == 30:
        previous = result_dir / "fidelity10.json"
        if previous.exists():
            previous_payload = json.loads(previous.read_text(encoding="utf-8"))
            config = previous_payload["config"]
            trial_number = previous_payload["source_trial_number"]
    metrics, checkpoint, history = train_trial(config, FixedTrial(), f"promoted_{architecture}_fidelity{fidelity}", trial_number, device, fidelity)
    payload = {"architecture": architecture, "fidelity": fidelity, "source_trial_number": trial_number,
               "metrics": metrics, "checkpoint": str(checkpoint), "config": config, "history": history, "test_checked": False}
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    update_best({"trial_id": f"promoted:{architecture}:fidelity{fidelity}", "variant": architecture,
                 "fidelity": fidelity, "metrics": metrics, "config": config, "checkpoint": str(checkpoint)})
    append_ledger({"event": "promotion_completed", "architecture": architecture, "fidelity": fidelity, "macro_f1": metrics["tuned_macro_f1"]})
    return payload


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--fidelity", type=int, choices=[10, 30], required=True); parser.add_argument("--all", action="store_true"); parser.add_argument("--architecture", choices=["a0_mean", "a1_shared", "a3_vsfs"]); args = parser.parse_args()
    if not args.all and not args.architecture: raise ValueError("use --all or --architecture")
    device = __import__("torch").device("cuda" if __import__("torch").cuda.is_available() else "cpu")
    architectures = ["a0_mean", "a1_shared", "a3_vsfs"] if args.all else [args.architecture]
    outputs = [promote(architecture, args.fidelity, device) for architecture in architectures]
    print(json.dumps({"fidelity": args.fidelity, "architectures": architectures, "macro_f1": [item["metrics"]["tuned_macro_f1"] for item in outputs], "test_checked": False}, indent=2))


if __name__ == "__main__": main()

