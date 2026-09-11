"""Create or resume the persistent Optuna study and AutoTune state."""

import importlib.util
import json
from pathlib import Path

import yaml

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from autotune_state import append_ledger, read_state, write_state  # noqa: E402


def main():
    if importlib.util.find_spec("optuna") is None:
        raise RuntimeError("Optuna is not installed. Install requirements-autotune.txt in the server pytorch environment first.")
    import optuna
    spec = yaml.safe_load((ROOT / "autotune/study.yaml").read_text(encoding="utf-8"))
    storage = spec["storage"]
    study = optuna.create_study(
        study_name=spec["study_name"], storage=storage, direction=spec["direction"],
        sampler=optuna.samplers.TPESampler(seed=int(spec["seed"])),
        pruner=optuna.pruners.HyperbandPruner(), load_if_exists=True,
    )
    state = read_state()
    if state.get("state") == "STOP":
        raise RuntimeError("AutoTune is already stopped; reset requires an explicit user decision.")
    if state.get("state") == "RUNNING":
        raise RuntimeError(f"AutoTune trial is marked RUNNING: {state.get('current_trial_id')}; inspect before restarting.")
    state.update({"study_name": spec["study_name"], "storage": storage,
                  "optuna_trials": len(study.trials), "test_checked": False})
    write_state(state)
    append_ledger({"event": "study_bootstrapped", "study_name": spec["study_name"], "optuna_trials": len(study.trials)})
    print(json.dumps({"study_name": study.study_name, "trials": len(study.trials), "state": state["state"], "test_checked": False}, indent=2))


if __name__ == "__main__": main()
