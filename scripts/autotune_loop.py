"""Long-running, resumable AutoTune controller.

The loop is intentionally bounded by autotune/stop_policy.yaml. It behaves as
an ongoing service, but stops on the configured trial/architecture/patience
limits instead of running without a scientific stopping rule.
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from autotune_state import append_ledger, read_state, record_failure, transition, write_state  # noqa: E402


ARCHITECTURES = ["a0_mean", "a1_shared", "a3_vsfs"]
PROMOTION10_MARKER = ROOT / "autotune/promotion_fidelity10.json"
PROMOTION30_MARKER = ROOT / "autotune/promotion_fidelity30.json"


def load_yaml(path):
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


def optuna_study(name):
    import optuna
    return optuna.load_study(study_name=f"scvd_main6_autotune_{name}", storage="sqlite:///autotune/optuna.db")


def recover_or_guard():
    state = read_state()
    if state.get("state") != "RUNNING":
        return state
    pid = state.get("pid")
    if pid:
        try:
            os.kill(int(pid), 0)
            raise RuntimeError(f"AutoTune trial is still running (pid={pid}, trial={state.get('current_trial_id')})")
        except ProcessLookupError:
            pass
    trial_id = str(state.get("current_trial_id") or "")
    match = re.match(r"(scvd_main6_autotune_[^:]+):(\d+)", trial_id)
    if match:
        try:
            study = __import__("optuna").load_study(study_name=match.group(1), storage="sqlite:///autotune/optuna.db")
            trial = next((item for item in study.trials if item.number == int(match.group(2))), None)
            if trial is not None and trial.state.name == "RUNNING":
                study._storage.set_trial_state_values(trial._trial_id, __import__("optuna").trial.TrialState.FAIL)
                record_failure({"trial_id": trial_id, "failure_type": "INTERRUPTED_TRIAL_RECOVERED"})
        except Exception as error:
            record_failure({"trial_id": trial_id, "failure_type": "RECOVERY_ERROR", "error": repr(error)})
    state.update({"state": "COMPARE", "current_trial_id": None, "pid": None})
    return write_state(state)


def main():
    if __import__("importlib.util").util.find_spec("optuna") is None:
        raise RuntimeError("Optuna is not installed")
    policy = load_yaml("autotune/stop_policy.yaml")
    state = recover_or_guard()
    if state.get("state") == "STOP":
        print(json.dumps({"state": "STOP", "reason": "state already stopped", "test_checked": False}, indent=2)); return
    no_gain = int(state.get("consecutive_architecture_rounds_without_gain", 0))
    last_round = int(state.get("last_architecture_round", 0))
    round_best = state.get("round_best_macro_f1")
    max_trials = int(policy["max_trials_per_architecture"])
    min_gain = float(policy["minimum_meaningful_gain"])
    while True:
        if state.get("state") == "INIT":
            transition("PROPOSE", current_architecture=None)
        counts = {}
        for architecture in ARCHITECTURES:
            try:
                counts[architecture] = len(optuna_study(architecture).trials)
            except Exception:
                counts[architecture] = 0
        if min(counts.values()) >= 3 and not PROMOTION10_MARKER.exists():
            append_ledger({"event": "promotion_dispatch", "fidelity": 10, "trial_counts": counts})
            completed = subprocess.run([sys.executable, str(ROOT / "scripts/autotune_promote.py"), "--fidelity", "10", "--all"], cwd=str(ROOT), check=False)
            if completed.returncode != 0:
                record_failure({"failure_type": "PROMOTION_PROCESS_ERROR", "fidelity": 10, "returncode": completed.returncode})
                raise RuntimeError("10-epoch promotion failed")
            PROMOTION10_MARKER.write_text(json.dumps({"fidelity": 10, "test_checked": False}, indent=2) + "\n", encoding="utf-8")
        if PROMOTION10_MARKER.exists() and not PROMOTION30_MARKER.exists():
            append_ledger({"event": "promotion_dispatch", "fidelity": 30, "trial_counts": counts})
            completed = subprocess.run([sys.executable, str(ROOT / "scripts/autotune_promote.py"), "--fidelity", "30", "--all"], cwd=str(ROOT), check=False)
            if completed.returncode != 0:
                record_failure({"failure_type": "PROMOTION_PROCESS_ERROR", "fidelity": 30, "returncode": completed.returncode})
                raise RuntimeError("30-epoch promotion failed")
            PROMOTION30_MARKER.write_text(json.dumps({"fidelity": 30, "test_checked": False}, indent=2) + "\n", encoding="utf-8")
        candidates = [architecture for architecture in ARCHITECTURES if counts[architecture] < max_trials]
        if not candidates:
            transition("STOP", reason="max_trials_per_architecture_reached", optuna_trials=counts)
            break
        architecture = min(candidates, key=lambda item: (counts[item], ARCHITECTURES.index(item)))
        before_path = ROOT / "autotune/best.json"
        before = json.loads(before_path.read_text(encoding="utf-8")) if before_path.exists() else {"best_macro_f1": None}
        append_ledger({"event": "loop_dispatch", "architecture": architecture, "trial_counts": counts})
        command = [sys.executable, str(ROOT / "scripts/autotune_campaign.py"), "--architecture", architecture, "--n-trials", "1", "--epochs", str(policy["trial_fidelity_epochs"])]
        completed = subprocess.run(command, cwd=str(ROOT), check=False)
        if completed.returncode != 0:
            record_failure({"failure_type": "CAMPAIGN_PROCESS_ERROR", "architecture": architecture, "returncode": completed.returncode})
            time.sleep(2)
            state = recover_or_guard()
            continue
        after = json.loads(before_path.read_text(encoding="utf-8")) if before_path.exists() else {"best_macro_f1": None}
        old = before.get("best_macro_f1"); new = after.get("best_macro_f1")
        new_counts = {}
        for name in ARCHITECTURES:
            try:
                new_counts[name] = len(optuna_study(name).trials)
            except Exception:
                new_counts[name] = 0
        completed_round = min(new_counts.values()) // 3
        if completed_round > last_round:
            previous_round_best = round_best
            current_best = after.get("best_macro_f1")
            gain = (float(current_best) - float(previous_round_best)) if previous_round_best is not None and current_best is not None else (float(current_best) if current_best is not None else 0.0)
            no_gain = 0 if gain >= min_gain else no_gain + 1
            round_best = current_best
            last_round = completed_round
            append_ledger({"event": "architecture_round_completed", "round": completed_round, "gain": gain, "no_gain_rounds": no_gain})
        state = recover_or_guard(); state["consecutive_architecture_rounds_without_gain"] = no_gain; state["last_architecture_round"] = last_round; state["round_best_macro_f1"] = round_best; state["trial_counts"] = new_counts; write_state(state)
        if no_gain >= int(policy["architecture_patience"]):
            transition("STOP", reason="architecture_patience_reached", no_gain_rounds=no_gain, minimum_meaningful_gain=min_gain)
            break
        if state.get("state") == "STOP": break
        time.sleep(1)
    print(json.dumps({"state": read_state().get("state"), "no_gain_rounds": no_gain, "test_checked": False}, indent=2))


if __name__ == "__main__": main()
