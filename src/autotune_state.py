"""Atomic persistence helpers for the resumable AutoTune state machine."""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT / "autotune"
STATE_PATH = STATE_DIR / "current_state.json"
LEDGER_PATH = STATE_DIR / "experiment_ledger.jsonl"
FAILURES_PATH = STATE_DIR / "failures.jsonl"
BEST_PATH = STATE_DIR / "best.json"

ALLOWED = {
    "INIT": {"PROPOSE"}, "PROPOSE": {"SMOKE", "INIT"}, "SMOKE": {"SUBMIT", "PROPOSE"},
    "SUBMIT": {"RUNNING", "PROPOSE"}, "RUNNING": {"EVALUATE", "COMPARE", "PROPOSE"},
    "EVALUATE": {"RECORD", "PROPOSE"}, "RECORD": {"COMPARE", "PROPOSE"}, "COMPARE": {"PROPOSE", "STOP"},
    "STOP": set(),
}


def _atomic_write(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def now():
    return datetime.now(timezone.utc).isoformat()


def read_state():
    if not STATE_PATH.exists():
        return {"state": "INIT", "dataset": "DIVE_main6_opcode_process01", "protocol": "validation_only", "seed": 42, "test_checked": False}
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def write_state(state):
    state = dict(state); state["updated_at"] = now(); state["test_checked"] = False
    _atomic_write(STATE_PATH, state)
    return state


def transition(next_state, **updates):
    state = read_state(); current = state.get("state", "INIT")
    if next_state not in ALLOWED.get(current, set()):
        raise ValueError(f"invalid AutoTune transition {current} -> {next_state}")
    state.update(updates); state["state"] = next_state
    append_ledger({"event": "state_transition", "from": current, "to": next_state, **updates})
    return write_state(state)


def append_ledger(event):
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": now(), **event, "test_checked": False}
    with LEDGER_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def record_failure(event):
    FAILURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": now(), **event, "test_checked": False}
    with FAILURES_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def update_best(record):
    current = json.loads(BEST_PATH.read_text(encoding="utf-8")) if BEST_PATH.exists() else {"best_macro_f1": None}
    score = float(record["metrics"]["tuned_macro_f1"])
    previous = current.get("best_macro_f1")
    if previous is None or score > float(previous):
        value = {"best_macro_f1": score, **record, "test_checked": False}
        _atomic_write(BEST_PATH, value)
        append_ledger({"event": "best_updated", "trial_id": record.get("trial_id"), "macro_f1": score})
        return True
    append_ledger({"event": "trial_recorded", "trial_id": record.get("trial_id"), "macro_f1": score, "best_macro_f1": previous})
    return False
