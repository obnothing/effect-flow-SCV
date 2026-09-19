"""Persistent 18-trial Optuna search for the frozen P11 architecture."""

import csv
import itertools
import json
import statistics
import sys
import time
from pathlib import Path

import optuna
import torch
import yaml

import run_polarity_queries as base


ROOT = base.ROOT
BASE_CONFIG = ROOT / "configs/p11_hparam_stage1/base.yaml"
RESULT_ROOT = ROOT / "results/p11_hparam_stage1"
REPORT_ROOT = ROOT / "reports/p11_hparam_stage1"
CONFIG_ROOT = ROOT / "configs/p11_hparam_stage1"
STORAGE = f"sqlite:///{(RESULT_ROOT / 'optuna.db').as_posix()}"
STUDY_NAME = "p11_stage1_refinement"
TOTAL_TRIALS = 18
ORIGINAL_TUNED = 0.833507102823917
ORIGINAL_FIXED = 0.8270858271394311
ORIGINAL_PER_LABEL = [0.8376623376623378, 0.871071716357776, 0.8160919540229885,
                      0.8549450549450549, 0.7532467532467533, 0.8680248007085918]
LABEL_KEYS = ["Reentrancy", "AccessControl", "Arithmetic", "URV", "DoS", "Time"]
SPACE = {
    "representation_dropout": [0.00, 0.03, 0.05, 0.08, 0.10],
    "lambda_pol": [0.03, 0.05, 0.08, 0.10, 0.15, 0.20],
    "pos_weight_power": [0.40, 0.45, 0.50, 0.55, 0.60],
    "effective_batch_size": [128, 256],
}
REFERENCE = (0.0, 0.10, 0.50, 256)
ALL_COMBINATIONS = list(itertools.product(*SPACE.values()))


def key(values):
    return (round(float(values[0]), 4), round(float(values[1]), 4),
            round(float(values[2]), 4), int(values[3]))


def read_base():
    return yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))


def trained_combinations():
    result = set()
    for path in RESULT_ROOT.glob("trial_*/P11/metrics.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = payload["requested_config"]
        result.add(key((config["representation_dropout"], config["auxiliary_weight"],
                        config["pos_weight_power"], config["effective_batch_size"])))
    return result


def resolve_parameters(trial):
    proposed = (
        trial.suggest_categorical("representation_dropout", SPACE["representation_dropout"]),
        trial.suggest_categorical("lambda_pol", SPACE["lambda_pol"]),
        trial.suggest_categorical("pos_weight_power", SPACE["pos_weight_power"]),
        trial.suggest_categorical("effective_batch_size", SPACE["effective_batch_size"]),
    )
    proposed = key(proposed)
    if trial.number == 0 and proposed != REFERENCE:
        raise ValueError(f"Trial 0 must be the P11 reference, got {proposed}")
    used = trained_combinations()
    if proposed not in used:
        selected = proposed
    else:
        available = [key(combo) for combo in ALL_COMBINATIONS if key(combo) not in used]
        if not available:
            raise RuntimeError("Search space exhausted")
        def distance(candidate):
            return (sum(left != right for left, right in zip(candidate, proposed)), candidate)
        selected = min(available, key=distance)
        trial.set_user_attr("deduplicated_from", list(proposed))
    trial.set_user_attr("actual_parameters", list(selected))
    return selected


def write_trial_config(trial_number, parameters):
    dropout, lambda_pol, power, effective_batch = parameters
    config = read_base()
    name = f"trial_{trial_number:03d}"
    config.update({
        "route_name": f"P11 hyperparameter stage 1 {name}",
        "result_root": f"results/p11_hparam_stage1/{name}",
        "checkpoint_root": f"checkpoints/p11_hparam_stage1/{name}",
        "representation_dropout": float(dropout),
        "auxiliary_weight": float(lambda_pol),
        "pos_weight_power": float(power),
        "effective_batch_size": int(effective_batch),
        "batch_size": 64,
        "gradient_accumulation_steps": int(effective_batch) // 64,
    })
    path = CONFIG_ROOT / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def result_record(trial_id, payload):
    config = payload["requested_config"]
    metrics = payload["metrics"]
    tuned, fixed = metrics["tuned"], metrics["fixed"]
    per_label = tuned["per_label_f1"]
    history = payload["history"]
    best_row = next(row for row in history if row["epoch"] == payload["best_epoch"])
    record = {
        "trial_id": int(trial_id),
        "representation_dropout": float(config["representation_dropout"]),
        "lambda_pol": float(config["auxiliary_weight"]),
        "pos_weight_power": float(config["pos_weight_power"]),
        "effective_batch_size": int(config["effective_batch_size"]),
        "physical_batch_size": int(payload["effective_config"]["batch_size"]),
        "gradient_accumulation_steps": int(payload["effective_config"]["gradient_accumulation_steps"]),
        "best_epoch": int(payload["best_epoch"]),
        "tuned_macro_f1": float(tuned["macro_f1"]),
        "fixed_macro_f1": float(fixed["macro_f1"]),
        "threshold_gain": float(tuned["macro_f1"] - fixed["macro_f1"]),
        "micro_f1": float(tuned["micro_f1"]),
        "detection_f1": float(metrics["detection_f1"]),
        "train_time_seconds": float(sum(row["train_seconds"] for row in history)),
        "peak_vram_mb": float(max(row["peak_memory_mb"] for row in history)),
        "best_train_total_loss": float(best_row["train_loss"]),
        "best_train_cls_loss": float(best_row["train_cls"]),
        "best_train_pol_loss": float(best_row["train_polarity"]),
        "best_validation_cls_loss": float(best_row["valid_loss"]),
        "best_gradient_norm": best_row.get("gradient_norm"),
        "status": "complete",
        "test_checked": False,
    }
    for label, value in zip(LABEL_KEYS, per_label):
        record[f"{label}_F1"] = float(value)
    for index, value in enumerate(metrics["thresholds"]):
        record[f"threshold_{index}"] = float(value)
    for index, value in enumerate(payload["pos_weight"]):
        record[f"pos_weight_{index}"] = float(value)
    record["delta_vs_original_p11"] = record["tuned_macro_f1"] - ORIGINAL_TUNED
    record["fixed_delta_vs_original_p11"] = record["fixed_macro_f1"] - ORIGINAL_FIXED
    record["labels_down_gt_0.01"] = sum(record[f"{label}_F1"] < baseline - 0.01
                                         for label, baseline in zip(LABEL_KEYS, ORIGINAL_PER_LABEL))
    return record


def collect_records():
    rows = []
    for path in sorted(RESULT_ROOT.glob("trial_*/P11/metrics.json")):
        trial_id = int(path.parents[1].name.split("_")[1])
        rows.append(result_record(trial_id, json.loads(path.read_text(encoding="utf-8"))))
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def write_curves(top_rows):
    curves = RESULT_ROOT / "curves"; curves.mkdir(parents=True, exist_ok=True)
    for record in top_rows:
        metrics_path = RESULT_ROOT / f"trial_{record['trial_id']:03d}/P11/metrics.json"
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows = []
        for epoch in payload["history"]:
            row = {
                "epoch": epoch["epoch"], "train_total_loss": epoch["train_loss"],
                "train_cls_loss": epoch["train_cls"], "train_pol_loss": epoch["train_polarity"],
                "validation_cls_loss": epoch["valid_loss"],
                "fixed_macro_f1": epoch["metrics"]["fixed"]["macro_f1"],
                "tuned_macro_f1": epoch["metrics"]["tuned"]["macro_f1"],
                "micro_f1": epoch["metrics"]["tuned"]["micro_f1"],
                "detection_f1": epoch["metrics"]["detection_f1"],
                "learning_rate": epoch.get("learning_rate", 0.001),
                "gradient_norm": epoch.get("gradient_norm"),
            }
            for index, value in enumerate(epoch["metrics"]["tuned"]["per_label_f1"]):
                row[f"label_{index}_f1"] = value
            for index, value in enumerate(epoch["metrics"]["thresholds"]):
                row[f"threshold_{index}"] = value
            rows.append(row)
        write_csv(curves / f"trial_{record['trial_id']:03d}.csv", rows)


def parameter_effects(rows):
    effect_rows = []
    for parameter in SPACE:
        for value in SPACE[parameter]:
            selected = [row for row in rows if row[parameter] == value]
            if selected:
                effect_rows.append({
                    "parameter": parameter, "value": value, "count": len(selected),
                    "mean_tuned_macro_f1": statistics.mean(row["tuned_macro_f1"] for row in selected),
                    "best_tuned_macro_f1": max(row["tuned_macro_f1"] for row in selected),
                    "mean_fixed_macro_f1": statistics.mean(row["fixed_macro_f1"] for row in selected),
                })
    write_csv(RESULT_ROOT / "hparam_effects.csv", effect_rows)
    lines = ["# Stage 1 Hyperparameter Effects", "", "Single-seed validation group summaries; descriptive, not causal.", "",
             "| Parameter | Value | n | Mean tuned macro | Best tuned macro | Mean fixed macro |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in effect_rows:
        lines.append(f"| {row['parameter']} | {row['value']} | {row['count']} | {row['mean_tuned_macro_f1']:.6f} | {row['best_tuned_macro_f1']:.6f} | {row['mean_fixed_macro_f1']:.6f} |")
    (REPORT_ROOT / "hparam_effects.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def classify(best):
    if (best["tuned_macro_f1"] >= 0.8385 and best["fixed_macro_f1"] >= ORIGINAL_FIXED - 0.002
            and best["labels_down_gt_0.01"] <= 2):
        return "IMPROVED"
    if best["tuned_macro_f1"] >= ORIGINAL_TUNED + 0.002 and best["fixed_macro_f1"] <= ORIGINAL_FIXED:
        return "THRESHOLD-SENSITIVE"
    return "NO_CLEAR_IMPROVEMENT"


def write_reports(rows):
    rows = sorted(rows, key=lambda item: item["tuned_macro_f1"], reverse=True)
    write_csv(RESULT_ROOT / "trials.csv", sorted(rows, key=lambda item: item["trial_id"]))
    write_csv(RESULT_ROOT / "top5.csv", rows[:5])
    write_curves(rows[:5]); parameter_effects(rows)
    summary = ["# P11 Hyperparameter Stage 1 Search Summary", "", f"Completed trials: {len(rows)}/{TOTAL_TRIALS}; validation only; test_checked=false.", "",
               "| Rank | Trial | Dropout | lambda_pol | Weight power | Effective batch | Best epoch | Tuned macro | Fixed macro | Delta | Fixed delta |",
               "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for rank, row in enumerate(rows[:5], 1):
        summary.append(f"| {rank} | {row['trial_id']} | {row['representation_dropout']} | {row['lambda_pol']} | {row['pos_weight_power']} | {row['effective_batch_size']} | {row['best_epoch']} | {row['tuned_macro_f1']:.6f} | {row['fixed_macro_f1']:.6f} | {row['delta_vs_original_p11']:+.6f} | {row['fixed_delta_vs_original_p11']:+.6f} |")
    (REPORT_ROOT / "search_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    if len(rows) < TOTAL_TRIALS:
        return
    best, top3 = rows[0], rows[:3]
    trial0 = next(row for row in rows if row["trial_id"] == 0)
    best_fixed = max(rows, key=lambda item: item["fixed_macro_f1"])
    verdict = classify(best)
    gains = {label: best[f"{label}_F1"] - baseline for label, baseline in zip(LABEL_KEYS, ORIGINAL_PER_LABEL)}
    best_dropout = max(SPACE["representation_dropout"], key=lambda value: max(
        (row["tuned_macro_f1"] for row in rows if row["representation_dropout"] == value), default=-1))
    best_lambda = max(SPACE["lambda_pol"], key=lambda value: max(
        (row["tuned_macro_f1"] for row in rows if row["lambda_pol"] == value), default=-1))
    best_power = max(SPACE["pos_weight_power"], key=lambda value: max(
        (row["tuned_macro_f1"] for row in rows if row["pos_weight_power"] == value), default=-1))
    batch_means = {value: statistics.mean(row["tuned_macro_f1"] for row in rows if row["effective_batch_size"] == value)
                   for value in SPACE["effective_batch_size"] if any(row["effective_batch_size"] == value for row in rows)}
    final = ["# P11 Hyperparameter Refinement Stage 1", "", "DIVE Main-6 process01; seed 42; validation only; test_checked=false.", "",
             f"ORIGINAL_P11: tuned={ORIGINAL_TUNED:.6f}, fixed={ORIGINAL_FIXED:.6f}",
             f"TRIAL_0_40_EPOCH: tuned={trial0['tuned_macro_f1']:.6f} ({trial0['delta_vs_original_p11']:+.6f}), fixed={trial0['fixed_macro_f1']:.6f}", "",
             "BEST_STAGE1_TRIAL:",
             f"trial_id: {best['trial_id']}", f"representation_dropout: {best['representation_dropout']}",
             f"lambda_pol: {best['lambda_pol']}", f"pos_weight_power: {best['pos_weight_power']}",
             f"effective_batch_size: {best['effective_batch_size']}", f"best_epoch: {best['best_epoch']}",
             f"Tuned Macro-F1: {best['tuned_macro_f1']:.6f}", f"Delta: {best['delta_vs_original_p11']:+.6f}",
             f"Fixed Macro-F1: {best['fixed_macro_f1']:.6f}", f"Fixed Delta: {best['fixed_delta_vs_original_p11']:+.6f}",
             f"Micro-F1: {best['micro_f1']:.6f}", f"Detection-F1: {best['detection_f1']:.6f}", "", "TOP3_FOR_STAGE2:"]
    for index, row in enumerate(top3, 1):
        final.append(f"{index}. trial {row['trial_id']}: dropout={row['representation_dropout']}, lambda={row['lambda_pol']}, power={row['pos_weight_power']}, batch={row['effective_batch_size']}, tuned={row['tuned_macro_f1']:.6f}, fixed={row['fixed_macro_f1']:.6f}")
    final.extend(["", "## Required Questions", "",
                  f"1. Extending the unchanged P11 setting to 40 epochs changed tuned Macro-F1 by {trial0['delta_vs_original_p11']:+.6f}.",
                  f"2. The best observed dropout level was {best_dropout}; group summaries are reported separately and do not establish causality.",
                  f"3. The best observed polarity coefficient was {best_lambda}.",
                  "4. All searched polarity coefficients are positive, so Stage 1 compares supervision strength but cannot prove auxiliary supervision is necessary versus lambda=0.",
                  f"5. The best observed positive-weight power was {best_power}.",
                  f"6. Effective-batch group means were {batch_means}.",
                  f"7. Best tuned Macro-F1 was {best['tuned_macro_f1']:.6f}.",
                  f"8. Best fixed Macro-F1 was {best_fixed['fixed_macro_f1']:.6f} in trial {best_fixed['trial_id']}.",
                  f"9. The winning trial threshold gain was {best['threshold_gain']:.6f}; its fixed delta was {best['fixed_delta_vs_original_p11']:+.6f}.",
                  "10. Winning-trial per-label deltas were " + ", ".join(f"{label}={value:+.6f}" for label, value in gains.items()) + ".",
                  f"11. DoS changed by {gains['DoS']:+.6f} in the winning single-seed trial; stability requires Stage 2 multi-seed confirmation.",
                  "12. The three configurations listed above are the only candidates selected for Stage 2.", "",
                  f"STAGE1_VERDICT: {verdict}", "", "No test artifact was read or created. Multi-seed Stage 2 was not started."])
    (REPORT_ROOT / "final_report.md").write_text("\n".join(final) + "\n", encoding="utf-8")
    base.atomic_json(RESULT_ROOT / "best_trial/reference.json", {"trial_id": best["trial_id"], "metrics": best,
                     "source": f"results/p11_hparam_stage1/trial_{best['trial_id']:03d}/P11/metrics.json", "test_checked": False})


def objective(trial):
    parameters = resolve_parameters(trial)
    config_path = write_trial_config(trial.number, parameters)
    config = base.load_config(config_path)
    if config["epochs"] != 40 or config["early_stopping_patience"] < 40:
        raise ValueError("Every Stage 1 trial must run all 40 epochs")
    if config["batch_size"] != 64 or config["effective_batch_size"] not in (128, 256):
        raise ValueError("Stage 1 batch protocol mismatch")
    train, valid = base.datasets(config)
    tokenizer = base.EVMOpcodeTokenizer.from_vocab_file(ROOT / config["vocab_path"])
    provenance = base.provenance(config, config_path)
    resources = {
        "signature": base.signature({"physical_batch": 64, "effective_batch": config["effective_batch_size"]}),
        "batch_size": 64,
        "gradient_accumulation_steps": config["effective_batch_size"] // 64,
        "effective_batch_size": config["effective_batch_size"],
        "threads": 2,
        "preflight": [{"variant": "P11", "batch_size": 64, "status": "fixed_from_verified_P11"}],
        "cpu_benchmark": [],
        "test_checked": False,
    }
    result = base.train_one("P11", config, tokenizer, train, valid, provenance, resources)
    tuned = float(result["metrics"]["tuned"]["macro_f1"])
    trial.set_user_attr("fixed_macro_f1", float(result["metrics"]["fixed"]["macro_f1"]))
    trial.set_user_attr("result_path", f"results/p11_hparam_stage1/trial_{trial.number:03d}/P11/metrics.json")
    write_reports(collect_records())
    return tuned


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    RESULT_ROOT.mkdir(parents=True, exist_ok=True); REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=8, multivariate=True)
    study = optuna.create_study(study_name=STUDY_NAME, storage=STORAGE, direction="maximize",
                                sampler=sampler, load_if_exists=True)
    if not study.trials:
        study.enqueue_trial({"representation_dropout": 0.0, "lambda_pol": 0.10,
                             "pos_weight_power": 0.50, "effective_batch_size": 256})
    completed = sum(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials)
    remaining = max(0, TOTAL_TRIALS - completed)
    print(f"[stage1] completed={completed} remaining={remaining} test_checked=false", flush=True)
    if remaining:
        study.optimize(objective, n_trials=remaining, gc_after_trial=True, show_progress_bar=False)
    rows = collect_records(); write_reports(rows)
    if len(rows) != TOTAL_TRIALS:
        raise RuntimeError(f"Expected {TOTAL_TRIALS} completed training trials, found {len(rows)}")
    print(f"[stage1] complete best={max(row['tuned_macro_f1'] for row in rows):.6f} test_checked=false", flush=True)


if __name__ == "__main__":
    main()
