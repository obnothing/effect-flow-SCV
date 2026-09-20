"""Run controlled one-factor refinements around Stage 1 Trial 14."""

import csv
import json
import sys
from pathlib import Path

import yaml

import run_polarity_queries as base


ROOT = base.ROOT
BASE_CONFIG = ROOT / "configs/p11_trial14_refinement/base.yaml"
GENERATED = ROOT / "configs/p11_trial14_refinement/generated"
RESULT_ROOT = ROOT / "results/p11_trial14_refinement"
REPORT_ROOT = ROOT / "reports/p11_trial14_refinement"
REFERENCE_PATH = ROOT / "results/p11_hparam_stage1/trial_014/P11/metrics.json"
TRIAL2_PATH = ROOT / "results/p11_hparam_stage1/trial_002/P11/metrics.json"
REFERENCE_TUNED = 0.8429913354692328
REFERENCE_FIXED = 0.8365436334788546
LABELS = ["Reentrancy", "Access Control", "Arithmetic", "Unchecked Return Values", "DoS", "Time manipulation"]
EXPERIMENTS = (
    ("R1_lambda005", {"auxiliary_weight": 0.05}),
    ("R2_lambda010", {"auxiliary_weight": 0.10}),
    ("R3_power055", {"pos_weight_power": 0.55}),
    ("R4_power065", {"pos_weight_power": 0.65}),
    ("R5_batch064", {"effective_batch_size": 64, "gradient_accumulation_steps": 1}),
    ("R6_batch256", {"effective_batch_size": 256, "gradient_accumulation_steps": 4}),
    ("R7_dropout002", {"representation_dropout": 0.02}),
    ("E1_embedding256", {"embedding_dim": 256}),
    ("E2_embedding512", {"embedding_dim": 512}),
    ("E3_embedding768", {"embedding_dim": 768}),
    ("Q1_query512", {"query_dim": 512}),
    ("Q2_query1024", {"query_dim": 1024}),
)
SEARCH_FIELDS = {"auxiliary_weight", "pos_weight_power", "effective_batch_size",
                 "gradient_accumulation_steps", "representation_dropout", "embedding_dim", "query_dim"}


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def make_config(name, changes):
    config = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    config.update(changes)
    config["route_name"] = f"P11 Trial 14 refinement {name}"
    config["result_root"] = f"results/p11_trial14_refinement/{name}"
    config["checkpoint_root"] = f"checkpoints/p11_trial14_refinement/{name}"
    if config["effective_batch_size"] % config["batch_size"]:
        raise ValueError(f"Configured effective batch must be divisible by initial physical batch: {name}")
    config["gradient_accumulation_steps"] = config["effective_batch_size"] // config["batch_size"]
    baseline = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    changed = {key for key in SEARCH_FIELDS if config.get(key) != baseline.get(key)}
    expected = set(changes)
    if "effective_batch_size" in expected:
        expected.add("gradient_accumulation_steps")
    if changed != expected:
        raise ValueError(f"{name} is not a one-factor experiment: changed={changed}, expected={expected}")
    GENERATED.mkdir(parents=True, exist_ok=True)
    path = GENERATED / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def row_from_metrics(name, payload, source, kind="new"):
    config, metrics = payload["requested_config"], payload["metrics"]
    history = payload["history"]
    return {
        "name": name,
        "kind": kind,
        "embedding_dim": int(config["embedding_dim"]),
        "query_dim": int(config.get("query_dim", 2 * config["gru_hidden_size"])),
        "representation_dropout": float(config.get("representation_dropout", 0.0)),
        "lambda_pol": float(config["auxiliary_weight"]),
        "pos_weight_power": float(config.get("pos_weight_power", 0.5)),
        "effective_batch_size": int(config.get("effective_batch_size", 256)),
        "physical_batch_size": int(payload["effective_config"]["batch_size"]),
        "gradient_accumulation_steps": int(payload["effective_config"]["gradient_accumulation_steps"]),
        "best_epoch": int(payload["best_epoch"]),
        "tuned_macro_f1": float(metrics["tuned"]["macro_f1"]),
        "fixed_macro_f1": float(metrics["fixed"]["macro_f1"]),
        "micro_f1": float(metrics["tuned"]["micro_f1"]),
        "detection_f1": float(metrics["detection_f1"]),
        "threshold_gain": float(metrics["tuned"]["macro_f1"] - metrics["fixed"]["macro_f1"]),
        "delta_vs_trial14": float(metrics["tuned"]["macro_f1"] - REFERENCE_TUNED),
        "fixed_delta_vs_trial14": float(metrics["fixed"]["macro_f1"] - REFERENCE_FIXED),
        "params": int(payload["params"]),
        "peak_vram_mb": float(max(epoch["peak_memory_mb"] for epoch in history)),
        "seconds_per_epoch": float(sum(epoch["train_seconds"] for epoch in history) / len(history)),
        "source": source,
        "test_checked": False,
        **{f"label_{index}_f1": float(value) for index, value in enumerate(metrics["tuned"]["per_label_f1"])},
        **{f"threshold_{index}": float(value) for index, value in enumerate(metrics["thresholds"])},
    }


def collect_rows():
    rows = []
    if REFERENCE_PATH.exists():
        rows.append(row_from_metrics("Trial14_reference", load_json(REFERENCE_PATH),
                    str(REFERENCE_PATH.relative_to(ROOT)), "reference"))
    if TRIAL2_PATH.exists():
        rows.append(row_from_metrics("Trial2_dropout005", load_json(TRIAL2_PATH),
                    str(TRIAL2_PATH.relative_to(ROOT)), "existing_control"))
    for name, _ in EXPERIMENTS:
        path = RESULT_ROOT / name / "P11/metrics.json"
        if path.exists():
            rows.append(row_from_metrics(name, load_json(path), str(path.relative_to(ROOT))))
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def write_reports(rows):
    RESULT_ROOT.mkdir(parents=True, exist_ok=True); REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    write_csv(RESULT_ROOT / "all_results.csv", rows)
    ranked = sorted(rows, key=lambda item: item["tuned_macro_f1"], reverse=True)
    write_csv(RESULT_ROOT / "ranking.csv", ranked)
    lines = ["# Trial 14 Controlled Refinement", "", "process01; seed 42; validation only; test_checked=false.", "",
             "| Rank | Experiment | Embedding | Query | Dropout | lambda | Power | Batch | Tuned | Fixed | Delta | Fixed delta | VRAM MB | sec/epoch |",
             "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for rank, row in enumerate(ranked, 1):
        lines.append(f"| {rank} | {row['name']} | {row['embedding_dim']} | {row['query_dim']} | {row['representation_dropout']} | "
                     f"{row['lambda_pol']} | {row['pos_weight_power']} | {row['effective_batch_size']} | "
                     f"{row['tuned_macro_f1']:.6f} | {row['fixed_macro_f1']:.6f} | {row['delta_vs_trial14']:+.6f} | "
                     f"{row['fixed_delta_vs_trial14']:+.6f} | {row['peak_vram_mb']:.1f} | {row['seconds_per_epoch']:.1f} |")
    (REPORT_ROOT / "interim_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    completed = sum(row["kind"] == "new" for row in rows)
    if completed != len(EXPERIMENTS):
        return
    best = ranked[0]
    per_label = [best[f"label_{index}_f1"] for index in range(6)]
    final = lines + ["", f"BEST_EXPERIMENT: {best['name']}", f"DELTA_VS_TRIAL14: {best['delta_vs_trial14']:+.6f}",
                     f"FIXED_DELTA_VS_TRIAL14: {best['fixed_delta_vs_trial14']:+.6f}",
                     "PER_LABEL_F1: " + ", ".join(f"{label}={value:.6f}" for label, value in zip(LABELS, per_label)),
                     "", "This round changes one factor at a time. No combined capacity model or test evaluation was run."]
    (REPORT_ROOT / "final_report.md").write_text("\n".join(final) + "\n", encoding="utf-8")


def main():
    if not REFERENCE_PATH.exists() or not TRIAL2_PATH.exists():
        raise FileNotFoundError("Stage 1 Trial 14 and Trial 2 artifacts are required")
    write_reports(collect_rows())
    for name, changes in EXPERIMENTS:
        config_path = make_config(name, changes)
        config = base.load_config(config_path)
        print(f"[trial14-refinement] starting {name} changes={changes}", flush=True)
        base.CONFIG = config_path; base.VARIANTS = ("P11",)
        sys.argv = [sys.argv[0], "--config", str(config_path)]
        base.main()
        metrics_path = ROOT / config["result_root"] / "P11/metrics.json"
        if not metrics_path.exists():
            raise RuntimeError(f"Missing completed metrics: {metrics_path}")
        write_reports(collect_rows())
        print(f"[trial14-refinement] completed {name}", flush=True)
    rows = collect_rows(); write_reports(rows)
    print("[trial14-refinement] all 12 experiments complete test_checked=false", flush=True)


if __name__ == "__main__":
    main()
