"""Run the four confirmed BiGRU/PDVQ width configurations."""

import csv
import json
import sys
from pathlib import Path

import run_polarity_queries as base


ROOT = base.ROOT
RESULT_ROOT = ROOT / "results/p11_width_study"
REPORT_ROOT = ROOT / "reports/p11_width_study"
REFERENCE = ROOT / "results/p11_hparam_stage1/trial_014/P11/metrics.json"
VARIANTS = (
    ("E1", ROOT / "configs/p11_width_study/e1.yaml"),
    ("E2", ROOT / "configs/p11_width_study/e2.yaml"),
    ("E3", ROOT / "configs/p11_width_study/e3.yaml"),
    ("E4", ROOT / "configs/p11_width_study/e4.yaml"),
)
LABELS = ["Reentrancy", "Access Control", "Arithmetic", "Unchecked Return Values", "DoS", "Time manipulation"]
REFERENCE_TUNED = 0.8429913354692328
REFERENCE_FIXED = 0.8365436334788546


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(name, payload, source):
    config, metrics, history = payload["requested_config"], payload["metrics"], payload["history"]
    query_dim = int(config.get("query_dim", 2 * config["gru_hidden_size"]))
    best = next(row for row in history if row["epoch"] == payload["best_epoch"])
    minimum = min(history, key=lambda row: row["valid_loss"])
    final = history[-1]
    final_tuned = final["metrics"]["tuned"]["macro_f1"]
    rebound = final["valid_loss"] - minimum["valid_loss"]
    f1_drop = metrics["tuned"]["macro_f1"] - final_tuned
    overfit = "strong" if rebound >= 0.15 and f1_drop >= 0.005 else (
              "moderate" if rebound >= 0.10 or f1_drop >= 0.003 else "limited")
    row = {
        "model": name,
        "embedding_dim": int(config["embedding_dim"]),
        "gru_hidden_per_direction": int(config["gru_hidden_size"]),
        "encoder_output_dim": 2 * int(config["gru_hidden_size"]),
        "attention_heads": int(config["attention_heads"]),
        "head_dim": query_dim // int(config["attention_heads"]),
        "query_dim_per_polarity": query_dim,
        "positive_negative_total_dim": 2 * query_dim,
        "best_epoch": int(payload["best_epoch"]),
        "tuned_macro_f1": float(metrics["tuned"]["macro_f1"]),
        "fixed_macro_f1": float(metrics["fixed"]["macro_f1"]),
        "micro_f1": float(metrics["tuned"]["micro_f1"]),
        "detection_f1": float(metrics["detection_f1"]),
        "delta_vs_trial14": float(metrics["tuned"]["macro_f1"] - REFERENCE_TUNED),
        "fixed_delta_vs_trial14": float(metrics["fixed"]["macro_f1"] - REFERENCE_FIXED),
        "min_valid_loss": float(minimum["valid_loss"]),
        "min_valid_loss_epoch": int(minimum["epoch"]),
        "best_epoch_train_cls_loss": float(best["train_cls"]),
        "best_epoch_valid_cls_loss": float(best["valid_loss"]),
        "final_train_cls_loss": float(final["train_cls"]),
        "final_valid_cls_loss": float(final["valid_loss"]),
        "valid_loss_rebound": float(rebound),
        "best_to_final_tuned_drop": float(f1_drop),
        "overfitting_evidence": overfit,
        "params": int(payload["params"]),
        "physical_batch_size": int(payload["effective_config"]["batch_size"]),
        "effective_batch_size": int(payload["effective_config"]["batch_size"] * payload["effective_config"]["gradient_accumulation_steps"]),
        "peak_vram_mb": float(max(item["peak_memory_mb"] for item in history)),
        "seconds_per_epoch": float(sum(item["train_seconds"] for item in history) / len(history)),
        "source": source,
        "test_checked": False,
    }
    row.update({f"label_{index}_f1": float(value) for index, value in enumerate(metrics["tuned"]["per_label_f1"])})
    return row


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def write_history(name, payload):
    curves = RESULT_ROOT / "curves"; curves.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in payload["history"]:
        row = {
            "epoch": item["epoch"], "train_total_loss": item["train_loss"],
            "train_cls_loss": item["train_cls"], "train_pol_loss": item["train_polarity"],
            "valid_cls_loss": item["valid_loss"], "fixed_macro_f1": item["metrics"]["fixed"]["macro_f1"],
            "tuned_macro_f1": item["metrics"]["tuned"]["macro_f1"],
            "micro_f1": item["metrics"]["tuned"]["micro_f1"],
            "detection_f1": item["metrics"]["detection_f1"], "gradient_norm": item.get("gradient_norm"),
        }
        row.update({f"label_{index}_f1": value for index, value in enumerate(item["metrics"]["tuned"]["per_label_f1"])})
        rows.append(row)
    write_csv(curves / f"{name}.csv", rows)


def collect():
    rows = []
    if REFERENCE.exists():
        rows.append(summarize("Trial14", load(REFERENCE), str(REFERENCE.relative_to(ROOT))))
    for name, config_path in VARIANTS:
        config = base.load_config(config_path)
        path = ROOT / config["result_root"] / "P11/metrics.json"
        if path.exists():
            payload = load(path); rows.append(summarize(name, payload, str(path.relative_to(ROOT))))
            write_history(name, payload)
    return rows


def report(rows):
    RESULT_ROOT.mkdir(parents=True, exist_ok=True); REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    write_csv(RESULT_ROOT / "width_results.csv", rows)
    ranked = sorted(rows, key=lambda row: row["tuned_macro_f1"], reverse=True)
    lines = ["# P11 BiGRU and Polarity Query Width Study", "", "process01; seed 42; validation only; test_checked=false.", "",
             "| Rank | Model | Embedding | GRU hidden/direction | Encoder output | Heads x head dim | q+/q- dim | Tuned | Fixed | Delta | Overfit | VRAM MB | sec/epoch |",
             "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|"]
    for rank, row in enumerate(ranked, 1):
        lines.append(f"| {rank} | {row['model']} | {row['embedding_dim']} | {row['gru_hidden_per_direction']} | "
                     f"{row['encoder_output_dim']} | {row['attention_heads']}x{row['head_dim']} | {row['query_dim_per_polarity']} | "
                     f"{row['tuned_macro_f1']:.6f} | {row['fixed_macro_f1']:.6f} | {row['delta_vs_trial14']:+.6f} | "
                     f"{row['overfitting_evidence']} | {row['peak_vram_mb']:.1f} | {row['seconds_per_epoch']:.1f} |")
    (REPORT_ROOT / "interim_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if len(rows) != 5:
        return
    overfit_lines = ["# Width Study Overfitting Analysis", "",
        "| Model | Best F1 epoch | Min valid-loss epoch | Min valid loss | Final valid loss | Rebound | Best-to-final F1 drop | Evidence |",
        "|---|---:|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        overfit_lines.append(f"| {row['model']} | {row['best_epoch']} | {row['min_valid_loss_epoch']} | {row['min_valid_loss']:.6f} | "
                             f"{row['final_valid_cls_loss']:.6f} | {row['valid_loss_rebound']:+.6f} | "
                             f"{row['best_to_final_tuned_drop']:+.6f} | {row['overfitting_evidence']} |")
    (REPORT_ROOT / "overfitting_analysis.md").write_text("\n".join(overfit_lines) + "\n", encoding="utf-8")
    best = ranked[0]
    lines.extend(["", f"BEST_WIDTH_MODEL: {best['model']}", f"DELTA_VS_TRIAL14: {best['delta_vs_trial14']:+.6f}",
                  f"FIXED_DELTA_VS_TRIAL14: {best['fixed_delta_vs_trial14']:+.6f}",
                  "", "No test data was read. Widths were not combined with other hyperparameter changes."])
    (REPORT_ROOT / "final_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    if not REFERENCE.exists():
        raise FileNotFoundError(REFERENCE)
    report(collect())
    for name, config_path in VARIANTS:
        print(f"[width-study] starting {name} config={config_path}", flush=True)
        base.CONFIG = config_path; base.VARIANTS = ("P11",)
        sys.argv = [sys.argv[0], "--config", str(config_path)]
        base.main()
        config = base.load_config(config_path)
        metrics = ROOT / config["result_root"] / "P11/metrics.json"
        if not metrics.exists():
            raise RuntimeError(f"Missing metrics for {name}")
        report(collect())
        print(f"[width-study] completed {name}", flush=True)
    report(collect())
    print("[width-study] all four experiments complete test_checked=false", flush=True)


if __name__ == "__main__":
    main()
