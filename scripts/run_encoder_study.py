"""Run and summarize the isolated PDVQ encoder architecture study."""

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import run_polarity_queries as base

ROOT = base.ROOT
P11_CONFIG = ROOT / "configs/light_label/polarity_queries_followup_p11.yaml"
P11_METRICS = ROOT / "results/light_label/polarity_queries_followup_p11/P11/metrics.json"
P11_CHECKPOINT = ROOT / "checkpoints/light_label/polarity_queries_followup_p11/P11/best.pt"
STUDIES = (
    ("E1_256", ROOT / "configs/encoder_study/e1_local256.yaml", "Local Transformer", "window=256"),
    ("E1_512", ROOT / "configs/encoder_study/e1_local512.yaml", "Local Transformer", "window=512"),
    ("E2", ROOT / "configs/encoder_study/e2_bigru_local.yaml", "BiGRU + Local Transformer", "GRU + window=256"),
    ("E3", ROOT / "configs/encoder_study/e3_bigru_block_global.yaml", "BiGRU + Block Global Transformer", "GRU + 128 blocks global"),
)
LABELS = ["Reentrancy", "Access Control", "Arithmetic", "Unchecked Return Values", "DoS", "Time manipulation"]
EXPECTED_P11 = 0.833507102823917


def json_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    base.atomic_json(path, value)


def reference_regression():
    if not P11_METRICS.exists() or not P11_CHECKPOINT.exists():
        raise FileNotFoundError("Historical P11 metrics/checkpoint are required for the A0 regression check")
    historical = json.loads(P11_METRICS.read_text(encoding="utf-8"))
    config = base.load_config(P11_CONFIG)
    train, valid = base.datasets(config)
    tokenizer = base.EVMOpcodeTokenizer.from_vocab_file(ROOT / config["vocab_path"])
    model = base.build_model("P11", config, len(tokenizer), tokenizer.pad_token_id).cuda()
    saved = torch.load(P11_CHECKPOINT, map_location="cpu")
    model.load_state_dict(saved["model_state_dict"])
    loader = base.make_loader(valid, config, tokenizer.pad_token_id, False)
    output = base.evaluate(model, "P11", loader, config, base.compute_weights(config, train).cuda(), False)
    metrics = base.metric_pack(config, output["labels"], output["logits"])
    observed = float(metrics["tuned"]["macro_f1"])
    if abs(observed - EXPECTED_P11) > 1e-8:
        raise RuntimeError(f"P11 validation regression failed: expected={EXPECTED_P11} observed={observed}")
    out = ROOT / "results/encoder_study/A0"
    out.mkdir(parents=True, exist_ok=True)
    base.atomic_save(out / "valid_predictions.pt", {"ids": output["ids"], "labels": output["labels"], "logits": output["logits"]})
    result = {
        "model": "A0",
        "encoder": "BiGRU",
        "attention_range": "recurrent full sequence",
        "metrics": metrics,
        "params": historical["params"],
        "encoder_params": historical.get("encoder_params"),
        "pdvq_params": historical.get("pdvq_params"),
        "best_epoch": historical["best_epoch"],
        "history": historical["history"],
        "inference_seconds": output["inference_seconds"],
        "regression_expected": EXPECTED_P11,
        "regression_observed": observed,
        "regression_passed": True,
        "test_checked": False,
    }
    json_write(out / "metrics.json", result)
    return result


def compact_row(name, encoder, attention, result):
    history = result.get("history", [])
    completed = [x for x in history if x.get("train_seconds")]
    return {
        "model": name,
        "encoder": encoder,
        "attention_range": attention,
        "macro_f1": float(result["metrics"]["tuned"]["macro_f1"]),
        "fixed_macro_f1": float(result["metrics"]["fixed"]["macro_f1"]),
        "micro_f1": float(result["metrics"]["tuned"]["micro_f1"]),
        "detection_f1": float(result["metrics"]["detection_f1"]),
        "params": int(result["params"]),
        "encoder_params": result.get("encoder_params"),
        "pdvq_params": result.get("pdvq_params"),
        "physical_batch": int(result.get("effective_config", {}).get("batch_size", 64)),
        "effective_batch": int(result.get("effective_config", {}).get("batch_size", 64)) * int(result.get("effective_config", {}).get("gradient_accumulation_steps", 4)),
        "peak_memory_mb": max((x.get("peak_memory_mb", 0.0) for x in completed), default=0.0),
        "seconds_per_epoch": float(np.mean([x["train_seconds"] for x in completed])) if completed else None,
        "samples_per_second": float(np.mean([x["contracts_per_second"] for x in completed])) if completed else None,
        "inference_seconds": float(result.get("inference_seconds", 0.0)),
        "best_epoch": int(result["best_epoch"]),
        "per_label_f1": result["metrics"]["tuned"]["per_label_f1"],
        "per_label_precision": result["metrics"]["tuned"]["per_label_precision"],
        "per_label_recall": result["metrics"]["tuned"]["per_label_recall"],
        "thresholds": result["metrics"]["thresholds"],
        "test_checked": False,
    }


def write_incremental(rows):
    out = ROOT / "results/encoder_study"
    out.mkdir(parents=True, exist_ok=True)
    reference = rows[0]["macro_f1"]
    payload = [dict(row, delta_vs_p11=row["macro_f1"] - reference) for row in rows]
    json_write(out / "main_results.json", payload)
    fields = ["model", "encoder", "attention_range", "macro_f1", "fixed_macro_f1", "micro_f1",
              "detection_f1", "delta_vs_p11", "params", "encoder_params", "pdvq_params", "physical_batch",
              "effective_batch", "peak_memory_mb", "seconds_per_epoch", "samples_per_second", "best_epoch", "test_checked"]
    with (out / "efficiency_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in payload)


def length_and_label_reports(rows):
    from metrics import compute_multilabel_metrics_from_probs, derived_detection_metrics_from_multilabel_probs

    cache = torch.load(ROOT / "data/features/light_label_process01/valid_max8192.pt", map_location="cpu")
    id_to_length = dict(zip(cache["ids"], cache["original_lengths"].tolist()))
    result_root = ROOT / "results/encoder_study"
    label_rows, length_rows = [], []
    prediction_paths = {"A0": result_root / "A0/valid_predictions.pt"}
    prediction_paths.update({name: ROOT / base.load_config(config)["result_root"] / "P11/valid_predictions.pt" for name, config, _, _ in STUDIES})
    bins = ((0, 2048), (2049, 4096), (4097, 6144), (6145, 8192))
    for row in rows:
        for index, label in enumerate(LABELS):
            label_rows.append({
                "model": row["model"], "label": label,
                "precision": row["per_label_precision"][index], "recall": row["per_label_recall"][index],
                "f1": row["per_label_f1"][index], "threshold": row["thresholds"][index],
            })
        predictions = torch.load(prediction_paths[row["model"]], map_location="cpu")
        lengths = np.asarray([id_to_length[str(value)] for value in predictions["ids"]])
        labels = predictions["labels"].numpy().astype(int)
        probabilities = predictions["logits"].sigmoid().numpy()
        thresholds = row["thresholds"]
        for lower, upper in bins:
            selected = (lengths >= lower) & (lengths <= upper)
            if not selected.any():
                continue
            metrics = compute_multilabel_metrics_from_probs(labels[selected], probabilities[selected], thresholds)
            detection = derived_detection_metrics_from_multilabel_probs(labels[selected], probabilities[selected], thresholds)
            length_rows.append({
                "model": row["model"], "length_bin": f"{lower}-{upper}", "count": int(selected.sum()),
                "macro_f1": float(metrics["recognition_macro_f1"]),
                "micro_f1": float(metrics["recognition_micro_f1"]),
                "detection_f1": float(detection["detection_f1"]),
            })
    for path, values in ((result_root / "per_label_results.csv", label_rows), (result_root / "length_bin_results.csv", length_rows)):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader(); writer.writerows(values)
    return length_rows


@torch.no_grad()
def best_transformer_diagnostics(rows):
    transformer = max(rows[1:], key=lambda item: item["macro_f1"])
    config_path = next(config for name, config, _, _ in STUDIES if name == transformer["model"])
    config = base.load_config(config_path)
    train, valid = base.datasets(config)
    tokenizer = base.EVMOpcodeTokenizer.from_vocab_file(ROOT / config["vocab_path"])
    model = base.build_model("P11", config, len(tokenizer), tokenizer.pad_token_id).cuda()
    checkpoint = torch.load(ROOT / config["checkpoint_root"] / "P11/best.pt", map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
    loader = DataLoader(valid, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=lambda items: base.collate_light_label(items, tokenizer.pad_token_id))
    overlaps = {k: [[] for _ in LABELS] for k in (1, 3, 5)}
    cosine = [[] for _ in LABELS]; margins = [[] for _ in LABELS]
    full_logits, labels = [], []
    positive_only, negative_only = [], []
    for batch in loader:
        out = base.forward(model, "P11", batch, "cuda", True)
        attention = out["attention"][0].float().cpu()
        valid = int(batch["mask"][0].sum())
        for label in range(len(LABELS)):
            cosine[label].append(float(torch.nn.functional.cosine_similarity(attention[label, 0, :valid], attention[label, 1, :valid], dim=0)))
            margins[label].append(float(out["energies"][0, label, 0] - out["energies"][0, label, 1]))
            for k in overlaps:
                size = min(k, valid)
                left = set(attention[label, 0, :valid].topk(size).indices.tolist())
                right = set(attention[label, 1, :valid].topk(size).indices.tolist())
                overlaps[k][label].append(len(left & right) / max(1, len(left | right)))
        representations = out["representations"].clone()
        altered = representations.clone(); altered[:, :, 1] = 0
        positive_only.append(model.score(altered)[0].float().cpu())
        altered = representations.clone(); altered[:, :, 0] = 0
        negative_only.append(model.score(altered)[0].float().cpu())
        full_logits.append(out["logits"].float().cpu()); labels.append(batch["labels"])
    targets = torch.cat(labels).numpy().astype(int)
    thresholds = transformer["thresholds"]
    from metrics import compute_multilabel_metrics_from_probs
    def measure(logits):
        value = compute_multilabel_metrics_from_probs(targets, torch.cat(logits).sigmoid().numpy(), thresholds)
        return {"macro_f1": float(value["recognition_macro_f1"]), "micro_f1": float(value["recognition_micro_f1"]),
                "per_label_f1": [float(x) for x in value["per_label_f1"]]}
    report = {
        "model": transformer["model"],
        "attention_cosine": {label: float(np.mean(cosine[i])) for i, label in enumerate(LABELS)},
        "topk_overlap_jaccard": {str(k): {label: float(np.mean(overlaps[k][i])) for i, label in enumerate(LABELS)} for k in overlaps},
        "margin_mean": {label: float(np.mean(margins[i])) for i, label in enumerate(LABELS)},
        "full_competition": measure(full_logits),
        "positive_only": measure(positive_only),
        "negative_only": measure(negative_only),
        "test_checked": False,
        "interpretation": "Model sensitivity diagnostics, not evidence ground truth.",
    }
    json_write(ROOT / "results/encoder_study/best_transformer_query_diagnostics.json", report)
    return report


def final_report(rows, length_rows, diagnostics):
    reference = rows[0]["macro_f1"]
    best = max(rows, key=lambda item: item["macro_f1"])
    delta = best["macro_f1"] - reference
    verdict = "KEEP_P11" if best["model"] == "A0" else f"UPGRADE_TO_{best['model'].replace('_256', '').replace('_512', '')}"
    report_dir = ROOT / "reports/encoder_study"; report_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# PDVQ Encoder Architecture Study", "", "process01; seed=42; validation only; test_checked=false.", "",
             "| Model | Encoder | Attention range | Macro-F1 | Fixed Macro-F1 | Micro-F1 | Detection-F1 | Params | VRAM MB | sec/epoch |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['model']} | {row['encoder']} | {row['attention_range']} | {row['macro_f1']:.6f} | "
                     f"{row['fixed_macro_f1']:.6f} | {row['micro_f1']:.6f} | {row['detection_f1']:.6f} | "
                     f"{row['params']:,} | {row['peak_memory_mb']:.1f} | {row['seconds_per_epoch'] or 0:.1f} |")
    lines.extend(["", f"BEST_ENCODER: {best['model']}", f"DELTA_VS_P11: {delta:+.6f}", f"VERDICT: {verdict}", "",
                  "The local implementations use O(TW) window attention; no full 8192x8192 token attention matrix is constructed.",
                  f"Best Transformer diagnostic target: {diagnostics['model']}.", ""])
    (report_dir / "final_report.md").write_text("\n".join(lines), encoding="utf-8")
    json_write(ROOT / "results/encoder_study/decision.json", {"best_encoder": best["model"], "delta_vs_p11": delta,
               "verdict": verdict, "test_checked": False})


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    rows = [compact_row("A0", "BiGRU", "recurrent full sequence", reference_regression())]
    write_incremental(rows)
    for name, config_path, encoder, attention in STUDIES:
        print(f"[encoder-study] starting {name} config={config_path}", flush=True)
        base.CONFIG = config_path
        base.VARIANTS = ("P11",)
        sys.argv = [sys.argv[0], "--config", str(config_path)]
        base.main()
        config = base.load_config(config_path)
        result = json.loads((ROOT / config["result_root"] / "P11/metrics.json").read_text(encoding="utf-8"))
        rows.append(compact_row(name, encoder, attention, result))
        write_incremental(rows)
        print(f"[encoder-study] completed {name} macro={rows[-1]['macro_f1']:.6f}", flush=True)
    length_rows = length_and_label_reports(rows)
    diagnostics = best_transformer_diagnostics(rows)
    final_report(rows, length_rows, diagnostics)
    print("[encoder-study] complete test_checked=false", flush=True)


if __name__ == "__main__":
    main()
