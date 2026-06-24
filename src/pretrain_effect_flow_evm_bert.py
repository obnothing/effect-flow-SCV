import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from effect_flow_pretraining_dataset import (  # noqa: E402
    BalancedDistributedBatchSampler,
    EffectFlowChunkDataset,
    build_or_load_corpus_index,
    calculate_balanced_class_weights,
    choose_internal_holdout,
    count_offsets_labels,
    load_pattern_subset,
    load_tokenizer,
    subtract_stats,
)
from effect_flow_pretraining_model import EffectFlowBertForPreTraining  # noqa: E402
from effect_flow_schema import EFFECT_TYPES  # noqa: E402


DATASET_ORDER = ["BJUT", "DIVE"]
TRANSDUCTIVE_WARNING = (
    "The base encoder was initialized from BJUT full-corpus EVM-BERT and is "
    "therefore a transductive initialization. Stage 16B itself uses only "
    "BJUT-train and DIVE-train effect-flow chunks."
)


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 16B effect-flow pretraining.")
    parser.add_argument(
        "--config",
        default="configs/pretrain_effect_flow_evm_bert_bjut_dive_train_balanced.yaml",
    )
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def relative(path):
    path = Path(path)
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def load_config(path):
    config_path = resolve(path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    for key in (
        "base_hf_model_path",
        "vocab_path",
        "efpp_pattern_config",
        "train_manifest_path",
        "internal_valid_manifest_path",
        "checkpoint_dir",
        "result_dir",
        "report_dir",
        "log_dir",
    ):
        config[key] = str(resolve(config[key]))
    config["train_corpora"] = {
        name: str(resolve(path)) for name, path in config["train_corpora"].items()
    }
    config["_config_path"] = str(config_path)
    return config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Stage 16B DDP requires CUDA.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    return distributed, rank, local_rank, world_size


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def unwrap(model):
    return model.module if hasattr(model, "module") else model


def encoder_checksum(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(unwrap(model).bert.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main_print(rank, message):
    if rank == 0:
        print(message, flush=True)


def limit_offsets(offsets, limit, seed):
    if limit is None or int(limit) >= len(offsets):
        return np.asarray(offsets, dtype=np.int64)
    rng = random.Random(seed)
    positions = sorted(rng.sample(range(len(offsets)), int(limit)))
    return np.asarray([offsets[index] for index in positions], dtype=np.int64)


def write_manifests(config, source_states, seed):
    train_entries = []
    valid_entries = []
    for name in DATASET_ORDER:
        state = source_states[name]
        common = {
            "source_dataset": name,
            "source_file": relative(state["path"]),
            "sample_count": int(state["full_count"]),
            "seed": seed,
        }
        train_entries.append(
            {
                **common,
                "selected_count": int(len(state["train_offsets"])),
                "runtime_selected_count": int(len(state["runtime_train_offsets"])),
                "split": "train",
            }
        )
        valid_entries.append(
            {
                **common,
                "selected_count": int(len(state["valid_offsets"])),
                "runtime_selected_count": int(len(state["runtime_valid_offsets"])),
                "split": "internal_valid",
            }
        )
    payloads = [
        (Path(config["train_manifest_path"]), {"entries": train_entries}),
        (Path(config["internal_valid_manifest_path"]), {"entries": valid_entries}),
    ]
    for path, payload in payloads:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def prepare_data(config, distributed, rank, seed):
    checkpoint_dir = Path(config["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        for name in DATASET_ORDER:
            path = Path(config["train_corpora"][name])
            print(f"[INDEX] {name}: {relative(path)}", flush=True)
            build_or_load_corpus_index(
                path,
                progress_callback=lambda count, n=name: print(
                    f"[INDEX] {n}: {count} chunks", flush=True
                ),
            )
    if distributed:
        dist.barrier()

    source_states = {}
    stats_by_dataset = {}
    ratio = float(config.get("internal_valid_ratio", 0.01))
    debug_train = config.get("debug_train_samples_per_dataset")
    debug_valid = config.get("debug_valid_samples_per_dataset")
    for dataset_index, name in enumerate(DATASET_ORDER):
        path = Path(config["train_corpora"][name])
        offsets, full_stats = build_or_load_corpus_index(path)
        expected = int(config.get("expected_train_chunks", {}).get(name, len(offsets)))
        if len(offsets) != expected:
            raise ValueError(
                f"{name} train chunk count mismatch: expected {expected}, got {len(offsets)}. "
                "For DIVE, rebuild the max-64 corpus before Stage 16B."
            )
        train_offsets, valid_offsets = choose_internal_holdout(
            offsets, ratio, seed + dataset_index * 100003
        )
        if rank == 0:
            heldout_stats = count_offsets_labels(path, valid_offsets)
            stats_by_dataset[name] = subtract_stats(full_stats, heldout_stats)
        source_states[name] = {
            "path": path,
            "full_count": len(offsets),
            "train_offsets": train_offsets,
            "valid_offsets": valid_offsets,
            "runtime_train_offsets": limit_offsets(
                train_offsets, debug_train, seed + dataset_index * 313
            ),
            "runtime_valid_offsets": limit_offsets(
                valid_offsets, debug_valid, seed + dataset_index * 919
            ),
        }

    state_path = checkpoint_dir / "stage16b_data_state.json"
    if rank == 0:
        pattern_config, original_indices = load_pattern_subset(
            config["efpp_pattern_config"]
        )
        weights = calculate_balanced_class_weights(
            stats_by_dataset,
            original_indices,
            max_etp_class_weight=float(config.get("max_etp_class_weight", 5.0)),
            max_efpp_pos_weight=float(config.get("max_efpp_pos_weight", 10.0)),
            normal_class_weight_scale=float(
                config.get("normal_class_weight_scale", 0.5)
            ),
        )
        data_state = {
            "stats_by_dataset": stats_by_dataset,
            "weights": weights,
            "pattern_config": pattern_config,
            "included_original_pattern_indices": original_indices,
        }
        state_path.write_text(
            json.dumps(data_state, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        write_manifests(config, source_states, seed)
    if distributed:
        dist.barrier()
    data_state = json.loads(state_path.read_text(encoding="utf-8"))
    return source_states, data_state


def metric_rows(tp, fp, fn, names):
    rows = []
    for index, name in enumerate(names):
        precision = tp[index] / (tp[index] + fp[index]) if tp[index] + fp[index] else 0.0
        recall = tp[index] / (tp[index] + fn[index]) if tp[index] + fn[index] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "name": name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(tp[index] + fn[index]),
                "predicted_positive_count": int(tp[index] + fp[index]),
            }
        )
    return rows


@torch.no_grad()
def evaluate(model, dataloader, device, pattern_names, fp16, bf16):
    model.eval()
    totals = {key: 0.0 for key in ("total", "mom", "etp", "efpp")}
    sample_count = 0
    mom_correct = mom_top5 = mom_count = 0
    etp_tp = np.zeros(len(EFFECT_TYPES), dtype=np.int64)
    etp_fp = np.zeros(len(EFFECT_TYPES), dtype=np.int64)
    etp_fn = np.zeros(len(EFFECT_TYPES), dtype=np.int64)
    efpp_tp = np.zeros(len(pattern_names), dtype=np.int64)
    efpp_fp = np.zeros(len(pattern_names), dtype=np.int64)
    efpp_fn = np.zeros(len(pattern_names), dtype=np.int64)
    efpp_true_all = []
    efpp_prob_all = []
    dtype = torch.bfloat16 if bf16 else torch.float16
    for batch in tqdm(dataloader, desc="internal_valid", leave=False):
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.cuda.amp.autocast(
            enabled=(fp16 or bf16), dtype=dtype
        ):
            outputs = model(**batch)
        size = batch["input_ids"].size(0)
        sample_count += size
        for key, output_key in (
            ("total", "loss"),
            ("mom", "mom_loss"),
            ("etp", "etp_loss"),
            ("efpp", "efpp_loss"),
        ):
            totals[key] += float(outputs[output_key].detach().cpu()) * size

        mom_mask = batch["mom_labels"] != -100
        if mom_mask.any():
            logits = outputs["mom_logits"][mom_mask]
            truth = batch["mom_labels"][mom_mask]
            mom_correct += int((logits.argmax(dim=-1) == truth).sum().item())
            topk = logits.topk(min(5, logits.size(-1)), dim=-1).indices
            mom_top5 += int((topk == truth.unsqueeze(-1)).any(dim=-1).sum().item())
            mom_count += int(truth.numel())

        etp_mask = batch["etp_labels"] != -100
        etp_pred = outputs["etp_logits"].argmax(dim=-1)
        etp_true = batch["etp_labels"]
        for index in range(len(EFFECT_TYPES)):
            pred = (etp_pred == index) & etp_mask
            true = (etp_true == index) & etp_mask
            etp_tp[index] += int((pred & true).sum().item())
            etp_fp[index] += int((pred & ~true).sum().item())
            etp_fn[index] += int((~pred & true).sum().item())

        efpp_prob = torch.sigmoid(outputs["efpp_logits"])
        efpp_pred = efpp_prob >= 0.5
        efpp_true = batch["efpp_labels"].bool()
        efpp_tp += (efpp_pred & efpp_true).sum(dim=0).cpu().numpy()
        efpp_fp += (efpp_pred & ~efpp_true).sum(dim=0).cpu().numpy()
        efpp_fn += (~efpp_pred & efpp_true).sum(dim=0).cpu().numpy()
        efpp_true_all.append(efpp_true.cpu().numpy().astype(np.int8))
        efpp_prob_all.append(efpp_prob.float().cpu().numpy())

    etp_rows = metric_rows(etp_tp, etp_fp, etp_fn, EFFECT_TYPES)
    efpp_rows = metric_rows(efpp_tp, efpp_fp, efpp_fn, pattern_names)
    etp_macro = float(np.mean([row["f1"] for row in etp_rows]))
    etp_macro_without_normal = float(np.mean([row["f1"] for row in etp_rows[1:]]))
    etp_accuracy = float(etp_tp.sum() / max(1, etp_tp.sum() + etp_fn.sum()))
    efpp_micro = float(
        2 * efpp_tp.sum()
        / max(1, 2 * efpp_tp.sum() + efpp_fp.sum() + efpp_fn.sum())
    )
    efpp_macro = float(np.mean([row["f1"] for row in efpp_rows]))
    efpp_auc = None
    try:
        from sklearn.metrics import roc_auc_score

        true_array = np.concatenate(efpp_true_all, axis=0)
        prob_array = np.concatenate(efpp_prob_all, axis=0)
        valid_columns = [
            index
            for index in range(true_array.shape[1])
            if np.unique(true_array[:, index]).size == 2
        ]
        if valid_columns:
            efpp_auc = float(
                roc_auc_score(
                    true_array[:, valid_columns],
                    prob_array[:, valid_columns],
                    average="macro",
                )
            )
    except Exception:
        efpp_auc = None
    return {
        "evaluated_samples": sample_count,
        "valid_loss_total": totals["total"] / max(1, sample_count),
        "valid_loss_mom": totals["mom"] / max(1, sample_count),
        "valid_loss_etp": totals["etp"] / max(1, sample_count),
        "valid_loss_efpp": totals["efpp"] / max(1, sample_count),
        "masked_token_accuracy": mom_correct / max(1, mom_count),
        "masked_token_top5_accuracy": mom_top5 / max(1, mom_count),
        "masked_token_count": mom_count,
        "etp_accuracy": etp_accuracy,
        "etp_macro_f1": etp_macro,
        "etp_macro_f1_excluding_normal": etp_macro_without_normal,
        "efpp_micro_f1": efpp_micro,
        "efpp_macro_f1": efpp_macro,
        "efpp_macro_auc": efpp_auc,
        "etp_per_effect": etp_rows,
        "efpp_per_pattern": efpp_rows,
    }


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    global_step,
    best_valid_loss,
    best_epoch,
    best_encoder_checksum,
    best_metrics,
    history,
    config,
    checksum_before,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": unwrap(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_valid_loss": best_valid_loss,
            "best_epoch": best_epoch,
            "best_encoder_checksum": best_encoder_checksum,
            "best_metrics": best_metrics,
            "history": history,
            "config": config,
            "encoder_checksum_before": checksum_before,
        },
        path,
    )


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            [{field: row.get(field) for field in fieldnames} for row in rows]
        )


def plot_loss_curves(path, history):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, [row["train_loss_total"] for row in history], marker="o", label="train")
    axes[0].plot(epochs, [row["valid_loss_total"] for row in history], marker="o", label="internal valid")
    axes[0].set_title("Total pretraining loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    for task, color in (("mom", "#2563eb"), ("etp", "#dc2626"), ("efpp", "#16a34a")):
        axes[1].plot(
            epochs,
            [row[f"train_loss_{task}"] for row in history],
            color=color,
            marker="o",
            label=f"train {task.upper()}",
        )
        axes[1].plot(
            epochs,
            [row[f"valid_loss_{task}"] for row in history],
            color=color,
            linestyle="--",
            marker="x",
            label=f"valid {task.upper()}",
        )
    axes[1].set_title("MOM / ETP / EFPP losses")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(fontsize=8, ncol=2)
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_outputs(config, report, history, metrics):
    result_dir = Path(config["result_dir"])
    report_dir = Path(config["report_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    log_fields = [
        "epoch",
        "global_step",
        "train_loss_total",
        "train_loss_mom",
        "train_loss_etp",
        "train_loss_efpp",
        "valid_loss_total",
        "valid_loss_mom",
        "valid_loss_etp",
        "valid_loss_efpp",
        "masked_token_accuracy",
        "masked_token_top5_accuracy",
        "etp_accuracy",
        "etp_macro_f1",
        "etp_macro_f1_excluding_normal",
        "efpp_micro_f1",
        "efpp_macro_f1",
        "efpp_macro_auc",
        "bjut_samples",
        "dive_samples",
        "learning_rate",
        "max_memory_allocated_mb",
        "max_memory_reserved_mb",
    ]
    write_csv(result_dir / "training_log.csv", log_fields, history)
    (result_dir / "internal_valid_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_csv(
        result_dir / "etp_per_effect_metrics.csv",
        list(metrics["etp_per_effect"][0]),
        metrics["etp_per_effect"],
    )
    write_csv(
        result_dir / "efpp_per_pattern_metrics.csv",
        list(metrics["efpp_per_pattern"][0]),
        metrics["efpp_per_pattern"],
    )
    plot_loss_curves(result_dir / "loss_curves.png", history)
    report_name = config["report_filename"]
    json_path = report_dir / f"{report_name}.json"
    txt_path = report_dir / f"{report_name}.txt"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "Stage 16B Effect-Flow-Aware EVM-BERT continued pretraining report",
        "",
        f"experiment_name: {report['experiment_name']}",
        f"base_hf_model_path: {report['base_hf_model_path']}",
        f"base_encoder_init: {report['base_encoder_init']}",
        f"base_encoder_init_setting: {report['base_encoder_init_setting']}",
        f"train_corpora: {report['train_corpora']}",
        f"internal_valid_ratio: {report['internal_valid_ratio']}",
        f"internal_valid_counts_by_dataset: {report['internal_valid_counts_by_dataset']}",
        f"dataset_balanced_sampling_ratio: {report['dataset_balanced_sampling_ratio']}",
        f"actual_sample_counts: {report['actual_sample_counts']}",
        f"efpp_pattern_subset: {report['efpp_pattern_subset']}",
        f"included_patterns: {report['included_patterns']}",
        f"excluded_patterns: {report['excluded_patterns']}",
        f"excluded_reasons: {report['excluded_reasons']}",
        f"loss_weights: {report['loss_weights']}",
        f"training_hyperparameters: {report['training_hyperparameters']}",
        f"encoder_checksum_before: {report['encoder_checksum_before']}",
        f"encoder_checksum_after: {report['encoder_checksum_after']}",
        f"encoder_checksum_changed: {report['encoder_checksum_changed']}",
        f"best_valid_loss: {report['best_valid_loss']}",
        f"best_epoch: {report['best_epoch']}",
        f"checkpoint_paths: {report['checkpoint_paths']}",
        f"hf_model_path: {report['hf_model_path']}",
        f"result_paths: {report['result_paths']}",
        f"sanity_success: {report['sanity_success']}",
        f"full_pretraining_success: {report['full_pretraining_success']}",
        f"recommendation_for_stage16c: {report['recommendation_for_stage16c']}",
        "",
        "Epoch loss table:",
    ]
    for row in history:
        lines.append(
            "epoch={epoch} train_total={train_loss_total:.6f} train_mom={train_loss_mom:.6f} "
            "train_etp={train_loss_etp:.6f} train_efpp={train_loss_efpp:.6f} "
            "valid_total={valid_loss_total:.6f} valid_mom={valid_loss_mom:.6f} "
            "valid_etp={valid_loss_etp:.6f} valid_efpp={valid_loss_efpp:.6f}".format(**row)
        )
    lines.extend(
        [
            "",
            f"MOM metrics: accuracy={metrics['masked_token_accuracy']:.6f}, "
            f"top5_accuracy={metrics['masked_token_top5_accuracy']:.6f}",
            f"ETP metrics: accuracy={metrics['etp_accuracy']:.6f}, "
            f"macro_f1={metrics['etp_macro_f1']:.6f}, "
            f"macro_f1_excluding_normal={metrics['etp_macro_f1_excluding_normal']:.6f}",
            f"EFPP metrics: micro_f1={metrics['efpp_micro_f1']:.6f}, "
            f"macro_f1={metrics['efpp_macro_f1']:.6f}, macro_auc={metrics['efpp_macro_auc']}",
            "",
            "ETP per-effect metrics:",
            "effect | precision | recall | f1 | support",
            *[
                f"{row['name']} | {row['precision']:.6f} | {row['recall']:.6f} | "
                f"{row['f1']:.6f} | {row['support']}"
                for row in metrics["etp_per_effect"]
            ],
            "",
            "EFPP per-pattern metrics:",
            "pattern | precision | recall | f1 | support",
            *[
                f"{row['name']} | {row['precision']:.6f} | {row['recall']:.6f} | "
                f"{row['f1']:.6f} | {row['support']}"
                for row in metrics["efpp_per_pattern"]
            ],
            "",
            "Warnings:",
            *[f"- {warning}" for warning in report["warnings"]],
        ]
    )
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote {relative(txt_path)}")


def main():
    args = parse_args()
    config = load_config(args.config)
    distributed, rank, local_rank, world_size = setup_distributed()
    seed = int(config.get("seed", 42))
    set_seed(seed + rank)
    if config.get("use_downstream_labels"):
        raise ValueError("Stage 16B forbids downstream vulnerability labels.")
    if config.get("use_original_valid_test_chunks"):
        raise ValueError("Stage 16B forbids original valid/test chunks.")
    if config.get("efpp_pattern_subset") != "conservative_22":
        raise ValueError("Stage 16B requires efpp_pattern_subset=conservative_22.")

    checkpoint_dir = Path(config["checkpoint_dir"])
    result_dir = Path(config["result_dir"])
    if rank == 0:
        for path in (checkpoint_dir, result_dir, Path(config["report_dir"]), Path(config["log_dir"])):
            path.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config["_config_path"], checkpoint_dir / "config_snapshot.yaml")
        shutil.copy2(config["efpp_pattern_config"], checkpoint_dir / "effect_flow_efpp_conservative_22.json")
    if distributed:
        dist.barrier()

    source_states, data_state = prepare_data(config, distributed, rank, seed)
    pattern_config = data_state["pattern_config"]
    pattern_names = pattern_config["included_pattern_names"]
    original_indices = data_state["included_original_pattern_indices"]
    weights = data_state["weights"]
    tokenizer = load_tokenizer(config["vocab_path"])
    vocab_size = len(tokenizer)

    train_sources = [
        {
            "name": name,
            "path": source_states[name]["path"],
            "offsets": source_states[name]["runtime_train_offsets"],
        }
        for name in DATASET_ORDER
    ]
    valid_sources = [
        {
            "name": name,
            "path": source_states[name]["path"],
            "offsets": source_states[name]["runtime_valid_offsets"],
        }
        for name in DATASET_ORDER
    ]
    train_dataset = EffectFlowChunkDataset(
        train_sources,
        tokenizer,
        original_indices,
        mlm_probability=float(config.get("mlm_probability", 0.15)),
        seed=seed,
    )
    valid_dataset = EffectFlowChunkDataset(
        valid_sources,
        tokenizer,
        original_indices,
        mlm_probability=float(config.get("mlm_probability", 0.15)),
        seed=seed + 700001,
    )
    batch_size = int(config["batch_size"])
    grad_accum = int(config.get("gradient_accumulation_steps", 1))
    steps_per_epoch = int(config["steps_per_epoch"])
    sampler = BalancedDistributedBatchSampler(
        [len(source["offsets"]) for source in train_sources],
        batch_size=batch_size,
        optimizer_steps_per_epoch=steps_per_epoch,
        gradient_accumulation_steps=grad_accum,
        seed=seed,
        rank=rank,
        world_size=world_size,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=min(4, int(config.get("num_workers", 4))),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(config.get("num_workers", 4)) > 0,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=int(config.get("eval_batch_size", batch_size)),
        shuffle=False,
        num_workers=min(4, int(config.get("num_workers", 4))),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(config.get("num_workers", 4)) > 0,
    )

    model = EffectFlowBertForPreTraining(
        config["base_hf_model_path"],
        vocab_size,
        num_effect_types=int(config.get("num_effect_types", 16)),
        num_efpp_patterns=len(pattern_names),
        lambda_mom=float(config.get("lambda_mom", 1.0)),
        lambda_etp=float(config.get("lambda_etp", 0.3)),
        lambda_efpp=float(config.get("lambda_efpp", 0.5)),
        etp_class_weights=weights["etp_class_weights"],
        efpp_pos_weights=weights["efpp_pos_weights"],
    )
    if config.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
    checksum_before = encoder_checksum(model) if rank == 0 else None
    device = torch.device(f"cuda:{local_rank}" if distributed else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        betas=(float(config.get("adam_beta1", 0.9)), float(config.get("adam_beta2", 0.999))),
        eps=float(config.get("adam_epsilon", 1e-8)),
        weight_decay=float(config.get("weight_decay", 0.01)),
    )
    epochs = int(config["epochs"])
    total_steps = steps_per_epoch * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * float(config.get("warmup_ratio", 0.06))),
        num_training_steps=total_steps,
    )
    fp16 = bool(config.get("fp16", True)) and torch.cuda.is_available()
    bf16 = bool(config.get("bf16", False)) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=fp16 and not bf16)
    dtype = torch.bfloat16 if bf16 else torch.float16
    start_epoch = 1
    global_step = 0
    best_valid_loss = float("inf")
    best_epoch = None
    best_checksum = checksum_before
    best_metrics = None
    history = []
    resume_path = args.resume or config.get("resume_from")
    if resume_path:
        checkpoint = torch.load(resolve(resume_path), map_location="cpu")
        unwrap(model).load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        best_valid_loss = float(checkpoint.get("best_valid_loss", float("inf")))
        best_epoch = checkpoint.get("best_epoch")
        best_checksum = checkpoint.get("best_encoder_checksum", checksum_before)
        best_metrics = checkpoint.get("best_metrics")
        history = checkpoint.get("history", [])
        main_print(rank, f"[RESUME] {relative(resolve(resume_path))} at epoch {start_epoch}")

    main_print(rank, f"[INFO] base_hf_model_path: {relative(config['base_hf_model_path'])}")
    main_print(rank, f"[INFO] base_encoder_init_setting: {config['base_encoder_init_setting']}")
    main_print(rank, f"[INFO] train chunks: BJUT={len(train_sources[0]['offsets'])}, DIVE={len(train_sources[1]['offsets'])}")
    main_print(rank, f"[INFO] internal valid: BJUT={len(valid_sources[0]['offsets'])}, DIVE={len(valid_sources[1]['offsets'])}")
    main_print(rank, f"[INFO] EFPP subset: conservative_22 ({len(pattern_names)} patterns)")
    main_print(rank, f"[INFO] world_size={world_size} per_gpu_batch={batch_size} global_effective_batch={batch_size * grad_accum * world_size}")
    main_print(rank, f"[INFO] optimizer_steps_per_epoch={steps_per_epoch} epochs={epochs}")
    main_print(rank, f"[WARNING] {TRANSDUCTIVE_WARNING}")

    last_metrics = None
    for epoch in range(start_epoch, epochs + 1):
        sampler.set_epoch(epoch)
        train_dataset.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulators = torch.zeros(7, dtype=torch.float64, device=device)
        progress = tqdm(train_loader, desc=f"stage16b {epoch}/{epochs}", disable=rank != 0)
        for micro_step, batch in enumerate(progress, start=1):
            source_ids = batch["source_dataset_id"]
            bjut_count = int((source_ids == 0).sum().item())
            dive_count = int((source_ids == 1).sum().item())
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            should_step = micro_step % grad_accum == 0
            sync_context = (
                contextlib.nullcontext()
                if should_step or not distributed
                else model.no_sync()
            )
            with sync_context:
                with torch.cuda.amp.autocast(enabled=(fp16 or bf16), dtype=dtype):
                    outputs = model(**batch)
                    loss = outputs["loss"]
                    scaled_loss = loss / grad_accum
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite Stage 16B loss at micro-step {micro_step}")
                scaler.scale(scaled_loss).backward()
            size = batch["input_ids"].size(0)
            accumulators += torch.tensor(
                [
                    float(outputs["loss"].detach()) * size,
                    float(outputs["mom_loss"].detach()) * size,
                    float(outputs["etp_loss"].detach()) * size,
                    float(outputs["efpp_loss"].detach()) * size,
                    size,
                    bjut_count,
                    dive_count,
                ],
                dtype=torch.float64,
                device=device,
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("max_grad_norm", 1.0)))
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if rank == 0 and global_step % int(config.get("logging_steps", 100)) == 0:
                    print(
                        f"[INFO] step={global_step} total={float(outputs['loss']):.6f} "
                        f"mom={float(outputs['mom_loss']):.6f} etp={float(outputs['etp_loss']):.6f} "
                        f"efpp={float(outputs['efpp_loss']):.6f} lr={scheduler.get_last_lr()[0]:.8g}",
                        flush=True,
                    )
            progress.set_postfix(total=float(loss.detach()))

        if distributed:
            dist.all_reduce(accumulators, op=dist.ReduceOp.SUM)
        samples = max(1.0, float(accumulators[4].item()))
        train_values = {
            "train_loss_total": float(accumulators[0].item() / samples),
            "train_loss_mom": float(accumulators[1].item() / samples),
            "train_loss_etp": float(accumulators[2].item() / samples),
            "train_loss_efpp": float(accumulators[3].item() / samples),
            "bjut_samples": int(accumulators[5].item()),
            "dive_samples": int(accumulators[6].item()),
        }
        if distributed:
            dist.barrier()
        metrics = None
        if rank == 0:
            valid_dataset.set_epoch(0)
            metrics = evaluate(
                unwrap(model), valid_loader, device, pattern_names, fp16, bf16
            )
            last_metrics = metrics
            allocated = torch.cuda.max_memory_allocated(device) / 1024 / 1024 if torch.cuda.is_available() else 0.0
            reserved = torch.cuda.max_memory_reserved(device) / 1024 / 1024 if torch.cuda.is_available() else 0.0
            row = {
                "epoch": epoch,
                "global_step": global_step,
                **train_values,
                **{key: value for key, value in metrics.items() if key not in {"etp_per_effect", "efpp_per_pattern"}},
                "learning_rate": scheduler.get_last_lr()[0],
                "max_memory_allocated_mb": allocated,
                "max_memory_reserved_mb": reserved,
            }
            history.append(row)
            print(
                f"epoch {epoch}/{epochs} train_total={row['train_loss_total']:.6f} "
                f"valid_total={row['valid_loss_total']:.6f} mom_acc={row['masked_token_accuracy']:.6f} "
                f"etp_macro_no_normal={row['etp_macro_f1_excluding_normal']:.6f} "
                f"efpp_micro={row['efpp_micro_f1']:.6f} efpp_macro={row['efpp_macro_f1']:.6f} "
                f"BJUT/DIVE={row['bjut_samples']}/{row['dive_samples']}",
                flush=True,
            )
            is_best = metrics["valid_loss_total"] < best_valid_loss
            if is_best:
                best_valid_loss = metrics["valid_loss_total"]
                best_epoch = epoch
                best_metrics = metrics
                best_checksum = encoder_checksum(model)
                save_checkpoint(
                    checkpoint_dir / "best_valid_loss.pt",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    global_step,
                    best_valid_loss,
                    best_epoch,
                    best_checksum,
                    best_metrics,
                    history,
                    config,
                    checksum_before,
                )
                metadata = {
                    "efpp_pattern_subset": "conservative_22",
                    "included_pattern_names": pattern_names,
                    "base_encoder_init_setting": config["base_encoder_init_setting"],
                    "continued_pretraining_train_only": True,
                }
                unwrap(model).save_hf_model(checkpoint_dir / "hf_model", metadata)
                shutil.copy2(config["vocab_path"], checkpoint_dir / "hf_model" / "evm_vocab.json")
            save_checkpoint(
                checkpoint_dir / "last.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                global_step,
                best_valid_loss,
                best_epoch,
                best_checksum,
                best_metrics,
                history,
                config,
                checksum_before,
            )
        if distributed:
            dist.barrier()

    if rank == 0:
        if last_metrics is None:
            raise RuntimeError("No internal validation metrics were produced.")
        total_bjut = sum(row["bjut_samples"] for row in history)
        total_dive = sum(row["dive_samples"] for row in history)
        total_seen = max(1, total_bjut + total_dive)
        losses_valid = all(
            math.isfinite(row[key]) and row[key] > 0
            for row in history
            for key in (
                "train_loss_total",
                "train_loss_mom",
                "train_loss_etp",
                "train_loss_efpp",
                "valid_loss_total",
                "valid_loss_mom",
                "valid_loss_etp",
                "valid_loss_efpp",
            )
        )
        balanced_ratio = total_bjut / total_seen
        sanity_success = bool(
            losses_valid
            and abs(balanced_ratio - 0.5) <= 0.01
            and (checkpoint_dir / "best_valid_loss.pt").exists()
        )
        is_sanity = config["experiment_name"].startswith("sanity_")
        selected_metrics = best_metrics or last_metrics
        full_success = bool(
            not is_sanity
            and losses_valid
            and len(history) >= 2
            and history[-1]["train_loss_total"] < history[0]["train_loss_total"]
            and history[-1]["train_loss_mom"] < history[0]["train_loss_mom"]
            and selected_metrics["etp_macro_f1_excluding_normal"] > 0
            and selected_metrics["efpp_macro_f1"] > 0
            and sum(row["f1"] > 0 for row in selected_metrics["efpp_per_pattern"])
            >= len(pattern_names) // 2
            and (checkpoint_dir / "hf_model" / "config.json").exists()
        )
        report = {
            "experiment_name": config["experiment_name"],
            "base_hf_model_path": relative(config["base_hf_model_path"]),
            "base_encoder_init": config["base_encoder_init"],
            "base_encoder_init_setting": config["base_encoder_init_setting"],
            "train_corpora": {
                name: {
                    "path": relative(source_states[name]["path"]),
                    "chunks": source_states[name]["full_count"],
                    "train_after_holdout": len(source_states[name]["train_offsets"]),
                }
                for name in DATASET_ORDER
            },
            "internal_valid_ratio": float(config["internal_valid_ratio"]),
            "internal_valid_counts_by_dataset": {
                name: len(source_states[name]["valid_offsets"])
                for name in DATASET_ORDER
            },
            "dataset_balanced_sampling_ratio": config["dataset_sampling_ratio"],
            "actual_sample_counts": {
                "BJUT": total_bjut,
                "DIVE": total_dive,
                "BJUT_ratio": total_bjut / total_seen,
                "DIVE_ratio": total_dive / total_seen,
            },
            "efpp_pattern_subset": "conservative_22",
            "included_patterns": pattern_names,
            "excluded_patterns": pattern_config["excluded_pattern_names"],
            "excluded_reasons": pattern_config["excluded_reasons"],
            "loss_weights": {
                "lambda_mom": float(config["lambda_mom"]),
                "lambda_etp": float(config["lambda_etp"]),
                "lambda_efpp": float(config["lambda_efpp"]),
                "etp_class_weights": weights["etp_class_weights"],
                "efpp_pos_weights": weights["efpp_pos_weights"],
            },
            "training_hyperparameters": {
                key: config[key]
                for key in (
                    "batch_size",
                    "eval_batch_size",
                    "gradient_accumulation_steps",
                    "steps_per_epoch",
                    "epochs",
                    "learning_rate",
                    "warmup_ratio",
                    "max_grad_norm",
                    "mlm_probability",
                    "num_workers",
                    "fp16",
                    "bf16",
                )
            },
            "epoch_loss_table": history,
            "final_internal_valid_metrics": last_metrics,
            "best_internal_valid_metrics": best_metrics,
            "encoder_checksum_before": checksum_before,
            "encoder_checksum_after": best_checksum,
            "encoder_checksum_changed": checksum_before != best_checksum,
            "best_valid_loss": best_valid_loss,
            "best_epoch": best_epoch,
            "checkpoint_paths": {
                "best_valid_loss": relative(checkpoint_dir / "best_valid_loss.pt"),
                "last": relative(checkpoint_dir / "last.pt"),
            },
            "hf_model_path": relative(checkpoint_dir / "hf_model"),
            "result_paths": {
                "training_log": relative(Path(config["result_dir"]) / "training_log.csv"),
                "internal_valid_metrics": relative(Path(config["result_dir"]) / "internal_valid_metrics.json"),
                "etp_per_effect_metrics": relative(Path(config["result_dir"]) / "etp_per_effect_metrics.csv"),
                "efpp_per_pattern_metrics": relative(Path(config["result_dir"]) / "efpp_per_pattern_metrics.csv"),
                "loss_curves": relative(Path(config["result_dir"]) / "loss_curves.png"),
            },
            "sanity_success": sanity_success,
            "full_pretraining_success": full_success,
            "warnings": [
                TRANSDUCTIVE_WARNING,
                "Continued pretraining uses train chunks only.",
                "No downstream vulnerability labels were used.",
                "MOM/ETP/EFPP metrics are pretraining-task metrics, not vulnerability-detection metrics.",
            ],
            "recommendation_for_stage16c": (
                config["stage16c_recommendation"] if full_success else "Do not start Stage 16C until full Stage 16B success criteria pass."
            ),
        }
        save_outputs(config, report, history, best_metrics or last_metrics)
        print(f"[OK] sanity_success={sanity_success}")
        print(f"[OK] full_pretraining_success={full_success}")
        print(f"[OK] hf_model={relative(checkpoint_dir / 'hf_model')}")
    cleanup_distributed()


if __name__ == "__main__":
    main()
