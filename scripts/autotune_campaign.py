"""Run a resumable Optuna campaign for one A0/A1/A3 aggregation family."""

import argparse
import importlib.util
import json
import math
import random
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autotune_state import append_ledger, read_state, record_failure, transition, update_best  # noqa: E402
from autotune_model import AutoTuneSequenceNet  # noqa: E402
from evm_tokenizer import EVMOpcodeTokenizer  # noqa: E402
from light_label_data import LengthBucketBatchSampler, LightLabelDataset, collate_light_label  # noqa: E402
from metrics import compute_multilabel_metrics_from_probs, derived_detection_metrics_from_multilabel_probs, select_per_label_thresholds  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


LABELS = ["Reentrancy", "Access Control", "Arithmetic", "Unchecked Return Values", "DoS", "Time manipulation"]


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_base():
    config = yaml.safe_load((ROOT / "configs/light_label/b2_label_attention.yaml").read_text(encoding="utf-8"))
    runtime = resolve("results/light_label/resolved_runtime.json")
    if runtime.exists():
        config.update(json.loads(runtime.read_text(encoding="utf-8")))
    return config


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def sample_config(trial, aggregation, base):
    encoder = trial.suggest_categorical("encoder", ["GRU", "BiGRU", "LSTM", "BiLSTM"])
    self_attention = trial.suggest_categorical("self_attention", [False, True])
    value = {
        "aggregation": aggregation,
        "embedding_dim": trial.suggest_categorical("embedding_dim", [64, 128, 256, 512]),
        "encoder": encoder,
        "hidden": trial.suggest_categorical("hidden", [96, 128, 192, 256, 384, 512]),
        "layers": trial.suggest_categorical("layers", [1, 2, 3]),
        "embedding_mlp": trial.suggest_categorical("embedding_mlp", [False, True]),
        "self_attention": self_attention,
        "residual": trial.suggest_categorical("residual", [False, True]),
        "dropout": trial.suggest_float("dropout", 0.05, 0.5),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "gradient_clip": trial.suggest_categorical("gradient_clip", [0.5, 1.0, 2.0]),
        "optimizer": trial.suggest_categorical("optimizer", ["Adam", "AdamW"]),
        "scheduler": trial.suggest_categorical("scheduler", ["none", "cosine", "plateau"]),
        "max_len": trial.suggest_categorical("max_len", [4096, 8192, 12288, 16384]),
        "batch_size": int(base.get("batch_size", 4)),
        "gradient_accumulation_steps": int(base.get("gradient_accumulation_steps", 4)),
        "num_labels": 6,
        "bidirectional": encoder.startswith("Bi"),
        "amp": bool(base.get("amp", True)),
        "weighted_bce": bool(base.get("weighted_bce", True)),
        "pos_weight_mode": base.get("pos_weight_mode", "sqrt_ratio"),
        "max_pos_weight": float(base.get("max_pos_weight", 5.0)),
    }
    if self_attention:
        value["heads"] = trial.suggest_categorical("heads", [1, 2, 4, 8])
        output_dim = int(value["hidden"]) * (2 if value["bidirectional"] else 1)
        if output_dim % int(value["heads"]) != 0:
            raise ValueError(f"invalid attention heads={value['heads']} for output_dim={output_dim}")
    return value


def make_data(config, split):
    path = resolve(config["cache_dir"]) / f"{split}_max{config['max_len']}.pt"
    if not path.exists():
        raise FileNotFoundError(f"cache missing for max_len={config['max_len']}: {path}")
    return LightLabelDataset(path, runtime_max_len=config["max_len"])


def loader(data, config, tokenizer, shuffle):
    if shuffle:
        sampler = LengthBucketBatchSampler(data, int(config["batch_size"]), 42)
        return DataLoader(data, batch_sampler=sampler, num_workers=0, collate_fn=partial(collate_light_label, pad_id=tokenizer.pad_token_id))
    return DataLoader(data, batch_size=int(config["batch_size"]), shuffle=False, num_workers=0, collate_fn=partial(collate_light_label, pad_id=tokenizer.pad_token_id))


def weights(data, config):
    labels = data.labels[data.indices]
    positive = labels.sum(0); negative = len(labels) - positive
    value = torch.sqrt(negative / positive.clamp_min(1)) if config["pos_weight_mode"] == "sqrt_ratio" else negative / positive.clamp_min(1)
    return value.clamp(1.0, float(config["max_pos_weight"]))


def metrics(config, labels, logits):
    probs = torch.sigmoid(logits).numpy(); targets = labels.numpy().astype(int)
    selected = select_per_label_thresholds(targets, probs, [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], LABELS, global_threshold=0.2)
    tuned = compute_multilabel_metrics_from_probs(targets, probs, selected["thresholds"])
    fixed = compute_multilabel_metrics_from_probs(targets, probs, 0.5)
    detection = derived_detection_metrics_from_multilabel_probs(targets, probs, selected["thresholds"])
    return {"tuned_macro_f1": float(tuned["recognition_macro_f1"]), "tuned_micro_f1": float(tuned["recognition_micro_f1"]),
            "fixed_macro_f1": float(fixed["recognition_macro_f1"]), "detection_f1": float(detection["detection_f1"]),
            "per_label_f1": [float(x) for x in tuned["per_label_f1"]], "thresholds": [float(x) for x in selected["thresholds"]]}


def train_trial(config, trial, study_name, trial_number, device, epochs):
    tokenizer = EVMOpcodeTokenizer.from_vocab_file(resolve(config["vocab_path"]))
    train_data = make_data(config, "train"); valid_data = make_data(config, "valid")
    train_loader = loader(train_data, config, tokenizer, True); valid_loader = loader(valid_data, config, tokenizer, False)
    model = AutoTuneSequenceNet(config, len(tokenizer), tokenizer.pad_token_id).to(device)
    optimizer_class = torch.optim.Adam if config["optimizer"] == "Adam" else torch.optim.AdamW
    optimizer = optimizer_class(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    scheduler = None
    if config["scheduler"] == "cosine": scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif config["scheduler"] == "plateau": scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=1)
    amp = device.type == "cuda" and bool(config["amp"])
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"): scaler = torch.amp.GradScaler("cuda", enabled=amp)
    else: scaler = torch.cuda.amp.GradScaler(enabled=amp)
    weight = weights(train_data, config).to(device) if config["weighted_bce"] else None
    best = -1.0; best_metrics = None; history = []
    for epoch in range(1, epochs + 1):
        model.train(); optimizer.zero_grad(set_to_none=True); losses=[]; start=time.perf_counter()
        for step, batch in enumerate(train_loader, 1):
            inputs=batch["input_ids"].to(device); mask=batch["mask"].to(device); target=batch["labels"].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                output=model(inputs,batch["lengths"],mask); loss=F.binary_cross_entropy_with_logits(output["logits"],target,pos_weight=weight)
            scaled=loss/int(config["gradient_accumulation_steps"])
            if scaler.is_enabled(): scaler.scale(scaled).backward()
            else: scaled.backward()
            losses.append(float(loss.detach()))
            if step % int(config["gradient_accumulation_steps"]) == 0 or step == len(train_loader):
                if scaler.is_enabled(): scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip"]))
                if scaler.is_enabled(): scaler.step(optimizer); scaler.update()
                else: optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        model.eval(); valid_logits=[]; valid_labels=[]
        with torch.no_grad():
            for batch in valid_loader:
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp): output=model(batch["input_ids"].to(device),batch["lengths"],batch["mask"].to(device))
                valid_logits.append(output["logits"].float().cpu()); valid_labels.append(batch["labels"])
        score=metrics(config,torch.cat(valid_labels),torch.cat(valid_logits)); history.append({"epoch":epoch,"train_loss":float(np.mean(losses)),"metrics":score,"epoch_seconds":time.perf_counter()-start})
        print(f"[autotune] trial={trial_number} epoch={epoch} max_len={config['max_len']} macro={score['tuned_macro_f1']:.6f}",flush=True)
        trial.report(score["tuned_macro_f1"], epoch)
        if trial.should_prune(): raise __import__("optuna").TrialPruned()
        if score["tuned_macro_f1"] > best: best=score["tuned_macro_f1"]; best_metrics=score
        if scheduler is not None:
            if config["scheduler"] == "plateau":
                scheduler.step(score["tuned_macro_f1"])
            else:
                scheduler.step()
    checkpoint = ROOT / f"checkpoints/autotune/{study_name}/trial_{trial_number}.pt"; checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": config, "metrics": best_metrics, "history": history, "test_checked": False}, checkpoint)
    return best_metrics, checkpoint, history


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--architecture",choices=["a0_mean","a1_shared","a3_vsfs"],required=True); parser.add_argument("--n-trials",type=int,default=3); parser.add_argument("--epochs",type=int,default=3); args=parser.parse_args()
    if importlib.util.find_spec("optuna") is None: raise RuntimeError("Optuna is not installed; install requirements-autotune.txt first")
    import optuna
    base=load_base(); study_name=f"scvd_main6_autotune_{args.architecture}"; storage="sqlite:///autotune/optuna.db"
    study=optuna.create_study(study_name=study_name,storage=storage,direction="maximize",sampler=optuna.samplers.TPESampler(seed=42),pruner=optuna.pruners.HyperbandPruner(),load_if_exists=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state=read_state()
    if state.get("state") == "RUNNING": raise RuntimeError(f"existing AutoTune trial is marked RUNNING: {state.get('current_trial_id')}")
    def objective(trial):
        if read_state().get("state") == "COMPARE": transition("PROPOSE", current_trial_id=f"{study_name}:{trial.number}")
        elif read_state().get("state") == "INIT": transition("PROPOSE", current_trial_id=f"{study_name}:{trial.number}")
        else: raise RuntimeError(f"unexpected AutoTune state: {read_state().get('state')}")
        trial_id=f"{study_name}:{trial.number}"; config=sample_config(trial,args.architecture,base); config.update({"data_dir":base["data_dir"],"vocab_path":base["vocab_path"],"cache_dir":base["cache_dir"]})
        transition("SMOKE", current_trial_id=trial_id, current_architecture=args.architecture); transition("SUBMIT", current_trial_id=trial_id); transition("RUNNING", current_trial_id=trial_id)
        try:
            if not (resolve(config["cache_dir"]) / f"train_max{config['max_len']}.pt").exists():
                record_failure({"trial_id":trial_id,"failure_type":"CACHE_MISSING","max_len":config["max_len"]}); raise optuna.TrialPruned()
            result, checkpoint, history=train_trial(config,trial,study_name,trial.number,device,args.epochs)
            transition("EVALUATE", current_trial_id=trial_id); transition("RECORD", current_trial_id=trial_id); update_best({"trial_id":trial_id,"variant":args.architecture,"metrics":result,"config":config,"checkpoint":str(checkpoint)})
            transition("COMPARE", current_trial_id=trial_id, last_completed_trial_id=trial_id); return result["tuned_macro_f1"]
        except optuna.TrialPruned:
            append_ledger({"event":"trial_pruned","trial_id":trial_id}); transition("COMPARE", current_trial_id=trial_id, last_completed_trial_id=trial_id); raise
        except torch.cuda.OutOfMemoryError as error:
            record_failure({"trial_id":trial_id,"failure_type":"OOM_RESOURCE_LIMIT","error":str(error)}); torch.cuda.empty_cache(); transition("COMPARE", current_trial_id=trial_id); raise optuna.TrialPruned()
        except Exception as error:
            record_failure({"trial_id":trial_id,"failure_type":"TRIAL_ERROR","error":repr(error)}); transition("COMPARE", current_trial_id=trial_id); raise
    study.optimize(objective,n_trials=args.n_trials)
    complete = [trial.value for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None]
    print(json.dumps({"study":study_name,"trials":len(study.trials),"best":max(complete) if complete else None,"test_checked":False},indent=2))


if __name__ == "__main__": main()
