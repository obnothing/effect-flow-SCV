import argparse
import ast
import csv
import json
import re
from pathlib import Path


DEFAULT_LABELS = [
    "Reentrancy",
    "Access Control",
    "Arithmetic",
    "Unchecked Return Values",
    "DoS",
    "Bad Randomness",
    "Front Running",
    "Time manipulation",
]


def load_history(path):
    path = Path(path)
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return parse_history_txt(path)


def parse_history_txt(path):
    rows = []
    current = None
    epoch_re = re.compile(
        r"epoch (?P<epoch>\d+)/(?P<epochs>\d+) "
        r"train_loss=(?P<train_loss>[-+0-9.eE]+) "
        r"valid_loss=(?P<valid_loss>[-+0-9.eE]+) "
        r"micro_f1=(?P<micro>[-+0-9.eE]+) "
        r"macro_f1=(?P<macro>[-+0-9.eE]+) "
        r"predicted_positive_total=(?P<pred>\d+)"
    )
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        match = epoch_re.match(line)
        if match:
            if current is not None:
                rows.append(current)
            current = {
                "epoch": int(match.group("epoch")),
                "epochs": int(match.group("epochs")),
                "train_loss": float(match.group("train_loss")),
                "valid_loss": float(match.group("valid_loss")),
                "recognition_micro_f1": float(match.group("micro")),
                "recognition_macro_f1": float(match.group("macro")),
                "predicted_positive_total": int(match.group("pred")),
            }
            continue
        if current is not None and line.startswith("per_label_f1: "):
            current["per_label_f1"] = ast.literal_eval(line.split(": ", 1)[1])
    if current is not None:
        rows.append(current)
    return rows


def write_csv(history, labels, output_csv):
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "epoch",
        "train_loss",
        "valid_loss",
        "micro_f1",
        "macro_f1",
        "predicted_positive_total",
    ] + [f"{label}_f1" for label in labels]
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in history:
            out = {
                "epoch": row["epoch"],
                "train_loss": row.get("train_loss"),
                "valid_loss": row.get("valid_loss"),
                "micro_f1": row.get("recognition_micro_f1"),
                "macro_f1": row.get("recognition_macro_f1"),
                "predicted_positive_total": row.get("predicted_positive_total"),
            }
            per_label = row.get("per_label_f1", [])
            for idx, label in enumerate(labels):
                out[f"{label}_f1"] = per_label[idx] if idx < len(per_label) else ""
            writer.writerow(out)


def best_line(history, key):
    best = max(history, key=lambda row: float(row.get(key, 0.0)))
    return best["epoch"], float(best.get(key, 0.0))


def write_report(history, labels, output_report):
    output_report = Path(output_report)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    lines = ["DIVE epoch F1 curve summary", ""]
    epoch, value = best_line(history, "recognition_micro_f1")
    lines.append(f"best_micro_f1: epoch={epoch} value={value:.6f}")
    epoch, value = best_line(history, "recognition_macro_f1")
    lines.append(f"best_macro_f1: epoch={epoch} value={value:.6f}")
    for idx, label in enumerate(labels):
        best = max(
            history,
            key=lambda row: float(row.get("per_label_f1", [0.0] * len(labels))[idx]),
        )
        lines.append(
            f"{label}: best_epoch={best['epoch']} "
            f"best_f1={float(best['per_label_f1'][idx]):.6f}"
        )
    output_report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_curves(history, labels, output_dir, prefix):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] matplotlib unavailable, skip png plots: {exc}")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, [row["recognition_micro_f1"] for row in history], label="micro-F1")
    plt.plot(epochs, [row["recognition_macro_f1"] for row in history], label="macro-F1")
    plt.xlabel("Epoch")
    plt.ylabel("Validation F1")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_micro_macro_f1.png", dpi=180)
    plt.close()

    plt.figure(figsize=(11, 6))
    for idx, label in enumerate(labels):
        values = [row.get("per_label_f1", [0.0] * len(labels))[idx] for row in history]
        plt.plot(epochs, values, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Validation per-label F1")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.25)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_per_label_f1.png", dpi=180)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot micro/macro and per-label F1 curves from MIL epoch history."
    )
    parser.add_argument("--history", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prefix", default="epoch")
    parser.add_argument("--labels", nargs="*", default=DEFAULT_LABELS)
    return parser.parse_args()


def main():
    args = parse_args()
    history = load_history(args.history)
    if not history:
        raise SystemExit(f"No epoch rows parsed from {args.history}")
    output_dir = Path(args.output_dir)
    prefix = args.prefix
    write_csv(history, args.labels, output_dir / f"{prefix}_f1_curves.csv")
    write_report(history, args.labels, output_dir / f"{prefix}_f1_curve_summary.txt")
    plot_curves(history, args.labels, output_dir, prefix)
    print(f"[OK] wrote F1 curve artifacts to {output_dir}")


if __name__ == "__main__":
    main()
