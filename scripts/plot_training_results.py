import argparse
import json
from pathlib import Path

import yaml


DEFAULT_LABEL_NAMES = [
    "Reentrancy",
    "Unknown address",
    "Integer overflow/underflow",
    "Timestamp dependence",
    "DoS failed call",
    "Assert violation",
    "Unchecked call return",
    "Unsafe send",
    "Multiplication after division",
    "Extra gas consumption",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot training history figures.")
    parser.add_argument(
        "--history",
        default="data/reports/train_evm_chunk_epoch_history.json",
        help="Path to epoch history JSON.",
    )
    parser.add_argument(
        "--config",
        default="configs/train_evm_chunk_server.yaml",
        help="Path to training config YAML.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for PNG figures. Defaults to data/reports/figures/<run_name>.",
    )
    return parser.parse_args()


def load_label_names(config_path):
    path = Path(config_path)
    if not path.exists():
        return DEFAULT_LABEL_NAMES
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return config.get("label_names") or DEFAULT_LABEL_NAMES


def load_history(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"History file not found: {path}")
    history = json.loads(path.read_text(encoding="utf-8"))
    if not history:
        raise ValueError(f"History is empty: {path}")
    return history


def resolve_output_dir(history_path, output_dir):
    if output_dir:
        path = Path(output_dir)
    else:
        history_path = Path(history_path)
        run_name = history_path.stem.replace("_epoch_history", "")
        path = Path("data/reports/figures") / run_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def get_series(history, key):
    return [record.get(key, 0.0) for record in history]


def save_line_plot(plt, output_path, epochs, series_list, title, ylabel):
    plt.figure(figsize=(9, 5))
    for label, values in series_list:
        plt.plot(epochs, values, marker="o", label=label)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_heatmap(plt, output_path, values, title, label_names, vmin=0.0, vmax=1.0):
    epochs = list(range(1, len(values) + 1))
    plt.figure(figsize=(max(9, len(label_names) * 0.85), max(4, len(epochs) * 0.45)))
    image = plt.imshow(values, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    plt.colorbar(image, fraction=0.025, pad=0.02)
    plt.title(title)
    plt.xlabel("Vulnerability label")
    plt.ylabel("Epoch")
    plt.xticks(range(len(label_names)), label_names, rotation=35, ha="right")
    plt.yticks(range(len(epochs)), epochs)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_per_label_lines(plt, output_path, epochs, values, title, ylabel, label_names):
    plt.figure(figsize=(11, 6))
    for label_idx, label_name in enumerate(label_names):
        label_values = [row[label_idx] for row in values]
        plt.plot(epochs, label_values, marker="o", linewidth=1.5, label=label_name)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_history(history, label_names, output_dir):
    plt = import_matplotlib()
    epochs = [record["epoch"] for record in history]

    save_line_plot(
        plt,
        output_dir / "loss_curves.png",
        epochs,
        [
            ("train_loss", get_series(history, "train_loss")),
            ("valid_loss", get_series(history, "valid_loss")),
        ],
        "Train and validation loss",
        "Loss",
    )
    save_line_plot(
        plt,
        output_dir / "overall_metrics.png",
        epochs,
        [
            ("detection_accuracy", get_series(history, "detection_accuracy")),
            ("recognition_micro_f1", get_series(history, "recognition_micro_f1")),
            ("recognition_macro_f1", get_series(history, "recognition_macro_f1")),
        ],
        "Validation overall metrics at default threshold",
        "Score",
    )
    save_line_plot(
        plt,
        output_dir / "predicted_positive_total.png",
        epochs,
        [("predicted_positive_total", get_series(history, "predicted_positive_total"))],
        "Predicted positive label count",
        "Count",
    )

    thresholds = sorted(history[0].get("threshold_scan", {}).keys(), key=float)
    if thresholds:
        save_line_plot(
            plt,
            output_dir / "threshold_micro_f1.png",
            epochs,
            [
                (
                    f"threshold={threshold}",
                    [
                        record["threshold_scan"][threshold]["micro_f1"]
                        for record in history
                    ],
                )
                for threshold in thresholds
            ],
            "Micro-F1 by threshold",
            "Micro-F1",
        )
        plt.figure(figsize=(9, 5))
        for threshold in thresholds:
            values = [
                record["threshold_scan"][threshold]["macro_f1"] for record in history
            ]
            plt.plot(epochs, values, marker="o", label=f"threshold={threshold}")
        plt.title("Macro-F1 by threshold")
        plt.xlabel("Epoch")
        plt.ylabel("Macro-F1")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "threshold_macro_f1.png", dpi=180)
        plt.close()

    metric_specs = [
        ("per_label_accuracy", "Per-label validation accuracy", "Accuracy"),
        ("per_label_precision", "Per-label validation precision", "Precision"),
        ("per_label_recall", "Per-label validation recall", "Recall"),
        ("per_label_f1", "Per-label validation F1", "F1"),
    ]
    for key, title, ylabel in metric_specs:
        values = [record.get(key, [0.0] * len(label_names)) for record in history]
        save_heatmap(
            plt,
            output_dir / f"{key}_heatmap.png",
            values,
            title,
            label_names,
        )
        save_per_label_lines(
            plt,
            output_dir / f"{key}_lines.png",
            epochs,
            values,
            title,
            ylabel,
            label_names,
        )

    predicted_counts = [
        record.get("per_label_predicted_positive_count", [0] * len(label_names))
        for record in history
    ]
    true_counts = [
        record.get("per_label_true_positive_count", [0] * len(label_names))
        for record in history
    ]
    for values, name, title in [
        (predicted_counts, "per_label_predicted_positive_count", "Predicted positives"),
        (true_counts, "per_label_true_positive_count", "True positives"),
    ]:
        plt.figure(figsize=(11, 6))
        for label_idx, label_name in enumerate(label_names):
            label_values = [row[label_idx] for row in values]
            plt.plot(epochs, label_values, marker="o", linewidth=1.2, label=label_name)
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel("Count")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
        plt.tight_layout()
        plt.savefig(output_dir / f"{name}_lines.png", dpi=180)
        plt.close()


def main():
    args = parse_args()
    history = load_history(args.history)
    label_names = load_label_names(args.config)
    output_dir = resolve_output_dir(args.history, args.output_dir)
    plot_history(history, label_names, output_dir)
    print(f"[OK] wrote figures to {output_dir}")


if __name__ == "__main__":
    main()
