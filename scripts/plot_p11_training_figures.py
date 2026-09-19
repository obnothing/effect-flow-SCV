"""Generate publication-ready P11 learning curves and per-label F1 figures."""

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results/light_label/polarity_queries_followup_p11/P11/metrics.json"
OUTPUT = ROOT / "artifacts/figures/p11_training"
LABELS = [
    "Reentrancy",
    "Access Control",
    "Arithmetic",
    "Unchecked Return\nValues",
    "DoS",
    "Time manipulation",
]


mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "legend.fontsize": 7,
    "lines.linewidth": 1.8,
})


def save_figure(fig, stem):
    for suffix, options in (
        ("svg", {}),
        ("pdf", {}),
        ("png", {"dpi": 600}),
        ("tiff", {"dpi": 600}),
    ):
        fig.savefig(OUTPUT / f"{stem}.{suffix}", bbox_inches="tight", facecolor="white", **options)
    svg_path = OUTPUT / f"{stem}.svg"
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n",
                        encoding="utf-8")


def write_source_data(history, tuned):
    with (OUTPUT / "p11_learning_curves_source.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_total_loss", "valid_classification_loss", "tuned_macro_f1"])
        writer.writeheader()
        for row in history:
            writer.writerow({
                "epoch": row["epoch"],
                "train_total_loss": row["train_loss"],
                "valid_classification_loss": row["valid_loss"],
                "tuned_macro_f1": row["metrics"]["tuned"]["macro_f1"],
            })
    with (OUTPUT / "p11_per_label_f1_source.csv").open("w", newline="", encoding="utf-8") as handle:
        clean_labels = [label.replace("\n", " ") for label in LABELS]
        writer = csv.DictWriter(handle, fieldnames=["epoch", *clean_labels])
        writer.writeheader()
        for row in history:
            values = row["metrics"]["tuned"]["per_label_f1"]
            writer.writerow({"epoch": row["epoch"], **dict(zip(clean_labels, values))})


def learning_curve(history, best_epoch):
    epoch = np.asarray([row["epoch"] for row in history])
    train_loss = np.asarray([row["train_loss"] for row in history])
    valid_loss = np.asarray([row["valid_loss"] for row in history])
    macro = np.asarray([row["metrics"]["tuned"]["macro_f1"] for row in history])

    fig, axis = plt.subplots(figsize=(160 / 25.4, 88 / 25.4), constrained_layout=True)
    macro_axis = axis.twinx()
    axis.spines["right"].set_visible(False)
    macro_axis.spines["top"].set_visible(False)

    axis.plot(epoch, train_loss, color="#335C81", label="Training objective")
    axis.plot(epoch, valid_loss, color="#D17A45", label="Validation classification loss")
    macro_axis.plot(epoch, macro, color="#258B88", marker="o", markersize=2.6,
                    markevery=2, label="Tuned Macro-F1")
    axis.axvline(best_epoch, color="#686868", linewidth=0.9, linestyle=(0, (3, 3)), zorder=0)
    macro_axis.scatter([best_epoch], [macro[best_epoch - 1]], color="#258B88", s=24, zorder=5,
                       edgecolor="white", linewidth=0.6)
    macro_axis.annotate(f"{macro[best_epoch - 1]:.3f}",
                        (best_epoch, macro[best_epoch - 1]), xytext=(-5, 8),
                        textcoords="offset points", ha="right", color="#176B69", fontsize=7)

    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    macro_axis.set_ylabel("Tuned Macro-F1", color="#176B69")
    macro_axis.tick_params(axis="y", colors="#176B69")
    axis.set_xlim(1, epoch.max())
    axis.set_ylim(0, max(train_loss.max(), valid_loss.max()) * 1.08)
    macro_axis.set_ylim(0.48, 0.86)
    axis.set_xticks([1, 5, 10, 15, 20, 25, 30])
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.75)

    handles1, labels1 = axis.get_legend_handles_labels()
    handles2, labels2 = macro_axis.get_legend_handles_labels()
    axis.legend(handles1 + handles2, labels1 + labels2, loc="center right")
    axis.text(0.01, 0.03,
              "Training objective includes the 0.1-weighted polarity auxiliary loss;\n"
              "validation loss is classification loss only.",
              transform=axis.transAxes, fontsize=6.2, color="#555555", va="bottom")
    save_figure(fig, "p11_learning_curves")
    plt.close(fig)


def per_label_curves(history):
    epoch = np.asarray([row["epoch"] for row in history])
    values = np.asarray([row["metrics"]["tuned"]["per_label_f1"] for row in history])
    styles = [
        ("#335C81", "-", "o"),
        ("#D17A45", "-", "s"),
        ("#258B88", "-", "^"),
        ("#7A6F9B", "--", "D"),
        ("#B55263", "--", "v"),
        ("#6D7478", "-.", "P"),
    ]
    fig, axis = plt.subplots(figsize=(168 / 25.4, 94 / 25.4), constrained_layout=True)
    for index, (label, (color, linestyle, marker)) in enumerate(zip(LABELS, styles)):
        axis.plot(epoch, values[:, index], color=color, linestyle=linestyle, marker=marker,
                  markersize=2.8, markevery=3, linewidth=1.65,
                  label=f"{label.replace(chr(10), ' ')} ({values[-1, index]:.3f})")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Tuned F1")
    axis.set_xlim(1, 30)
    axis.set_ylim(0.2, 0.91)
    axis.set_xticks([1, 5, 10, 15, 20, 25, 30])
    axis.set_yticks(np.arange(0.2, 1.0, 0.1))
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.75)
    axis.legend(loc="lower right", ncol=2, columnspacing=1.2, handlelength=2.5)
    save_figure(fig, "p11_per_label_tuned_f1")
    plt.close(fig)


def main():
    if not METRICS.exists():
        raise FileNotFoundError(METRICS)
    payload = json.loads(METRICS.read_text(encoding="utf-8"))
    if payload.get("test_checked") is not False:
        raise ValueError("Expected validation-only P11 artifact")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    history = payload["history"]
    tuned = dict(payload["metrics"]["tuned"], thresholds=payload["metrics"]["thresholds"])
    write_source_data(history, tuned)
    learning_curve(history, int(payload["best_epoch"]))
    per_label_curves(history)
    metadata = {
        "dataset": "DIVE_main6_opcode_process01",
        "split": "validation",
        "seed": 42,
        "best_epoch": int(payload["best_epoch"]),
        "tuned_macro_f1": float(tuned["macro_f1"]),
        "test_checked": False,
        "variability": "Single seed; no error bars or confidence intervals are implied.",
        "loss_note": "Training loss includes the polarity auxiliary term; validation loss is classification-only.",
    }
    (OUTPUT / "figure_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
