"""Plot B0/B1/B2 loss and validation Macro-F1 curves for process01."""

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    "b0_mean": "B0 Mean",
    "b1_shared_attention": "B1 Shared Attention",
    "b2_label_attention": "B2 Label Attention",
}
COLORS = {
    "b0_mean": "#4C78A8",
    "b1_shared_attention": "#F58518",
    "b2_label_attention": "#54A24B",
}


def load_history(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for item in payload["history"]:
        metrics = item["metrics"]
        rows.append({
            "epoch": int(item["epoch"]),
            "train_loss": float(item["train_loss"]),
            "valid_loss": float(item["valid_loss"]),
            "fixed_macro_f1": float(metrics["fixed"]["macro_f1"]),
            "tuned_macro_f1": float(metrics["tuned"]["macro_f1"]),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", choices=["full", "smoke"], default="full")
    parser.add_argument("--output", default="results/light_label/light_label_training_curves")
    args = parser.parse_args()

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
    })

    histories = {}
    for variant in VARIANTS:
        path = ROOT / "results" / "light_label" / variant / args.run / "metrics.json"
        if path.exists():
            histories[variant] = load_history(path)
    if not histories:
        raise FileNotFoundError(f"No {args.run} metrics found under results/light_label")

    fig, (loss_ax, f1_ax) = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)
    for variant, rows in histories.items():
        epochs = [row["epoch"] for row in rows]
        color = COLORS[variant]
        label = VARIANTS[variant]
        loss_ax.plot(epochs, [row["train_loss"] for row in rows], color=color, lw=2.0, label=f"{label} train")
        loss_ax.plot(epochs, [row["valid_loss"] for row in rows], color=color, lw=1.8, ls="--", label=f"{label} valid")
        f1_ax.plot(epochs, [row["tuned_macro_f1"] for row in rows], color=color, lw=2.2, label=f"{label} tuned")
        f1_ax.plot(epochs, [row["fixed_macro_f1"] for row in rows], color=color, lw=1.5, ls="--", alpha=0.85, label=f"{label} fixed")

    loss_ax.set_title("Training and validation loss")
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("BCE loss")
    loss_ax.grid(axis="y", color="#D9D9D9", lw=0.6, alpha=0.7)
    loss_ax.legend(frameon=False, fontsize=8, ncol=2)
    f1_ax.set_title("Validation Macro-F1")
    f1_ax.set_xlabel("Epoch")
    f1_ax.set_ylabel("Macro-F1")
    f1_ax.set_ylim(0.0, 1.0)
    f1_ax.grid(axis="y", color="#D9D9D9", lw=0.6, alpha=0.7)
    f1_ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle("LabelGuidedOpcodeNet on DIVE Main-6 process01", fontsize=12, fontweight="bold")

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output) + ".png", dpi=300)
    fig.savefig(str(output) + ".svg")
    fig.savefig(str(output) + ".pdf")
    plt.close(fig)
    print(json.dumps({"output_base": str(output), "variants": list(histories), "run": args.run}, indent=2))


if __name__ == "__main__":
    main()
