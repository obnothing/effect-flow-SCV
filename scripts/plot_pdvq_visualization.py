"""Create validation-only PDVQ representation visualizations with matplotlib."""

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/visualization"
FIGURES = ROOT / "figures"
REPORTS = ROOT / "reports"
LABELS = ["Reentrancy", "Access Control", "Arithmetic", "Unchecked Return Values", "DoS", "Time manipulation"]
FOCAL = [(0, "Reentrancy"), (4, "DoS"), (1, "Access Control")]

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix, options in (("pdf", {}), ("svg", {}), ("png", {"dpi": 300})):
        fig.savefig(path.with_suffix(f".{suffix}"), bbox_inches="tight", facecolor="white", **options)
    svg = path.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")


def pca_reduce(features):
    components = min(50, features.shape[0] - 1, features.shape[1])
    return PCA(n_components=components, random_state=42).fit_transform(features)


def tsne(features):
    reduced = pca_reduce(features)
    return TSNE(n_components=2, perplexity=30, random_state=42, init="pca", learning_rate="auto", max_iter=1000).fit_transform(reduced)


def combination_groups(labels):
    strings = np.asarray(["".join(str(int(item)) for item in row) for row in labels])
    unique, counts = np.unique(strings, return_counts=True)
    top = [item for item, _ in sorted(zip(unique, counts), key=lambda pair: (-pair[1], pair[0]))[:10]]
    groups = np.asarray([item if item in top else "Other" for item in strings])
    return groups, top, {item: int((strings == item).sum()) for item in top}


def scatter_combinations(axis, coords, groups, top, title):
    palette = ["#3B5B92", "#D17A45", "#258B88", "#8B6FA8", "#B55263", "#6D7478", "#B2945B", "#4D8A66", "#8C6D5A", "#577590", "#B0B0B0"]
    ordered = top + (["Other"] if "Other" in groups else [])
    for index, group in enumerate(ordered):
        selected = groups == group
        axis.scatter(coords[selected, 0], coords[selected, 1], s=8, alpha=0.68, linewidths=0,
                     color=palette[index], label=f"{group} (n={selected.sum()})")
    axis.set_title(title, fontsize=9)
    axis.set_xlabel("t-SNE 1"); axis.set_ylabel("t-SNE 2")
    axis.set_xticks([]); axis.set_yticks([])


def intra_class_variance(values, target):
    result = {}
    for state in (0, 1):
        selected = values[target == state]
        center = selected.mean(0, keepdims=True)
        result[str(state)] = float(np.mean(np.sum((selected - center) ** 2, axis=1)))
    return result


def combination_statistics(features, groups):
    selected = groups != "Other"
    values, names = features[selected], groups[selected]
    values = (values - values.mean(0, keepdims=True)) / values.std(0, keepdims=True).clip(min=1e-12)
    encoded = {name: index for index, name in enumerate(sorted(set(names)))}
    targets = np.asarray([encoded[name] for name in names])
    compactness = []
    for name in sorted(encoded):
        group = values[names == name]
        compactness.append(float(np.mean(np.sum((group - group.mean(0, keepdims=True)) ** 2, axis=1))))
    return {
        "samples": int(len(values),),
        "groups": int(len(encoded)),
        "pca50_silhouette": float(silhouette_score(values, targets)),
        "mean_within_group_squared_distance": float(np.mean(compactness)),
        "interpretation": "PCA-50 features are standardized before descriptive statistics for the ten frequent label combinations; not a classification metric.",
    }


def bin_attention(values, bins=192):
    pieces = np.array_split(values, min(bins, len(values)))
    binned = np.asarray([piece.mean() for piece in pieces])
    return binned / max(float(binned.max()), 1e-12)


def top_text(top, limit=10):
    pairs = []
    for query, rows in top.items():
        for row in rows:
            pairs.append((row["attention"], query, row["opcode"], row["position"]))
    pairs.sort(reverse=True)
    return "  ".join(f"{query}:{opcode}@{position}" for _, query, opcode, position in pairs[:limit])


def top_lines(top, limit=10):
    pairs = []
    names = {"Reentrancy_positive": "Re+", "Reentrancy_negative": "Re-", "DoS_positive": "DoS+",
             "DoS_negative": "DoS-", "Access Control_positive": "AC+", "Access Control_negative": "AC-"}
    for query, rows in top.items():
        for row in rows:
            pairs.append((row["attention"], names.get(query, query), row["opcode"], row["position"]))
    pairs.sort(reverse=True)
    def display_opcode(opcode):
        if opcode.startswith("0x") or len(opcode) > 16:
            return "IMM_DATA"
        return opcode[:12]
    return "\n".join(f"{rank + 1:>2}. {query:<4} {display_opcode(opcode):<10} @ {position}" for rank, (_, query, opcode, position) in enumerate(pairs[:limit]))


def main():
    manifest = json.loads((DATA / "extraction_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("test_checked") is not False:
        raise ValueError("Expected validation-only extraction")
    archive = np.load(DATA / "pdvq_evidence_features.npz", allow_pickle=False)
    pos, neg, label, mean = archive["z_pos"], archive["z_neg"], archive["z_label"], archive["mean_pool"]
    labels = archive["labels"].astype(int)
    groups, top, counts = combination_groups(labels)
    FIGURES.mkdir(parents=True, exist_ok=True); REPORTS.mkdir(parents=True, exist_ok=True)

    pdvq_pca = pca_reduce(label.reshape(len(label), -1))
    mean_pca = pca_reduce(mean)
    pdvq_coords = TSNE(n_components=2, perplexity=30, random_state=42, init="pca", learning_rate="auto", max_iter=1000).fit_transform(pdvq_pca)
    np.save(DATA / "pdvq_tsne_coordinates.npy", pdvq_coords)
    fig, axis = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    scatter_combinations(axis, pdvq_coords, groups, top, "PDVQ evidence-difference representation")
    axis.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=6.5, title="Label combination", title_fontsize=7)
    save(fig, FIGURES / "pdvq_tsne_label_combination"); plt.close(fig)

    mean_coords = TSNE(n_components=2, perplexity=30, random_state=42, init="pca", learning_rate="auto", max_iter=1000).fit_transform(mean_pca)
    np.savez(DATA / "baseline_vs_pdvq_tsne_coordinates.npz", mean_pool=mean_coords, pdvq=pdvq_coords)
    comparison_statistics = {"mean_pooling": combination_statistics(mean_pca, groups),
                             "pdvq_evidence_difference": combination_statistics(pdvq_pca, groups),
                             "test_checked": False}
    (DATA / "baseline_vs_pdvq_statistics.json").write_text(json.dumps(comparison_statistics, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.45), constrained_layout=True)
    scatter_combinations(axes[0], mean_coords, groups, top, "(a) Mean pooling")
    scatter_combinations(axes[1], pdvq_coords, groups, top, "(b) PDVQ-Net")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=5.8, title="Combination", title_fontsize=6.5)
    save(fig, FIGURES / "baseline_vs_pdvq_tsne"); plt.close(fig)

    polarity_stats = {"labels": {}, "test_checked": False,
                      "interpretation": "Positive and negative queries capture different evidence distributions; these diagnostics do not identify ground-truth vulnerable or safe regions."}
    polarity_dir = FIGURES / "polarity_space"; polarity_dir.mkdir(parents=True, exist_ok=True)
    for index, name in FOCAL:
        positive_coords, negative_coords = tsne(pos[:, index]), tsne(neg[:, index])
        np.savez(DATA / f"{name.lower().replace(' ', '_')}_polarity_tsne.npz", positive=positive_coords, negative=negative_coords)
        target = labels[:, index]
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.25), constrained_layout=True)
        for axis, coords, branch in zip(axes, (positive_coords, negative_coords), ("Positive query", "Negative query")):
            for state, color, text in ((0, "#6D7478", "label=0"), (1, "#B55263", "label=1")):
                selected = target == state
                axis.scatter(coords[selected, 0], coords[selected, 1], s=8, alpha=0.68, linewidths=0, color=color, label=text)
            axis.set_title(f"{name}: {branch}", fontsize=9); axis.set_xlabel("t-SNE 1"); axis.set_ylabel("t-SNE 2")
            axis.set_xticks([]); axis.set_yticks([]); axis.legend(fontsize=6.5)
        save(fig, polarity_dir / f"{name.lower().replace(' ', '_')}_positive_negative_tsne"); plt.close(fig)
        cosine = np.sum(pos[:, index] * neg[:, index], axis=1) / (np.linalg.norm(pos[:, index], axis=1) * np.linalg.norm(neg[:, index], axis=1) + 1e-12)
        polarity_stats["labels"][name] = {
            "positive_negative_cosine_mean": float(cosine.mean()), "positive_negative_cosine_std": float(cosine.std()),
            "positive_intra_class_variance": intra_class_variance(pos[:, index], target),
            "negative_intra_class_variance": intra_class_variance(neg[:, index], target),
        }
    (DATA / "polarity_statistics.json").write_text(json.dumps(polarity_stats, indent=2), encoding="utf-8")

    attention_archive = np.load(DATA / "attention_selected_contracts.npz", allow_pickle=True)
    attention_ids, attentions = attention_archive["ids"].tolist(), attention_archive["attentions"].tolist()
    top_data = json.loads((DATA / "attention_top_opcodes.json").read_text(encoding="utf-8"))
    query_rows = [(0, 0, "Re+"), (0, 1, "Re-"), (4, 0, "DoS+"), (4, 1, "DoS-"), (1, 0, "AC+"), (1, 1, "AC-")]
    fig, axes = plt.subplots(3, 2, figsize=(7.0, 7.2), constrained_layout=True)
    for axis, contract_id, attention in zip(axes.flat, attention_ids, attentions):
        heat = np.vstack([bin_attention(attention[label, polarity]) for label, polarity, _ in query_rows])
        image = axis.imshow(heat, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0, vmax=1)
        meta = top_data["contracts"][contract_id]
        axis.set_title(f"{meta['focus_label']} example: {contract_id}", fontsize=7.5)
        axis.set_yticks(range(6), [name for _, _, name in query_rows], fontsize=6.5)
        axis.set_xlabel("Opcode position bins", fontsize=6.5)
        axis.set_xticks([0, heat.shape[1] - 1], ["0", str(meta["original_length"] - 1)], fontsize=6)
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.62, label="Relative attention within query")
    heatmap_path = FIGURES / "query_attention_heatmap"
    fig.savefig(heatmap_path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(heatmap_path.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    svg = heatmap_path.with_suffix(".svg")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    table_fig, table_axes = plt.subplots(3, 2, figsize=(7.0, 7.2), constrained_layout=True)
    for axis, contract_id in zip(table_axes.flat, attention_ids):
        meta = top_data["contracts"][contract_id]
        axis.axis("off")
        axis.set_title(f"{meta['focus_label']} example: {contract_id}", fontsize=8)
        axis.text(0.02, 0.93, top_lines(top_data["top_opcodes"][contract_id]), transform=axis.transAxes,
                  family="monospace", fontsize=6.2, va="top")
    save(table_fig, FIGURES / "query_attention_top_opcodes")
    with PdfPages(heatmap_path.with_suffix(".pdf")) as pdf:
        pdf.savefig(fig, bbox_inches="tight", facecolor="white")
        pdf.savefig(table_fig, bbox_inches="tight", facecolor="white")
    plt.close(fig); plt.close(table_fig)

    attention_stats = json.loads((DATA / "attention_statistics.json").read_text(encoding="utf-8"))
    (REPORTS / "pdvq_tsne_report.md").write_text(
        "# PDVQ vulnerability-aware representation t-SNE\n\n"
        "Input features are validation-contract `Z+ - Z-` evidence differences for all six labels, flattened from `[6,768]` to 4608 dimensions, reduced by PCA to 50 dimensions and t-SNE to two dimensions (perplexity 30, random state 42). Colors encode the ten most frequent multilabel combinations; rarer combinations are grouped as Other. The visualization suggests representation structure but does not represent a true classification boundary or prove separability.\n\n"
        f"Validation contracts: {manifest['contracts']}. Top-combination counts: {json.dumps(counts)}. Test was not read.\n", encoding="utf-8")
    (REPORTS / "baseline_vs_pdvq_tsne.md").write_text(
        "# Mean Pooling vs PDVQ t-SNE\n\n"
        "Both panels use token states from the same frozen P11 BiGRU checkpoint. Panel (a) uses masked mean pooling of `H`; panel (b) uses the six-label evidence-difference representation `Z+ - Z-`. This is an aggregation comparison, not a separately trained baseline-classifier comparison. Visual compactness or overlap suggests differences in representation organization but does not prove a decision boundary. Test was not read.\n\n"
        "PCA-50 descriptive combination statistics:\n\n"
        f"- Mean pooling: silhouette={comparison_statistics['mean_pooling']['pca50_silhouette']:.6f}; mean within-group squared distance={comparison_statistics['mean_pooling']['mean_within_group_squared_distance']:.6f}.\n"
        f"- PDVQ evidence difference: silhouette={comparison_statistics['pdvq_evidence_difference']['pca50_silhouette']:.6f}; mean within-group squared distance={comparison_statistics['pdvq_evidence_difference']['mean_within_group_squared_distance']:.6f}.\n",
        encoding="utf-8")
    polarity_lines = ["# Positive and Negative Evidence Spaces", "", polarity_stats["interpretation"], ""]
    for name, values in polarity_stats["labels"].items():
        polarity_lines.append(f"- {name}: mean positive-negative cosine = {values['positive_negative_cosine_mean']:.6f}; std = {values['positive_negative_cosine_std']:.6f}.")
    (REPORTS / "polarity_space_analysis.md").write_text("\n".join(polarity_lines) + "\n", encoding="utf-8")
    (REPORTS / "query_attention_analysis.md").write_text(
        "# Query Attention Opcode Localization\n\n"
        "Six validation contracts were selected with positive focal labels, preferring fewer co-occurring labels and moderate opcode length. Heatmaps show log-free, row-max-normalized attention for readability; raw attention and top attended opcodes are saved in the visualization results. These are model attention distributions and decision-sensitivity diagnostics, not ground-truth vulnerability localization.\n\n"
        f"Mean positive-negative attention cosine by label: {json.dumps(attention_stats['positive_negative_attention_cosine'])}. Test was not read.\n", encoding="utf-8")
    (ROOT / "summary_visualization.md").write_text(
        "# PDVQ-Net Visualization Summary\n\n"
        "- `pdvq_tsne_label_combination`: displays six-label `Z+ - Z-` evidence differences by frequent label combination; suitable for the main paper if visual structure is legible.\n"
        "- `baseline_vs_pdvq_tsne`: compares masked mean pooling and PDVQ evidence-difference aggregation from the same P11 token encoder; suitable for the main paper as a representation comparison.\n"
        "- `polarity_space/*`: compares positive and negative query evidence spaces for Reentrancy, DoS and Access Control; suitable for supplementary material or a focused main-paper panel.\n"
        "- `query_attention_heatmap`: shows polarity-specific query attention and top attended opcode positions for six validation examples; suitable for supplementary material.\n"
        "All artifacts are validation-only, derive from an existing checkpoint, and should be described as model representation or attention diagnostics rather than ground-truth evidence localization.\n", encoding="utf-8")


if __name__ == "__main__":
    main()
