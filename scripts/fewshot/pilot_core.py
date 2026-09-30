"""Shared math and scoring helpers for the few-shot query pilot."""

import math

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


def strict_metric(probability, labels):
    probability = np.asarray(probability, dtype=float).reshape(-1)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    pred = (probability >= 0.5).astype(int)
    return {"pr_auc": float(average_precision_score(labels, probability)),
            "roc_auc": float(roc_auc_score(labels, probability)) if np.unique(labels).size == 2 else float("nan"),
            "f1_05": float(f1_score(labels, pred, zero_division=0)),
            "precision_05": float(precision_score(labels, pred, zero_division=0)),
            "recall_05": float(recall_score(labels, pred, zero_division=0))}


def match_old_distribution(values, reference):
    """Match scalar mean/std/norm scale of a reference query collection."""
    values = torch.as_tensor(values, dtype=torch.float32)
    reference = torch.as_tensor(reference, dtype=torch.float32)
    centered = values - values.mean()
    centered = centered / centered.std(unbiased=False).clamp_min(1e-8)
    result = centered * reference.std(unbiased=False) + reference.mean()
    target_norm = reference.norm(dim=-1).mean()
    centered_result = result - result.mean()
    centered_result = centered_result * (max(float(target_norm**2 - result.numel() * reference.mean()**2), 1e-8) ** 0.5) / centered_result.norm().clamp_min(1e-8)
    result = centered_result + reference.mean()
    return result


def initialize_random_query(old_queries, seed):
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    pair = []
    stats = []
    for polarity in range(2):
        reference = old_queries[:, polarity, :].detach().float().cpu()
        random_vector = torch.randn(reference.shape[-1], generator=generator)
        vector = match_old_distribution(random_vector, reference)
        pair.append(vector)
        stats.append({"reference_mean": float(reference.mean()), "sample_mean": float(vector.mean()),
                      "reference_std": float(reference.std(unbiased=False)), "sample_std": float(vector.std(unbiased=False)),
                      "reference_mean_norm": float(reference.norm(dim=-1).mean()), "sample_norm": float(vector.norm())})
    return torch.stack(pair), stats


def query_span_basis(queries, tolerance=1e-6):
    """Orthonormal column basis spanning query rows [M,D]."""
    matrix = torch.as_tensor(queries, dtype=torch.float64).detach().cpu().T
    u, singular, _ = torch.linalg.svd(matrix, full_matrices=False)
    if not singular.numel() or float(singular.max()) == 0:
        return torch.empty((matrix.shape[0], 0), dtype=torch.float32)
    rank = int((singular > singular.max() * tolerance).sum())
    return u[:, :rank].float()


def random_orthogonal_basis(queries, rank, seed):
    q_basis = query_span_basis(queries)
    dim = queries.shape[-1]
    effective = min(int(rank), dim - q_basis.shape[1])
    if effective <= 0:
        return torch.empty((dim, 0)), q_basis
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    matrix = torch.randn((dim, effective), generator=gen)
    if q_basis.numel():
        matrix -= q_basis @ (q_basis.T @ matrix)
    basis, _ = torch.linalg.qr(matrix, mode="reduced")
    return basis[:, :effective].contiguous(), q_basis


def gradient_residual_basis(queries, gradients, rank=4):
    """Return an orthogonal low-rank basis and novelty diagnostics."""
    q_basis = query_span_basis(queries)
    matrix = torch.as_tensor(gradients, dtype=torch.float32).detach().cpu()
    if matrix.ndim != 2 or not matrix.shape[0]:
        return torch.empty((queries.shape[-1], 0)), {"ratio": 0.0, "c4": 0.0, "rank": 0, "orthogonality": 0.0}
    original_sq = float(matrix.square().sum())
    if q_basis.numel():
        projected = matrix - (matrix @ q_basis) @ q_basis.T
    else:
        projected = matrix
    residual_sq = float(projected.square().sum())
    _, singular, vh = torch.linalg.svd(projected.double(), full_matrices=False)
    if singular.numel() and float(singular.max()) > 0:
        effective = int((singular > singular.max() * 1e-6).sum())
        use_rank = min(int(rank), effective, queries.shape[-1] - q_basis.shape[1])
        basis = vh[:use_rank].T.float().contiguous() if use_rank else torch.empty((queries.shape[-1], 0))
        energy = singular.square()
        c4 = float(energy[:min(4, len(energy))].sum() / energy.sum().clamp_min(1e-20))
    else:
        effective = use_rank = 0
        basis = torch.empty((queries.shape[-1], 0))
        c4 = 0.0
    orthogonality = float((q_basis.T @ basis).norm()) if q_basis.numel() and basis.numel() else 0.0
    return basis, {"ratio": residual_sq / max(original_sq, 1e-20), "c4": c4,
                   "rank": int(use_rank), "effective_rank": int(effective), "orthogonality": orthogonality}


def query_mixture(old_queries, alpha_plus, alpha_minus):
    plus = torch.softmax(alpha_plus, dim=0) @ old_queries[:, 0, :]
    minus = torch.softmax(alpha_minus, dim=0) @ old_queries[:, 1, :]
    return torch.stack([plus, minus])


def novel_forward(base_model, hidden, mask, queries, scorer, bias):
    batch = hidden.shape[0]
    q = queries.unsqueeze(0).expand(batch, -1, -1)
    evidence, _ = base_model.cross_attention(q, hidden, hidden, key_padding_mask=~mask, need_weights=False)
    evidence = F.dropout(evidence, p=base_model.representation_dropout.p, training=base_model.training)
    energies = torch.einsum("bpd,d->bp", evidence, scorer) + bias.unsqueeze(0)
    logits = energies[:, 0] - energies[:, 1]
    return {"logits": logits[:, None], "energies": energies[:, None, :], "representations": evidence[:, None, :, :]}


def novel_loss(output, targets, auxiliary_weight=0.08):
    y = targets.float().reshape(-1)
    logits = output["logits"].reshape(-1).float()
    energies = output["energies"][:, 0, :].float()
    cls = F.binary_cross_entropy_with_logits(logits, y)
    polarity = 0.5 * (F.binary_cross_entropy_with_logits(energies[:, 0], y) +
                      F.binary_cross_entropy_with_logits(energies[:, 1], 1-y))
    return cls + float(auxiliary_weight) * polarity, cls, polarity


def encode_batch(model, batch, device, amp=True):
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=bool(amp)):
        hidden = model.encode_tokens(input_ids, batch["lengths"], mask)
    return hidden, mask


def support_loss_curve(rows):
    return [{"epoch": i+1, "support_loss": float(value)} for i, value in enumerate(rows)]
