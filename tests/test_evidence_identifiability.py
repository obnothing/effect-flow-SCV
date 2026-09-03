import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from diagnose_evidence_identifiability import build_deletion_batch, rank_local  # noqa: E402


def test_rank_local_respects_mask_and_top_k():
    values = torch.tensor([0.1, 0.9, 0.8, 0.7])
    mask = torch.tensor([True, False, True, True])
    assert rank_local(values, mask, 2, True) == [2, 3]


def test_rank_local_can_select_low_influence_chunks():
    values = torch.tensor([0.1, 0.9, 0.8])
    mask = torch.tensor([True, True, False])
    assert rank_local(values, mask, 5, False) == [0, 1]


def test_influence_definition_does_not_select_padding():
    values = torch.tensor([0.5, 0.2, float("nan")])
    mask = torch.isfinite(values)
    assert rank_local(values, mask, 3, True) == [0, 1]


def test_deletion_batch_preserves_other_chunks_and_rejects_all_padding():
    data = {
        "features": torch.randn(1, 3, 8, 4),
        "mask": torch.tensor([[True, True, False]]),
    }
    features, masks = build_deletion_batch(data, 0, [0])
    assert features.shape == (1, 3, 8, 4)
    assert masks.tolist() == [[False, True, False]]
    try:
        build_deletion_batch(data, 0, [0, 1])
    except ValueError as exc:
        assert "only valid chunk" in str(exc)
    else:
        raise AssertionError("expected all-padding deletion to fail")
