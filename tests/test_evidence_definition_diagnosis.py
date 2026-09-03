import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from diagnose_evidence_definition import (  # noqa: E402
    contiguous_spans,
    jaccard,
    rank_indices,
    select_ranked_or_fallback,
)


def test_jaccard_and_masked_ranking():
    assert jaccard([1, 2, 3], [2, 3, 4]) == 0.5
    values = torch.tensor([0.2, 0.9, 0.8, 0.7])
    mask = torch.tensor([True, False, True, True])
    assert rank_indices(values, mask, 2) == [2, 3]


def test_contiguous_spans_uses_only_valid_consecutive_chunks():
    values = torch.tensor([0.9, 0.8, 0.7, 0.1])
    mask = torch.tensor([True, True, True, False])
    assert contiguous_spans(values, mask, 1, 3) == [[0, 1, 2]]


def test_contiguous_spans_falls_back_when_no_window_exists():
    values = torch.tensor([0.9, 0.1, 0.8])
    mask = torch.tensor([True, False, True])
    assert contiguous_spans(values, mask, 2, 3) == [[0], [2]]


def test_ranked_selection_falls_back_to_active_chunk_for_nan_influence():
    values = torch.tensor([float("nan"), float("nan"), float("nan")])
    mask = torch.tensor([False, True, False])
    assert select_ranked_or_fallback(values, mask, 5) == [1]
