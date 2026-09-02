import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_retrieval_validation import M0, M1, M2, build_retrieval_indices  # noqa: E402


def synthetic_memory():
    return {
        "ids": ["a", "b", "c"],
        "contract": torch.nn.functional.normalize(torch.eye(3, 4), dim=-1),
        "chunks": torch.randn(3, 2, 4),
        "mask": torch.tensor([[True, True], [True, False], [True, True]]),
        "labels": torch.tensor([[1, 0], [0, 1], [1, 1]], dtype=torch.float32),
    }


def test_train_retrieval_excludes_same_contract():
    memory = synthetic_memory()
    query = {key: value for key, value in memory.items()}
    query["ids"] = list(memory["ids"])
    result = build_retrieval_indices(
        {"contract_top_k": 2, "evidence_top_k": 1, "evidence_candidate_contracts": 2},
        memory,
        query,
        "train",
    )
    for row, index in enumerate(result["contract_neighbors"][:, 0].tolist()):
        assert index != row


def test_retrieval_models_have_expected_shapes():
    query = torch.randn(2, 4)
    contract = torch.randn(2, 4)
    evidence = torch.randn(2, 2, 4)
    assert M0(4, 8, 2, 0.0)(query).shape == (2, 2)
    assert M1(4, 8, 2, 0.0)(query, contract_evidence=contract).shape == (2, 2)
    assert M2(4, 8, 2, 0.0)(query, evidence=evidence).shape == (2, 2)
