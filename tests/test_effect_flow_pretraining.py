import json
import sys
import tempfile
from pathlib import Path

import torch
from transformers import BertConfig, BertForMaskedLM


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from effect_flow_pretraining_dataset import (  # noqa: E402
    BalancedDistributedBatchSampler,
    EffectFlowChunkDataset,
    build_or_load_corpus_index,
    choose_internal_holdout,
    choose_internal_holdout_by_contract,
    load_pattern_subset,
    load_tokenizer,
)
from effect_flow_pretraining_model import EffectFlowBertForPreTraining  # noqa: E402


def _write_vocab(path):
    tokens = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "PUSH1",
        "MSTORE",
        "CALL",
        "SLOAD",
        "SSTORE",
        "JUMPI",
        "ISZERO",
    ]
    path.write_text(
        json.dumps({"token_to_id": {token: index for index, token in enumerate(tokens)}}),
        encoding="utf-8",
    )
    return tokens


def _corpus_item(index, include_downstream_label=False):
    global_labels = [f"label_{value}" for value in range(17)]
    item = {
        "id": str(index),
        "source_dataset": "TEST",
        "source_split": "train",
        "input_ids": [2, 5, 6, 7, 8, 9, 10, 3],
        "attention_mask": [1] * 8,
        "effect_type_ids": [-100, 0, 15, 1, 2, 3, 5, -100],
        "etp_loss_mask": [0, 1, 1, 1, 1, 1, 1, 0],
        "efpp_pattern_labels": [int((index + value) % 3 == 0) for value in range(27)],
        "effect_relations": {"labels": [0] * 6},
        "chunk_vulnerability_evidence": [0.0] * 17,
        "vulnerability_template_matches": [0.0] * 17,
        "active_vulnerability_label_mask": [0.0] * 17,
        "global_vulnerability_label_names": global_labels,
    }
    if include_downstream_label:
        item["multi_labels"] = [1, 0]
    return item


def test_conservative_subset_has_exact_22_pattern_mapping():
    config, indices = load_pattern_subset(
        ROOT / "configs" / "effect_flow_efpp_conservative_22.json"
    )
    assert len(indices) == 22
    assert len(set(indices)) == 22
    assert set(config["excluded_pattern_names"]) == {
        "arithmetic_chain",
        "arithmetic_followed_by_guard_or_revert",
        "revert_or_invalid_present",
        "external_call_near_loop_candidate",
        "storage_write_near_loop_candidate",
    }


def test_train_only_index_dynamic_mask_and_balanced_batches():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        vocab_path = root / "vocab.json"
        _write_vocab(vocab_path)
        corpus_path = root / "train.jsonl"
        corpus_path.write_text(
            "".join(json.dumps(_corpus_item(index)) + "\n" for index in range(20)),
            encoding="utf-8",
        )
        offsets, stats = build_or_load_corpus_index(corpus_path)
        assert len(offsets) == 20
        assert stats["contains_downstream_labels"] is False
        train_offsets, valid_offsets = choose_internal_holdout(offsets, 0.1, 42)
        assert len(train_offsets) == 18
        assert len(valid_offsets) == 2
        _, pattern_indices = load_pattern_subset(
            ROOT / "configs" / "effect_flow_efpp_conservative_22.json"
        )
        tokenizer = load_tokenizer(vocab_path)
        dataset = EffectFlowChunkDataset(
            [
                {"name": "BJUT", "path": corpus_path, "offsets": train_offsets},
                {"name": "DIVE", "path": corpus_path, "offsets": train_offsets},
            ],
            tokenizer,
            pattern_indices,
            seed=42,
        )
        sample = dataset[(0, 0, 7)]
        assert sample["efpp_labels"].numel() == 22
        assert sample["mom_labels"][0].item() == -100
        assert sample["mom_labels"][-1].item() == -100
        assert (sample["mom_labels"] != -100).any()
        sampler = BalancedDistributedBatchSampler(
            [18, 18], batch_size=6, optimizer_steps_per_epoch=4, seed=42
        )
        for batch in sampler:
            assert sum(key[0] == 0 for key in batch) == 3
            assert sum(key[0] == 1 for key in batch) == 3
        dataset.close()
        del offsets


def test_index_rejects_downstream_labels():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "bad.jsonl"
        path.write_text(json.dumps(_corpus_item(0, True)) + "\n", encoding="utf-8")
        try:
            build_or_load_corpus_index(path)
        except ValueError as exc:
            assert "Downstream labels" in str(exc)
        else:
            raise AssertionError("Corpus index accepted a downstream vulnerability label.")


def test_contract_holdout_keeps_all_contract_chunks_together():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "train.jsonl"
        rows = []
        for contract_id in ("a", "a", "b", "b", "c", "c"):
            row = _corpus_item(len(rows))
            row["id"] = contract_id
            rows.append(row)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        offsets, _ = build_or_load_corpus_index(path)
        train_offsets, valid_offsets = choose_internal_holdout_by_contract(path, offsets, 1 / 3, 42)
        def ids_for(selected):
            values = set()
            with path.open("rb") as handle:
                for offset in selected:
                    handle.seek(int(offset))
                    values.add(json.loads(handle.readline().decode("utf-8"))["id"])
            return values
        train_ids = ids_for(train_offsets)
        valid_ids = ids_for(valid_offsets)
        assert train_ids.isdisjoint(valid_ids)
        assert len(train_ids | valid_ids) == 3
        del offsets


def test_three_head_model_loads_existing_encoder_and_backpropagates():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "base"
        config = BertConfig(
            vocab_size=32,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=64,
            max_position_embeddings=16,
        )
        BertForMaskedLM(config).save_pretrained(path)
        model = EffectFlowBertForPreTraining(
            path,
            vocab_size=32,
            etp_class_weights=[1.0] * 16,
            efpp_pos_weights=[1.0] * 22,
        )
        input_ids = torch.randint(5, 32, (2, 8))
        attention_mask = torch.ones_like(input_ids)
        pooling_mask = attention_mask.bool()
        mom_labels = torch.full_like(input_ids, -100)
        mom_labels[:, 2] = input_ids[:, 2]
        etp_labels = torch.randint(0, 16, (2, 8))
        efpp_labels = torch.randint(0, 2, (2, 22)).float()
        err_labels = torch.full((2,), -100, dtype=torch.long)
        vep_labels = torch.zeros((2, 18), dtype=torch.float32)
        vtm_labels = torch.zeros((2, 18), dtype=torch.float32)
        vulnerability_loss_mask = torch.zeros((2, 18), dtype=torch.float32)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pooling_mask=pooling_mask,
            mom_labels=mom_labels,
            etp_labels=etp_labels,
            efpp_labels=efpp_labels,
            err_labels=err_labels,
            vep_labels=vep_labels,
            vtm_labels=vtm_labels,
            vulnerability_loss_mask=vulnerability_loss_mask,
        )
        assert outputs["mom_logits"].shape == (2, 8, 32)
        assert outputs["etp_logits"].shape == (2, 8, 16)
        assert outputs["efpp_logits"].shape == (2, 22)
        assert torch.isfinite(outputs["loss"])
        assert outputs["mom_loss"].item() > 0
        assert outputs["etp_loss"].item() > 0
        assert outputs["efpp_loss"].item() > 0
        assert outputs["err_loss"].item() == 0
        outputs["loss"].backward()
        assert model.bert.embeddings.word_embeddings.weight.grad is not None
        assert model.err_head.weight.grad is not None
        assert torch.count_nonzero(model.err_head.weight.grad) == 0
        export_path = Path(directory) / "export"
        model.save_hf_model(
            export_path,
            {"efpp_pattern_subset": "conservative_22"},
        )
        exported = BertForMaskedLM.from_pretrained(export_path, local_files_only=True)
        assert exported.config.vocab_size == 32
        assert (export_path / "effect_flow_heads.pt").exists()
