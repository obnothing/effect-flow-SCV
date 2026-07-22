import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from effect_flow_ontology import RELATION_TYPES
from evm_tokenizer import EVMOpcodeTokenizer


FORBIDDEN_LABEL_FIELDS = {
    "binary_label",
    "multi_labels",
    "labels",
    "vulnerability_labels",
}
NUM_EFFECT_TYPES = 16
ORIGINAL_PATTERN_COUNT = 27
NUM_RELATION_TYPES = len(RELATION_TYPES)


def _cache_paths(corpus_path):
    corpus_path = Path(corpus_path)
    return (
        corpus_path.with_suffix(corpus_path.suffix + ".stage16b_offsets.npy"),
        corpus_path.with_suffix(corpus_path.suffix + ".stage16b_stats.json"),
    )


def _cache_valid(corpus_path, offset_path, stats_path):
    if not offset_path.exists() or not stats_path.exists():
        return False
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    source = Path(corpus_path).stat()
    return (
        int(stats.get("source_size", -1)) == source.st_size
        and int(stats.get("source_mtime_ns", -1)) == source.st_mtime_ns
    )


def build_or_load_corpus_index(corpus_path, force=False, progress_callback=None):
    """Build a byte-offset index and train-only weak-label statistics."""
    corpus_path = Path(corpus_path)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Effect-flow corpus not found: {corpus_path}")
    offset_path, stats_path = _cache_paths(corpus_path)
    if not force and _cache_valid(corpus_path, offset_path, stats_path):
        return (
            np.load(offset_path, mmap_mode="r"),
            json.loads(stats_path.read_text(encoding="utf-8")),
        )

    offsets = []
    etp_counts = [0] * NUM_EFFECT_TYPES
    efpp_positive_counts = [0] * ORIGINAL_PATTERN_COUNT
    relation_counts = [0] * NUM_RELATION_TYPES
    vep_positive_counts = None
    vtm_positive_counts = None
    vulnerability_label_count = None
    vulnerability_label_names = None
    active_etp_tokens = 0
    with corpus_path.open("rb") as handle:
        line_no = 0
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.strip():
                continue
            line_no += 1
            item = json.loads(line.decode("utf-8"))
            forbidden = FORBIDDEN_LABEL_FIELDS.intersection(item)
            if forbidden:
                raise ValueError(
                    f"Downstream labels found in pretraining corpus {corpus_path}: {forbidden}"
                )
            if item.get("source_split") != "train":
                raise ValueError(
                    f"Stage 16B accepts train chunks only; got source_split="
                    f"{item.get('source_split')} in {corpus_path}:{line_no}"
                )
            if len(item.get("efpp_pattern_labels", [])) != ORIGINAL_PATTERN_COUNT:
                raise ValueError(f"Invalid EFPP label width at {corpus_path}:{line_no}")
            relation_labels = item.get("effect_relations", {}).get("labels", [])
            if len(relation_labels) != NUM_RELATION_TYPES:
                raise ValueError(f"Invalid relation label width at {corpus_path}:{line_no}")
            effect_ids = item.get("effect_type_ids", [])
            loss_mask = item.get("etp_loss_mask", [])
            if len(effect_ids) != len(loss_mask):
                raise ValueError(f"ETP labels/mask length mismatch at {corpus_path}:{line_no}")
            vep = item.get("chunk_vulnerability_evidence", [])
            vtm = item.get("vulnerability_template_matches", [])
            active_mask = item.get("active_vulnerability_label_mask", [])
            if vulnerability_label_count is None:
                vulnerability_label_count = len(vtm)
                vulnerability_label_names = item.get("global_vulnerability_label_names", [])
                vep_positive_counts = [0] * vulnerability_label_count
                vtm_positive_counts = [0] * vulnerability_label_count
            if len(vep) != vulnerability_label_count or len(vtm) != vulnerability_label_count:
                raise ValueError(f"Template target width mismatch at {corpus_path}:{line_no}")
            if len(active_mask) != vulnerability_label_count:
                raise ValueError(f"Active vulnerability mask width mismatch at {corpus_path}:{line_no}")
            if item.get("global_vulnerability_label_names", []) != vulnerability_label_names:
                raise ValueError(f"Inconsistent global vulnerability label names at {corpus_path}:{line_no}")
            offsets.append(offset)
            for effect_id, active in zip(effect_ids, loss_mask):
                if active:
                    effect_id = int(effect_id)
                    if not 0 <= effect_id < NUM_EFFECT_TYPES:
                        raise ValueError(f"Invalid effect type {effect_id} at line {line_no}")
                    etp_counts[effect_id] += 1
                    active_etp_tokens += 1
            for index, value in enumerate(item["efpp_pattern_labels"]):
                efpp_positive_counts[index] += int(value)
            for index, value in enumerate(relation_labels):
                relation_counts[index] += int(value)
            for index, value in enumerate(vep):
                vep_positive_counts[index] += int(float(value) > 0.0)
            for index, value in enumerate(vtm):
                vtm_positive_counts[index] += int(float(value) > 0.0)
            if progress_callback and line_no % 10000 == 0:
                progress_callback(line_no)

    source = corpus_path.stat()
    offsets_array = np.asarray(offsets, dtype=np.int64)
    offset_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(offset_path, offsets_array)
    stats = {
        "source_file": str(corpus_path),
        "source_size": source.st_size,
        "source_mtime_ns": source.st_mtime_ns,
        "sample_count": len(offsets),
        "active_etp_tokens": active_etp_tokens,
        "etp_counts": etp_counts,
        "efpp_positive_counts": efpp_positive_counts,
        "relation_counts": relation_counts,
        "vulnerability_label_count": vulnerability_label_count,
        "global_vulnerability_label_names": vulnerability_label_names,
        "vep_positive_counts": vep_positive_counts,
        "vtm_positive_counts": vtm_positive_counts,
        "contains_downstream_labels": False,
        "source_split": "train",
    }
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return np.load(offset_path, mmap_mode="r"), stats


def choose_internal_holdout(offsets, ratio, seed):
    count = len(offsets)
    valid_count = max(1, int(round(count * float(ratio)))) if count else 0
    rng = random.Random(int(seed))
    valid_indices = set(rng.sample(range(count), valid_count))
    train_offsets = np.asarray(
        [offset for index, offset in enumerate(offsets) if index not in valid_indices],
        dtype=np.int64,
    )
    valid_offsets = np.asarray(
        [offset for index, offset in enumerate(offsets) if index in valid_indices],
        dtype=np.int64,
    )
    return train_offsets, valid_offsets


def choose_internal_holdout_by_contract(corpus_path, offsets, ratio, seed):
    """Split an effect-flow corpus by contract id so chunks cannot cross holdout."""
    grouped_offsets = {}
    with Path(corpus_path).open("rb") as handle:
        for raw_offset in offsets:
            offset = int(raw_offset)
            handle.seek(offset)
            item = json.loads(handle.readline().decode("utf-8"))
            contract_id = str(item.get("id", ""))
            if not contract_id:
                raise ValueError(f"Effect-flow corpus item at {offset} has no contract id")
            grouped_offsets.setdefault(contract_id, []).append(offset)
    contract_ids = sorted(grouped_offsets)
    valid_count = max(1, int(round(len(contract_ids) * float(ratio)))) if contract_ids else 0
    rng = random.Random(int(seed))
    valid_ids = set(rng.sample(contract_ids, valid_count))
    train_offsets = []
    valid_offsets = []
    for contract_id, contract_offsets in grouped_offsets.items():
        target = valid_offsets if contract_id in valid_ids else train_offsets
        target.extend(contract_offsets)
    return np.asarray(train_offsets, dtype=np.int64), np.asarray(valid_offsets, dtype=np.int64)


def count_offsets_labels(corpus_path, offsets):
    etp_counts = [0] * NUM_EFFECT_TYPES
    efpp_counts = [0] * ORIGINAL_PATTERN_COUNT
    relation_counts = [0] * NUM_RELATION_TYPES
    vep_counts = None
    vtm_counts = None
    vulnerability_label_count = None
    vulnerability_label_names = None
    active_tokens = 0
    with Path(corpus_path).open("rb") as handle:
        for offset in sorted(int(value) for value in offsets):
            handle.seek(offset)
            item = json.loads(handle.readline().decode("utf-8"))
            for effect_id, active in zip(
                item["effect_type_ids"], item["etp_loss_mask"]
            ):
                if active:
                    etp_counts[int(effect_id)] += 1
                    active_tokens += 1
            for index, value in enumerate(item["efpp_pattern_labels"]):
                efpp_counts[index] += int(value)
            relation_labels = item["effect_relations"]["labels"]
            for index, value in enumerate(relation_labels):
                relation_counts[index] += int(value)
            vep = item["chunk_vulnerability_evidence"]
            vtm = item["vulnerability_template_matches"]
            active_mask = item["active_vulnerability_label_mask"]
            if vulnerability_label_count is None:
                vulnerability_label_count = len(vtm)
                vulnerability_label_names = item.get("global_vulnerability_label_names", [])
                vep_counts = [0] * vulnerability_label_count
                vtm_counts = [0] * vulnerability_label_count
            if len(active_mask) != vulnerability_label_count:
                raise ValueError("active_vulnerability_label_mask width mismatch")
            if item.get("global_vulnerability_label_names", []) != vulnerability_label_names:
                raise ValueError("global_vulnerability_label_names mismatch")
            for index, value in enumerate(vep):
                vep_counts[index] += int(float(value) > 0.0)
            for index, value in enumerate(vtm):
                vtm_counts[index] += int(float(value) > 0.0)
    return {
        "sample_count": len(offsets),
        "active_etp_tokens": active_tokens,
        "etp_counts": etp_counts,
        "efpp_positive_counts": efpp_counts,
        "relation_counts": relation_counts,
        "vulnerability_label_count": vulnerability_label_count,
        "global_vulnerability_label_names": vulnerability_label_names,
        "vep_positive_counts": vep_counts,
        "vtm_positive_counts": vtm_counts,
    }


def subtract_stats(full_stats, heldout_stats):
    return {
        "sample_count": full_stats["sample_count"] - heldout_stats["sample_count"],
        "active_etp_tokens": full_stats["active_etp_tokens"]
        - heldout_stats["active_etp_tokens"],
        "etp_counts": [
            int(left - right)
            for left, right in zip(full_stats["etp_counts"], heldout_stats["etp_counts"])
        ],
        "efpp_positive_counts": [
            int(left - right)
            for left, right in zip(
                full_stats["efpp_positive_counts"],
                heldout_stats["efpp_positive_counts"],
            )
        ],
        "relation_counts": [
            int(left - right)
            for left, right in zip(
                full_stats["relation_counts"],
                heldout_stats["relation_counts"],
            )
        ],
        "vulnerability_label_count": full_stats["vulnerability_label_count"],
        "global_vulnerability_label_names": full_stats["global_vulnerability_label_names"],
        "vep_positive_counts": [
            int(left - right)
            for left, right in zip(
                full_stats["vep_positive_counts"],
                heldout_stats["vep_positive_counts"],
            )
        ],
        "vtm_positive_counts": [
            int(left - right)
            for left, right in zip(
                full_stats["vtm_positive_counts"],
                heldout_stats["vtm_positive_counts"],
            )
        ],
    }


def calculate_balanced_class_weights(
    train_stats_by_dataset,
    included_original_pattern_indices,
    max_etp_class_weight=5.0,
    max_efpp_pos_weight=10.0,
    normal_class_weight_scale=0.5,
):
    datasets = sorted(train_stats_by_dataset)
    etp_rates = []
    efpp_rates = []
    relation_rates = []
    vep_rates = []
    vtm_rates = []
    for dataset in datasets:
        stats = train_stats_by_dataset[dataset]
        etp_total = max(1, stats["active_etp_tokens"])
        sample_total = max(1, stats["sample_count"])
        etp_rates.append([count / etp_total for count in stats["etp_counts"]])
        efpp_rates.append(
            [count / sample_total for count in stats["efpp_positive_counts"]]
        )
        relation_rates.append(
            [count / sample_total for count in stats["relation_counts"]]
        )
        vep_rates.append(
            [count / sample_total for count in stats["vep_positive_counts"]]
        )
        vtm_rates.append(
            [count / sample_total for count in stats["vtm_positive_counts"]]
        )
    balanced_etp_rates = [
        sum(rates[index] for rates in etp_rates) / len(etp_rates)
        for index in range(NUM_EFFECT_TYPES)
    ]
    etp_weights = []
    for index, frequency in enumerate(balanced_etp_rates):
        weight = math.sqrt(1.0 / max(NUM_EFFECT_TYPES * frequency, 1e-12))
        if index == 0:
            weight *= float(normal_class_weight_scale)
        etp_weights.append(min(float(max_etp_class_weight), weight))

    balanced_pattern_rates = [
        sum(rates[index] for rates in efpp_rates) / len(efpp_rates)
        for index in range(ORIGINAL_PATTERN_COUNT)
    ]
    selected_rates = [balanced_pattern_rates[index] for index in included_original_pattern_indices]
    efpp_pos_weights = [
        min(
            float(max_efpp_pos_weight),
            math.sqrt(max(0.0, 1.0 - rate) / max(rate, 1e-12)),
        )
        for rate in selected_rates
    ]
    balanced_relation_rates = [
        sum(rates[index] for rates in relation_rates) / len(relation_rates)
        for index in range(NUM_RELATION_TYPES)
    ]
    relation_class_weights = [
        min(float(max_etp_class_weight), math.sqrt(1.0 / max(NUM_RELATION_TYPES * rate, 1e-12)))
        for rate in balanced_relation_rates
    ]
    balanced_vep_rates = [
        sum(rates[index] for rates in vep_rates) / len(vep_rates)
        for index in range(len(vep_rates[0]))
    ]
    balanced_vtm_rates = [
        sum(rates[index] for rates in vtm_rates) / len(vtm_rates)
        for index in range(len(vtm_rates[0]))
    ]
    vep_pos_weights = [
        min(
            float(max_efpp_pos_weight),
            math.sqrt(max(0.0, 1.0 - rate) / max(rate, 1e-12)),
        )
        for rate in balanced_vep_rates
    ]
    vtm_pos_weights = [
        min(
            float(max_efpp_pos_weight),
            math.sqrt(max(0.0, 1.0 - rate) / max(rate, 1e-12)),
        )
        for rate in balanced_vtm_rates
    ]
    return {
        "etp_class_weights": etp_weights,
        "efpp_pos_weights": efpp_pos_weights,
        "relation_class_weights": relation_class_weights,
        "vep_pos_weights": vep_pos_weights,
        "vtm_pos_weights": vtm_pos_weights,
        "balanced_etp_frequencies": balanced_etp_rates,
        "balanced_efpp_positive_rates": selected_rates,
        "balanced_relation_rates": balanced_relation_rates,
        "balanced_vep_positive_rates": balanced_vep_rates,
        "balanced_vtm_positive_rates": balanced_vtm_rates,
        "weighting_distribution": "equal-weight average of BJUT and DIVE train distributions",
    }


class EffectFlowChunkDataset(Dataset):
    def __init__(
        self,
        sources,
        tokenizer,
        included_original_pattern_indices,
        mlm_probability=0.15,
        seed=42,
    ):
        self.sources = sources
        self.tokenizer = tokenizer
        self.pattern_indices = list(included_original_pattern_indices)
        self.mlm_probability = float(mlm_probability)
        self.seed = int(seed)
        self.epoch = 0
        self._files = {}
        self.pad_token_id = tokenizer.vocab["[PAD]"]
        self.cls_token_id = tokenizer.vocab["[CLS]"]
        self.sep_token_id = tokenizer.vocab["[SEP]"]
        self.mask_token_id = tokenizer.vocab["[MASK]"]
        self.original_mask_token_id = self.mask_token_id
        special_ids = {
            tokenizer.vocab[token]
            for token in ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")
        }
        self.random_token_ids = [
            token_id for token_id in tokenizer.vocab.values() if token_id not in special_ids
        ]
        if not self.random_token_ids:
            raise ValueError("EVM vocabulary has no non-special random replacement tokens.")
        self.cumulative = []
        running = 0
        for source in sources:
            running += len(source["offsets"])
            self.cumulative.append(running)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_files"] = {}
        return state

    def close(self):
        for handle in self._files.values():
            handle.close()
        self._files = {}

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.cumulative[-1] if self.cumulative else 0

    def _resolve_key(self, key):
        draw_id = 0
        if isinstance(key, tuple):
            source_index, local_index, draw_id = key
            return int(source_index), int(local_index), int(draw_id)
        index = int(key)
        previous = 0
        for source_index, end in enumerate(self.cumulative):
            if index < end:
                return source_index, index - previous, draw_id
            previous = end
        raise IndexError(index)

    def _file(self, source_index):
        if source_index not in self._files:
            self._files[source_index] = Path(
                self.sources[source_index]["path"]
            ).open("rb")
        return self._files[source_index]

    def _mask_tokens(self, original_ids, pooling_mask, seed_value):
        labels = original_ids.clone()
        generator = torch.Generator()
        generator.manual_seed(int(seed_value) % (2**63 - 1))
        probabilities = torch.full(original_ids.shape, self.mlm_probability)
        probabilities.masked_fill_(~pooling_mask, 0.0)
        probabilities.masked_fill_(original_ids == self.original_mask_token_id, 0.0)
        masked = torch.bernoulli(probabilities, generator=generator).bool()
        if not masked.any():
            candidates = pooling_mask.nonzero(as_tuple=False).view(-1)
            if candidates.numel():
                selected = torch.randint(
                    0, candidates.numel(), (1,), generator=generator
                ).item()
                masked[candidates[selected]] = True
        labels[~masked] = -100
        random_values = torch.rand(original_ids.shape, generator=generator)
        replace_mask = masked & (random_values < 0.8)
        replace_random = masked & (random_values >= 0.8) & (random_values < 0.9)
        input_ids = original_ids.clone()
        input_ids[replace_mask] = self.mask_token_id
        random_positions = replace_random.nonzero(as_tuple=False).view(-1)
        if random_positions.numel():
            choices = torch.randint(
                0,
                len(self.random_token_ids),
                (random_positions.numel(),),
                generator=generator,
            )
            replacement = torch.tensor(self.random_token_ids, dtype=torch.long)[choices]
            input_ids[random_positions] = replacement
        return input_ids, labels

    def __getitem__(self, key):
        source_index, local_index, draw_id = self._resolve_key(key)
        source = self.sources[source_index]
        offset = int(source["offsets"][local_index])
        handle = self._file(source_index)
        handle.seek(offset)
        item = json.loads(handle.readline().decode("utf-8"))
        input_ids = torch.tensor(item["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(item["attention_mask"], dtype=torch.long)
        if input_ids.numel() != attention_mask.numel():
            raise ValueError("input_ids and attention_mask length mismatch.")
        pooling_mask = attention_mask.bool()
        pooling_mask &= input_ids != self.pad_token_id
        pooling_mask &= input_ids != self.cls_token_id
        pooling_mask &= input_ids != self.sep_token_id
        etp_labels = torch.tensor(item["effect_type_ids"], dtype=torch.long)
        etp_loss_mask = torch.tensor(item["etp_loss_mask"], dtype=torch.bool)
        etp_labels[~etp_loss_mask] = -100
        efpp_labels = torch.tensor(
            [item["efpp_pattern_labels"][index] for index in self.pattern_indices],
            dtype=torch.float32,
        )
        relation_labels = torch.tensor(
            item["effect_relations"]["labels"],
            dtype=torch.float32,
        )
        active_relations = [idx for idx, value in enumerate(relation_labels.tolist()) if value > 0]
        err_label = active_relations[0] if active_relations else -100
        vep_labels = torch.tensor(item["chunk_vulnerability_evidence"], dtype=torch.float32)
        vtm_labels = torch.tensor(item["vulnerability_template_matches"], dtype=torch.float32)
        vulnerability_loss_mask = torch.tensor(
            item["active_vulnerability_label_mask"],
            dtype=torch.float32,
        )
        seed_value = (
            self.seed
            + self.epoch * 1_000_000_007
            + source_index * 10_000_019
            + local_index * 97
            + draw_id
        )
        masked_ids, mom_labels = self._mask_tokens(
            input_ids, pooling_mask, seed_value
        )
        return {
            "input_ids": masked_ids,
            "attention_mask": attention_mask,
            "token_type_ids": torch.zeros_like(input_ids),
            "pooling_mask": pooling_mask,
            "mom_labels": mom_labels,
            "etp_labels": etp_labels,
            "efpp_labels": efpp_labels,
            "err_labels": torch.tensor(err_label, dtype=torch.long),
            "relation_labels": relation_labels,
            "vep_labels": vep_labels,
            "vtm_labels": vtm_labels,
            "vulnerability_loss_mask": vulnerability_loss_mask,
            "source_dataset_id": torch.tensor(source_index, dtype=torch.long),
        }


class BalancedDistributedBatchSampler(Sampler):
    def __init__(
        self,
        source_sizes,
        batch_size,
        optimizer_steps_per_epoch,
        gradient_accumulation_steps=1,
        seed=42,
        rank=0,
        world_size=1,
    ):
        if len(source_sizes) != 2 or any(size <= 0 for size in source_sizes):
            raise ValueError("Balanced sampling requires two non-empty datasets.")
        self.source_sizes = [int(size) for size in source_sizes]
        self.batch_size = int(batch_size)
        self.optimizer_steps_per_epoch = int(optimizer_steps_per_epoch)
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.epoch = 0
        if self.batch_size < 2:
            raise ValueError("batch_size must be at least 2 for balanced batches.")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.optimizer_steps_per_epoch * self.gradient_accumulation_steps

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch * 1009 + self.rank * 9176)
        for step in range(len(self)):
            left_count = self.batch_size // 2
            if self.batch_size % 2 and step % 2 == 0:
                left_count += 1
            right_count = self.batch_size - left_count
            batch = []
            draw_base = (
                self.epoch * len(self) * self.batch_size
                + step * self.batch_size
                + self.rank * len(self) * self.batch_size * 17
            )
            for source_index, count in ((0, left_count), (1, right_count)):
                for local_draw in range(count):
                    local_index = rng.randrange(self.source_sizes[source_index])
                    batch.append(
                        (source_index, local_index, draw_base + len(batch) + local_draw)
                    )
            rng.shuffle(batch)
            yield batch


def load_pattern_subset(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("subset_name") != "conservative_22":
        raise ValueError("Stage 16B requires efpp_pattern_subset=conservative_22.")
    included = data["included_pattern_names"]
    excluded = data["excluded_pattern_names"]
    if len(included) != 22 or len(excluded) != 5:
        raise ValueError("Conservative EFPP subset must include 22 and exclude 5 patterns.")
    original_indices = [data["original_pattern_index"][name] for name in included]
    if set(original_indices) & {
        data["original_pattern_index"][name] for name in excluded
    }:
        raise ValueError("Included and excluded EFPP pattern indices overlap.")
    return data, original_indices


def load_tokenizer(vocab_path):
    tokenizer = EVMOpcodeTokenizer.from_vocab_file(vocab_path)
    required = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    missing = [token for token in required if token not in tokenizer.vocab]
    if missing:
        raise ValueError(f"EVM tokenizer missing special tokens: {missing}")
    return tokenizer
