"""Chunk-level MLM dataset backed by the train-only stack relation cache."""

import torch
from torch.utils.data import Dataset


class StackRelationMLMDataset(Dataset):
    def __init__(self, path, seed=42, mlm_probability=0.15):
        payload = torch.load(path, map_location="cpu")
        if payload.get("schema") != "main6_opcode_stack_relational_v2":
            raise ValueError("stack relation cache is stale; rebuild it with the v2 extractor")
        self.input_ids = payload["input_ids"].long()
        self.attention_mask = payload["attention_mask"].bool()
        self.stack_state = payload["stack_state"].long()
        self.boundary_state = payload["boundary_state"].long()
        self.edge_offsets = payload["edge_offsets"].long()
        self.edge_src = payload["edge_src"].long()
        self.edge_dst = payload["edge_dst"].long()
        self.edge_type = payload["edge_type"].long()
        self.edge_slot = payload["edge_slot"].long()
        self.edge_distance = payload["edge_distance"].long()
        self.edge_confidence = payload["edge_confidence"].float()
        self.seed = int(seed)
        self.mlm_probability = float(mlm_probability)
        self.epoch = 0
        self.special_ids = {0, 1, 2, 3, 4}
        self.mask_token_id = 4
        self.random_token_ids = []

    def set_random_token_ids(self, token_ids, mask_token_id):
        self.random_token_ids = [int(x) for x in token_ids if int(x) not in self.special_ids]
        self.mask_token_id = int(mask_token_id)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.input_ids.shape[0]

    def __getitem__(self, index):
        original = self.input_ids[index].clone()
        labels = original.clone()
        generator = torch.Generator().manual_seed(self.seed + self.epoch * max(1, len(self)) + int(index))
        eligible = self.attention_mask[index] & ~torch.isin(original, torch.tensor(list(self.special_ids)))
        masked = torch.bernoulli(torch.full(original.shape, self.mlm_probability), generator=generator).bool() & eligible
        if not masked.any() and eligible.any():
            candidates = eligible.nonzero(as_tuple=False).view(-1)
            forced = torch.randint(candidates.numel(), (1,), generator=generator).item()
            masked[candidates[forced]] = True
        labels[~masked] = -100
        replacement = torch.rand(original.shape, generator=generator)
        masked_input = original.clone()
        masked_input[masked & (replacement < 0.8)] = self.mask_token_id
        if self.random_token_ids:
            random_pos = (masked & (replacement >= 0.8) & (replacement < 0.9)).nonzero(as_tuple=False).view(-1)
            if random_pos.numel():
                choices = torch.randint(len(self.random_token_ids), (random_pos.numel(),), generator=generator)
                masked_input[random_pos] = torch.tensor(self.random_token_ids, dtype=torch.long)[choices]
        left, right = int(self.edge_offsets[index]), int(self.edge_offsets[index + 1])
        return {
            "input_ids": masked_input, "attention_mask": self.attention_mask[index], "stack_state": self.stack_state[index],
            "boundary_state": self.boundary_state[index], "labels": labels,
            "edge_src": self.edge_src[left:right], "edge_dst": self.edge_dst[left:right],
            "edge_type": self.edge_type[left:right], "edge_slot": self.edge_slot[left:right],
            "edge_distance": self.edge_distance[left:right], "edge_confidence": self.edge_confidence[left:right],
        }


def collate_stack_relation_mlm(batch):
    offsets = [0]; fields = {name: [] for name in ("edge_src", "edge_dst", "edge_type", "edge_slot", "edge_distance", "edge_confidence")}
    for item in batch:
        for name in fields:
            fields[name].append(item[name])
        offsets.append(offsets[-1] + item["edge_src"].numel())
    empty_long = torch.empty(0, dtype=torch.long); empty_float = torch.empty(0)
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]), "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "stack_state": torch.stack([x["stack_state"] for x in batch]), "boundary_state": torch.stack([x["boundary_state"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch]), "edge_offsets": torch.tensor(offsets, dtype=torch.long),
        **{name: torch.cat(values) if values else (empty_float if name == "edge_confidence" else empty_long) for name, values in fields.items()},
    }
