import json
from pathlib import Path

import torch
import yaml

from effect_flow_schema import EFFECT_TYPES, EFPP_PATTERNS
from effect_flow_utils import GLOBAL_VULNERABILITY_LABELS


ROLE_NAMES = [
    "risk_behavior",
    "protective_behavior",
    "missing_check_behavior",
]
RELATION_TYPES = [
    "call_then_state_write",
    "state_write_then_call",
    "call_then_return_check",
    "state_write_then_auth_guard",
    "env_then_branch",
    "div_then_mul",
]


def resolve_path(path, project_root=None):
    path = Path(path)
    if path.is_absolute() or project_root is None:
        return path
    return Path(project_root) / path


def load_yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_ontology(path):
    ontology = load_yaml(path)
    effect_names = [row["name"] for row in ontology["effect_types"]]
    if effect_names != EFFECT_TYPES:
        raise ValueError(
            "Ontology effect_types must exactly match effect_flow_schema.EFFECT_TYPES"
        )
    role_names = ontology.get("role_names", ROLE_NAMES)
    if role_names != ROLE_NAMES:
        raise ValueError(f"Ontology role_names must be {ROLE_NAMES}")
    relation_types = ontology.get("relation_types", RELATION_TYPES)
    if relation_types != RELATION_TYPES:
        raise ValueError(f"Ontology relation_types must be {RELATION_TYPES}")
    grouped_patterns = []
    for group_name in ("risk_pattern", "protective_pattern", "missing_check_pattern"):
        grouped_patterns.extend(ontology["pattern_groups"].get(group_name, []))
    unknown = sorted(set(grouped_patterns) - set(EFPP_PATTERNS))
    if unknown:
        raise ValueError(f"Ontology contains unknown patterns: {unknown}")
    if len(set(grouped_patterns)) != len(grouped_patterns):
        raise ValueError("Ontology pattern_groups contain duplicated patterns.")
    return ontology


def load_vulnerability_templates(path, ontology, expected_label_names=None):
    payload = load_yaml(path)
    templates = payload["templates"]
    if expected_label_names is not None:
        missing = [name for name in expected_label_names if name not in templates]
        if missing:
            raise ValueError(f"Missing vulnerability templates: {missing}")
    for label_name, spec in templates.items():
        for key in (
            "role_focus",
            "required_effect_types",
            "required_patterns",
            "optional_patterns",
            "forbidden_or_counter_patterns",
            "critical_relations",
            "weak_patterns",
        ):
            if key not in spec:
                raise ValueError(f"{label_name} missing template field: {key}")
        for role_key in ("risk", "protective", "missing_check"):
            if role_key not in spec["role_focus"]:
                raise ValueError(f"{label_name}.role_focus missing {role_key}")
        _validate_template_lists(label_name, spec, ontology)
    return templates


def _validate_template_lists(label_name, spec, ontology):
    valid_effects = set(EFFECT_TYPES)
    valid_patterns = set(EFPP_PATTERNS)
    valid_relations = set(ontology.get("relation_types", RELATION_TYPES))
    for effect_name in spec["required_effect_types"]:
        if effect_name not in valid_effects:
            raise ValueError(f"{label_name} unknown effect type: {effect_name}")
    for field in (
        "required_patterns",
        "optional_patterns",
        "forbidden_or_counter_patterns",
        "weak_patterns",
    ):
        for pattern_name in spec[field]:
            if pattern_name not in valid_patterns:
                raise ValueError(f"{label_name} unknown pattern in {field}: {pattern_name}")
    for relation_name in spec["critical_relations"]:
        if relation_name not in valid_relations:
            raise ValueError(
                f"{label_name} unknown relation in critical_relations: {relation_name}"
            )


def _one_hot(names, vocabulary):
    vocab_to_id = {name: index for index, name in enumerate(vocabulary)}
    vector = [0.0] * len(vocabulary)
    for name in names:
        if name in vocab_to_id:
            vector[vocab_to_id[name]] = 1.0
    return vector


def build_template_tensor_bundle(label_names, templates, ontology):
    role_vocab = ROLE_NAMES
    pattern_vocab = EFPP_PATTERNS
    effect_vocab = EFFECT_TYPES
    relation_vocab = ontology.get("relation_types", RELATION_TYPES)
    records = []
    for label_name in label_names:
        spec = templates[label_name]
        role_focus = spec["role_focus"]
        record = {
            "label_name": label_name,
            "role_focus_vector": [
                1.0 if role_focus["risk"] else 0.0,
                1.0 if role_focus["protective"] else 0.0,
                1.0 if role_focus["missing_check"] else 0.0,
            ],
            "required_effect_types": _one_hot(spec["required_effect_types"], effect_vocab),
            "required_patterns": _one_hot(spec["required_patterns"], pattern_vocab),
            "optional_patterns": _one_hot(spec["optional_patterns"], pattern_vocab),
            "forbidden_or_counter_patterns": _one_hot(
                spec["forbidden_or_counter_patterns"], pattern_vocab
            ),
            "critical_relations": _one_hot(spec["critical_relations"], relation_vocab),
            "weak_patterns": _one_hot(spec["weak_patterns"], pattern_vocab),
            "role_risk_patterns": _one_hot(role_focus["risk"], pattern_vocab),
            "role_protective_patterns": _one_hot(role_focus["protective"], pattern_vocab),
            "role_missing_check_patterns": _one_hot(
                role_focus["missing_check"], pattern_vocab
            ),
        }
        records.append(record)
    bundle = {
        "label_names": label_names,
        "role_names": role_vocab,
        "effect_type_names": effect_vocab,
        "pattern_names": pattern_vocab,
        "relation_names": relation_vocab,
        "role_focus_vector": torch.tensor(
            [row["role_focus_vector"] for row in records], dtype=torch.float32
        ),
        "required_effect_types": torch.tensor(
            [row["required_effect_types"] for row in records], dtype=torch.float32
        ),
        "required_patterns": torch.tensor(
            [row["required_patterns"] for row in records], dtype=torch.float32
        ),
        "optional_patterns": torch.tensor(
            [row["optional_patterns"] for row in records], dtype=torch.float32
        ),
        "forbidden_or_counter_patterns": torch.tensor(
            [row["forbidden_or_counter_patterns"] for row in records],
            dtype=torch.float32,
        ),
        "critical_relations": torch.tensor(
            [row["critical_relations"] for row in records], dtype=torch.float32
        ),
        "weak_patterns": torch.tensor(
            [row["weak_patterns"] for row in records], dtype=torch.float32
        ),
        "role_risk_patterns": torch.tensor(
            [row["role_risk_patterns"] for row in records], dtype=torch.float32
        ),
        "role_protective_patterns": torch.tensor(
            [row["role_protective_patterns"] for row in records], dtype=torch.float32
        ),
        "role_missing_check_patterns": torch.tensor(
            [row["role_missing_check_patterns"] for row in records], dtype=torch.float32
        ),
    }
    bundle["template_feature_vector"] = torch.cat(
        [
            bundle["role_focus_vector"],
            bundle["required_effect_types"],
            bundle["required_patterns"],
            bundle["optional_patterns"],
            bundle["forbidden_or_counter_patterns"],
            bundle["critical_relations"],
            bundle["weak_patterns"],
            bundle["role_risk_patterns"],
            bundle["role_protective_patterns"],
            bundle["role_missing_check_patterns"],
        ],
        dim=1,
    )
    return bundle


def global_vulnerability_index():
    return {
        label_name: index for index, label_name in enumerate(GLOBAL_VULNERABILITY_LABELS)
    }


def template_score_vector(
    label_names,
    templates,
    ontology,
    effect_histogram,
    pattern_labels,
    relation_labels,
):
    effect_index = {name: idx for idx, name in enumerate(EFFECT_TYPES)}
    pattern_index = {name: idx for idx, name in enumerate(EFPP_PATTERNS)}
    relation_index = {
        name: idx
        for idx, name in enumerate(ontology.get("relation_types", RELATION_TYPES))
    }
    global_index = global_vulnerability_index()
    global_scores = [0.0] * len(GLOBAL_VULNERABILITY_LABELS)
    for label_name in label_names:
        spec = templates[label_name]
        required_effect_score = _mean_lookup(
            effect_histogram,
            spec["required_effect_types"],
            effect_index,
        )
        required_pattern_score = _mean_lookup(
            pattern_labels,
            spec["required_patterns"],
            pattern_index,
        )
        optional_pattern_score = _mean_lookup(
            pattern_labels,
            spec["optional_patterns"],
            pattern_index,
        )
        forbidden_score = _mean_lookup(
            pattern_labels,
            spec["forbidden_or_counter_patterns"],
            pattern_index,
        )
        relation_score = _mean_lookup(
            relation_labels,
            spec["critical_relations"],
            relation_index,
        )
        weak_score = _mean_lookup(
            pattern_labels,
            spec["weak_patterns"],
            pattern_index,
        )
        role_focus = spec["role_focus"]
        risk_score = _mean_lookup(pattern_labels, role_focus["risk"], pattern_index)
        protective_score = _mean_lookup(
            pattern_labels,
            role_focus["protective"],
            pattern_index,
        )
        missing_score = _mean_lookup(
            pattern_labels,
            role_focus["missing_check"],
            pattern_index,
        )
        score = (
            0.25 * required_effect_score
            + 0.25 * required_pattern_score
            + 0.10 * optional_pattern_score
            + 0.15 * relation_score
            + 0.10 * risk_score
            + 0.10 * missing_score
            + 0.05 * weak_score
            - 0.20 * forbidden_score
            - 0.10 * protective_score
        )
        global_scores[global_index[label_name]] = float(max(0.0, min(1.0, score)))
    return global_scores


def pseudo_evidence_vector(global_template_scores, contract_multi_labels, active_label_names):
    global_index = global_vulnerability_index()
    evidence = [0.0] * len(GLOBAL_VULNERABILITY_LABELS)
    for label_id, label_name in enumerate(active_label_names):
        if label_id >= len(contract_multi_labels):
            break
        if int(contract_multi_labels[label_id]) == 1:
            global_id = global_index[label_name]
            evidence[global_id] = float(global_template_scores[global_id])
    return evidence


def _mean_lookup(values, names, name_to_index):
    if not names:
        return 0.0
    selected = [
        float(values[name_to_index[name]]) for name in names if name in name_to_index
    ]
    if not selected:
        return 0.0
    return sum(selected) / len(selected)
