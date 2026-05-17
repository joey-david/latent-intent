from collections import defaultdict

from latent_intent_probe.config import DatasetConfig
from latent_intent_probe.dataset import build_dataset
from latent_intent_probe.phases import PHASE_NAMES
from latent_intent_probe.run import _balanced_limit


def test_dataset_is_balanced_and_paired() -> None:
    records = build_dataset(DatasetConfig(examples_per_label=24), seed=1)
    labels = [record.label for record in records]

    assert len(records) == 48
    assert labels.count(0) == 24
    assert labels.count(1) == 24

    by_pair = defaultdict(list)
    for record in records:
        by_pair[record.pair_id].append(record)

    assert len(by_pair) == 24
    for pair in by_pair.values():
        assert {record.label for record in pair} == {0, 1}
        assert len({record.objective_a for record in pair}) == 1
        assert len({record.objective_b for record in pair}) == 1
        assert len({record.harmful_slot for record in pair}) == 1
        assert len({record.fixed_response for record in pair}) == 1


def test_counterfactual_pair_differs_only_at_selector() -> None:
    records = build_dataset(DatasetConfig(examples_per_label=1), seed=2)
    assert len(records) == 2

    left, right = records
    left_prompt = left.messages[1]["content"]
    right_prompt = right.messages[1]["content"]
    normalized_left = left_prompt.replace(
        f"ACTIVE OBJECTIVE: {left.active_slot}",
        "ACTIVE OBJECTIVE: X",
    )
    normalized_right = right_prompt.replace(
        f"ACTIVE OBJECTIVE: {right.active_slot}",
        "ACTIVE OBJECTIVE: X",
    )

    assert normalized_left == normalized_right
    assert left.active_slot != right.active_slot
    assert left.label != right.label


def test_records_have_selected_and_inactive_objectives() -> None:
    record = build_dataset(DatasetConfig(examples_per_label=1), seed=3)[0]

    assert record.messages[0]["role"] == "system"
    assert record.messages[1]["role"] == "user"
    assert record.private_objective
    assert record.inactive_objective
    assert record.private_objective != record.inactive_objective


def test_selector_centered_phase_names_are_ordered() -> None:
    assert PHASE_NAMES == [
        "pre_selector",
        "post_selector",
        "public_task_span",
        "final_prompt",
        "response_first",
        "response_mean",
    ]


def test_balanced_limit_preserves_complete_pairs() -> None:
    records = [record.__dict__ for record in build_dataset(DatasetConfig(examples_per_label=30), seed=4)]
    limited = _balanced_limit(records, 24)
    labels = [record["label"] for record in limited]
    pair_ids = {record["pair_id"] for record in limited}

    assert len(limited) == 24
    assert labels.count(0) == 12
    assert labels.count(1) == 12
    assert len(pair_ids) == 12
