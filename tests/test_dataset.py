from latent_intent_probe.config import DatasetConfig
from latent_intent_probe.dataset import build_dataset
from latent_intent_probe.phases import PHASE_NAMES
from latent_intent_probe.run import _balanced_limit


def test_dataset_is_balanced_for_binary_labels() -> None:
    records = build_dataset(DatasetConfig(examples_per_label=24, include_neutral_decoys=True), seed=1)
    labels = [record.label for record in records]

    assert labels.count(0) == 24
    assert labels.count(1) == 24
    assert labels.count(2) > 0
    assert len({record.template_id for record in records if record.label in (0, 1)}) > 2


def test_records_have_chat_messages() -> None:
    record = build_dataset(DatasetConfig(examples_per_label=1, include_neutral_decoys=False), seed=1)[0]

    assert record.messages[0]["role"] == "system"
    assert record.messages[1]["role"] == "user"
    assert record.private_objective


def test_temporal_phase_names_are_ordered() -> None:
    assert PHASE_NAMES == [
        "objective_span",
        "post_objective",
        "public_task_span",
        "final_prompt",
        "response_first",
        "response_mean",
    ]


def test_balanced_limit_preserves_binary_labels() -> None:
    records = [record.__dict__ for record in build_dataset(DatasetConfig(examples_per_label=30), seed=2)]
    limited = _balanced_limit(records, 24)
    labels = [record["label"] for record in limited]

    assert len(limited) == 24
    assert labels.count(0) == 12
    assert labels.count(1) == 12
