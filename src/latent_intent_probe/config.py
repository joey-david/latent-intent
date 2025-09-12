from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml


@dataclass
class RunConfig:
    name: str = "temporal-latent-intent-qwen15b"
    seed: int = 20260607
    output_dir: str = "results"


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    dtype: str = "auto"
    device_map: str = "auto"
    trust_remote_code: bool = False
    batch_size: int = 2
    max_new_tokens: int = 80
    attn_implementation: str = "eager"
    collect_attentions: bool = False


@dataclass
class DatasetConfig:
    examples_per_label: int = 180
    include_neutral_decoys: bool = True


@dataclass
class ProbeConfig:
    cv_folds: int = 5
    max_iter: int = 2000
    regularization_c: float = 0.25
    transfer_top_k_layers: int = 8


@dataclass
class ReportConfig:
    top_k_layers: int = 8
    top_k_heads: int = 20


@dataclass
class ExperimentConfig:
    run: RunConfig
    model: ModelConfig
    dataset: DatasetConfig
    probe: ProbeConfig
    report: ReportConfig


T = TypeVar("T")


def _from_dict(cls: type[T], data: dict[str, Any] | None) -> T:
    data = data or {}
    valid = {field.name for field in fields(cls)}
    return cls(**{key: value for key, value in data.items() if key in valid})


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    config = ExperimentConfig(
        run=_from_dict(RunConfig, raw.get("run")),
        model=_from_dict(ModelConfig, raw.get("model")),
        dataset=_from_dict(DatasetConfig, raw.get("dataset")),
        probe=_from_dict(ProbeConfig, raw.get("probe")),
        report=_from_dict(ReportConfig, raw.get("report")),
    )

    config.model.name = os.getenv("MODEL_NAME", config.model.name)
    config.run.name = os.getenv("RUN_NAME", config.run.name)
    return config


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    if not is_dataclass(config):
        raise TypeError("config_to_dict expects an ExperimentConfig dataclass")
    return asdict(config)


def write_resolved_config(config: ExperimentConfig, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config_to_dict(config), handle, sort_keys=False)
