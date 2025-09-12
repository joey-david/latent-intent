from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from latent_intent_probe.config import ModelConfig
from latent_intent_probe.dataset import write_jsonl
from latent_intent_probe.phases import PHASE_NAMES

ATTENTION_COLUMNS = [
    "id",
    "label",
    "label_name",
    "template_id",
    "scenario_id",
    "query_phase",
    "layer",
    "head",
    "objective_token_count",
    "public_task_token_count",
    "attention_mass_to_objective",
    "attention_mass_to_public_task",
    "attention_density_to_objective",
    "attention_density_to_public_task",
    "goal_anchor_asymmetry",
    "goal_anchor_log_ratio",
]


def load_model_and_tokenizer(config: ModelConfig):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = _hf_token()
    tokenizer = AutoTokenizer.from_pretrained(
        config.name,
        token=token,
        trust_remote_code=config.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    torch_dtype: str | torch.dtype
    if config.dtype == "auto":
        torch_dtype = "auto"
    else:
        torch_dtype = getattr(torch, config.dtype)

    model_kwargs: dict[str, Any] = {
        "token": token,
        "torch_dtype": torch_dtype,
        "device_map": config.device_map,
        "trust_remote_code": config.trust_remote_code,
    }
    if config.attn_implementation:
        model_kwargs["attn_implementation"] = config.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(config.name, **model_kwargs)
    model.eval()
    return model, tokenizer


def _hf_token() -> str | None:
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_HUB_TOKEN", "hf_token", "huggingface_token"):
        token = os.getenv(name)
        if token:
            return token
    return None


def format_chat(tokenizer: Any, messages: list[dict[str, str]], config: ModelConfig | None = None) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if config is not None and config.enable_thinking is not None:
            kwargs["enable_thinking"] = config.enable_thinking
        try:
            return tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            return tokenizer.apply_chat_template(messages, **kwargs)
    rendered = []
    for message in messages:
        rendered.append(f"{message['role'].upper()}: {message['content']}")
    rendered.append("ASSISTANT:")
    return "\n\n".join(rendered)


def collect_phase_readouts_and_generations(
    records: list[dict],
    model: Any,
    tokenizer: Any,
    config: ModelConfig,
    output_dir: str | Path,
) -> tuple[Path, Path, Path]:
    import torch

    output_dir = Path(output_dir)
    phase_activation_batches: list[np.ndarray] = []
    attention_rows: list[dict] = []
    enriched_records: list[dict] = []
    prompts = [format_chat(tokenizer, record["messages"], config) for record in records]

    first_device = next(model.parameters()).device
    iterator = range(0, len(records), config.batch_size)
    for start in tqdm(iterator, desc="inference", unit="batch"):
        batch_records = records[start : start + config.batch_size]
        batch_prompts = prompts[start : start + config.batch_size]
        prompt_encoded = tokenizer(
            batch_prompts,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        prompt_encoded = {key: value.to(first_device) for key, value in prompt_encoded.items()}

        with torch.inference_mode():
            generated = model.generate(
                **prompt_encoded,
                max_new_tokens=config.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_only = generated[:, prompt_encoded["input_ids"].shape[1] :]
        responses = tokenizer.batch_decode(generated_only, skip_special_tokens=True)
        full_texts = [prompt + response for prompt, response in zip(batch_prompts, responses)]

        full_encoded = _encode_with_offsets(tokenizer, full_texts)
        offsets = full_encoded.pop("offset_mapping")
        full_encoded = {key: value.to(first_device) for key, value in full_encoded.items()}

        with torch.inference_mode():
            outputs = model(
                **full_encoded,
                output_hidden_states=True,
                output_attentions=config.collect_attentions,
                use_cache=False,
            )

        batch_phase_vectors = _extract_phase_activations(
            outputs.hidden_states,
            full_encoded["attention_mask"],
            offsets,
            batch_records,
            batch_prompts,
            responses,
        )
        phase_activation_batches.append(batch_phase_vectors)

        if config.collect_attentions and outputs.attentions:
            attention_rows.extend(
                _extract_attention_rows(
                    outputs.attentions,
                    full_encoded["attention_mask"],
                    offsets,
                    batch_records,
                    batch_prompts,
                    responses,
                )
            )

        for record, prompt, response, row_offsets, attention_mask in zip(
            batch_records,
            batch_prompts,
            responses,
            offsets,
            full_encoded["attention_mask"].detach().cpu().numpy(),
        ):
            phase_positions = _phase_positions(row_offsets, attention_mask, record, prompt, response)
            enriched = dict(record)
            enriched["rendered_prompt"] = prompt
            enriched["response"] = response.strip()
            enriched["phase_token_counts"] = {phase: len(tokens) for phase, tokens in phase_positions.items()}
            enriched["objective_char_span"] = _char_span(prompt, record["private_objective"])
            enriched["public_task_char_span"] = _char_span(prompt, record["cover_task"])
            enriched["response_char_span"] = [len(prompt), len(prompt) + len(response)]
            enriched_records.append(enriched)

    activations = np.concatenate(phase_activation_batches, axis=0).astype(np.float32)
    activations_path = output_dir / "activations_by_phase.npz"
    records_path = output_dir / "records_with_responses.jsonl"
    attention_path = output_dir / "attention_asymmetry_rows.csv"

    np.savez_compressed(activations_path, activations=activations, phase_names=np.array(PHASE_NAMES))
    write_jsonl(enriched_records, records_path)
    pd.DataFrame(attention_rows, columns=ATTENTION_COLUMNS).to_csv(attention_path, index=False)
    return activations_path, records_path, attention_path


def _encode_with_offsets(tokenizer: Any, texts: list[str]) -> dict[str, Any]:
    try:
        return tokenizer(
            texts,
            padding=True,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
    except NotImplementedError as exc:
        raise RuntimeError(
            "This experiment needs a fast tokenizer that supports offset mappings "
            "so objective/task/response phases can be localized."
        ) from exc


def _extract_phase_activations(
    hidden_states: tuple[Any, ...],
    attention_mask: Any,
    offsets: Any,
    records: list[dict],
    prompts: list[str],
    responses: list[str],
) -> np.ndarray:
    mask_np = attention_mask.detach().cpu().numpy()
    layer_outputs = hidden_states[1:]
    batch_vectors: list[np.ndarray] = []

    for row_idx, (record, prompt, response) in enumerate(zip(records, prompts, responses)):
        positions = _phase_positions(offsets[row_idx], mask_np[row_idx], record, prompt, response)
        layer_vectors = []
        for hidden in layer_outputs:
            hidden_row = hidden[row_idx].detach().float().cpu().numpy()
            phase_vectors = []
            for phase in PHASE_NAMES:
                token_positions = positions[phase] or positions["final_prompt"]
                phase_vectors.append(hidden_row[token_positions].mean(axis=0))
            layer_vectors.append(np.stack(phase_vectors, axis=0))
        # Shape per example: phase, layer, hidden_size.
        batch_vectors.append(np.stack(layer_vectors, axis=1))
    return np.stack(batch_vectors, axis=0)


def _extract_attention_rows(
    attentions: tuple[Any, ...],
    attention_mask: Any,
    offsets: Any,
    records: list[dict],
    prompts: list[str],
    responses: list[str],
) -> list[dict]:
    mask_np = attention_mask.detach().cpu().numpy()
    rows: list[dict] = []

    for row_idx, (record, prompt, response) in enumerate(zip(records, prompts, responses)):
        positions = _phase_positions(offsets[row_idx], mask_np[row_idx], record, prompt, response)
        objective_positions = positions["objective_span"] or positions["post_objective"]
        task_positions = positions["public_task_span"] or positions["final_prompt"]
        if not objective_positions or not task_positions:
            continue

        for layer, attention in enumerate(attentions):
            attention_row = attention[row_idx].detach().float().cpu().numpy()
            for phase in PHASE_NAMES:
                query_positions = positions[phase] or positions["final_prompt"]
                objective_mass = attention_row[:, query_positions][:, :, objective_positions].sum(axis=2).mean(axis=1)
                task_mass = attention_row[:, query_positions][:, :, task_positions].sum(axis=2).mean(axis=1)
                objective_density = objective_mass / len(objective_positions)
                task_density = task_mass / len(task_positions)
                for head, (obj_mass, pub_mass, obj_density, pub_density) in enumerate(
                    zip(objective_mass, task_mass, objective_density, task_density)
                ):
                    rows.append(
                        {
                            "id": record["id"],
                            "label": record["label"],
                            "label_name": record["label_name"],
                            "template_id": record["template_id"],
                            "scenario_id": record["scenario_id"],
                            "query_phase": phase,
                            "layer": layer,
                            "head": head,
                            "objective_token_count": len(objective_positions),
                            "public_task_token_count": len(task_positions),
                            "attention_mass_to_objective": float(obj_mass),
                            "attention_mass_to_public_task": float(pub_mass),
                            "attention_density_to_objective": float(obj_density),
                            "attention_density_to_public_task": float(pub_density),
                            "goal_anchor_asymmetry": float(obj_density - pub_density),
                            "goal_anchor_log_ratio": float(math.log((obj_density + 1e-8) / (pub_density + 1e-8))),
                        }
                    )
    return rows


def _phase_positions(
    offsets: Any,
    attention_mask: Any,
    record: dict,
    prompt: str,
    response: str,
) -> dict[str, list[int]]:
    offsets_np = _as_numpy(offsets)
    mask_np = np.asarray(attention_mask)
    objective_span = _char_span(prompt, record["private_objective"])
    task_span = _char_span(prompt, record["cover_task"])
    response_span = (len(prompt), len(prompt) + len(response))
    real_tokens = [idx for idx, keep in enumerate(mask_np.tolist()) if keep]
    prompt_tokens = _tokens_overlapping(offsets_np, mask_np, (0, len(prompt)))
    final_prompt = [prompt_tokens[-1] if prompt_tokens else real_tokens[-1]]

    objective_tokens = _tokens_overlapping(offsets_np, mask_np, objective_span)
    task_tokens = _tokens_overlapping(offsets_np, mask_np, task_span)
    response_tokens = _tokens_overlapping(offsets_np, mask_np, response_span)

    return {
        "objective_span": objective_tokens,
        "post_objective": [objective_tokens[-1]] if objective_tokens else final_prompt,
        "public_task_span": task_tokens,
        "final_prompt": final_prompt,
        "response_first": [response_tokens[0]] if response_tokens else final_prompt,
        "response_mean": response_tokens if response_tokens else final_prompt,
    }


def _tokens_overlapping(offsets: np.ndarray, attention_mask: np.ndarray, span: tuple[int, int]) -> list[int]:
    start, end = span
    tokens = []
    for idx, ((tok_start, tok_end), keep) in enumerate(zip(offsets.tolist(), attention_mask.tolist())):
        if not keep or tok_end <= tok_start:
            continue
        if tok_start < end and tok_end > start:
            tokens.append(idx)
    return tokens


def _char_span(text: str, needle: str) -> tuple[int, int]:
    start = text.find(needle)
    if start < 0:
        return (max(0, len(text) - 1), len(text))
    return (start, start + len(needle))


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)
