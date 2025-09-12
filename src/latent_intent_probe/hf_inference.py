from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from latent_intent_probe.config import ModelConfig
from latent_intent_probe.dataset import write_jsonl


def load_model_and_tokenizer(config: ModelConfig):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = os.getenv("HF_TOKEN") or None
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

    model = AutoModelForCausalLM.from_pretrained(
        config.name,
        token=token,
        torch_dtype=torch_dtype,
        device_map=config.device_map,
        trust_remote_code=config.trust_remote_code,
    )
    model.eval()
    return model, tokenizer


def format_chat(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    rendered = []
    for message in messages:
        rendered.append(f"{message['role'].upper()}: {message['content']}")
    rendered.append("ASSISTANT:")
    return "\n\n".join(rendered)


def collect_activations_and_generations(
    records: list[dict],
    model: Any,
    tokenizer: Any,
    config: ModelConfig,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    import torch

    output_dir = Path(output_dir)
    activation_batches: list[np.ndarray] = []
    enriched_records: list[dict] = []
    prompts = [format_chat(tokenizer, record["messages"]) for record in records]

    first_device = next(model.parameters()).device
    iterator = range(0, len(records), config.batch_size)
    for start in tqdm(iterator, desc="inference", unit="batch"):
        batch_records = records[start : start + config.batch_size]
        batch_prompts = prompts[start : start + config.batch_size]
        encoded = tokenizer(batch_prompts, padding=True, return_tensors="pt")
        encoded = {key: value.to(first_device) for key, value in encoded.items()}

        with torch.inference_mode():
            outputs = model(**encoded, output_hidden_states=True, use_cache=False)
            layer_vectors = [
                hidden[:, -1, :].detach().float().cpu().numpy()
                for hidden in outputs.hidden_states[1:]
            ]
            # Shape: batch, layer, hidden_size.
            activation_batches.append(np.stack(layer_vectors, axis=1))

            generated = model.generate(
                **encoded,
                max_new_tokens=config.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_only = generated[:, encoded["input_ids"].shape[1] :]
        responses = tokenizer.batch_decode(generated_only, skip_special_tokens=True)
        for record, prompt, response in zip(batch_records, batch_prompts, responses):
            enriched = dict(record)
            enriched["rendered_prompt"] = prompt
            enriched["response"] = response.strip()
            enriched_records.append(enriched)

    activations = np.concatenate(activation_batches, axis=0).astype(np.float32)
    activations_path = output_dir / "activations_last_prompt_token.npz"
    records_path = output_dir / "records_with_responses.jsonl"
    np.savez_compressed(activations_path, activations=activations)
    write_jsonl(enriched_records, records_path)
    return activations_path, records_path
