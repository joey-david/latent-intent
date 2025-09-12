from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from typing import Any

from latent_intent_probe.config import ExperimentConfig


def allow_cpu_from_env() -> bool:
    value = os.getenv("ALLOW_CPU", "false").strip().lower()
    return value in {"1", "true", "yes", "y"}


def require_cuda(config: ExperimentConfig) -> bool:
    return bool(config.model.require_cuda and not allow_cpu_from_env())


def assert_cuda_runtime(config: ExperimentConfig) -> None:
    if not require_cuda(config):
        return
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available to PyTorch, and model.require_cuda is true. "
            "This run would execute on CPU. Fix the server environment or set "
            "ALLOW_CPU=true only if you intentionally want a CPU run."
        )


def validate_model_placement(model: Any, config: ExperimentConfig) -> None:
    device_map = getattr(model, "hf_device_map", None)
    if not require_cuda(config):
        return

    if device_map:
        bad = {name: device for name, device in device_map.items() if str(device).startswith(("cpu", "disk"))}
        if bad:
            sample = dict(list(bad.items())[:8])
            raise RuntimeError(
                "Transformers placed part of the model on CPU/disk while CUDA is required. "
                f"Sample placements: {sample}. This would be very slow."
            )
        return

    try:
        device = next(model.parameters()).device
    except StopIteration:
        return
    if device.type != "cuda":
        raise RuntimeError(f"Model parameters are on {device}, not CUDA. This would be very slow.")


def build_diagnostics(config: ExperimentConfig, load_model: bool = False) -> dict[str, Any]:
    report: dict[str, Any] = {
        "python": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "env": {
            "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES"),
            "MODEL_NAME": os.getenv("MODEL_NAME"),
            "ALLOW_CPU": os.getenv("ALLOW_CPU"),
        },
        "config": asdict(config),
    }

    report["nvidia_smi"] = _nvidia_smi()

    try:
        import torch

        cuda_devices = []
        if torch.cuda.is_available():
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                cuda_devices.append(
                    {
                        "index": idx,
                        "name": torch.cuda.get_device_name(idx),
                        "capability": list(torch.cuda.get_device_capability(idx)),
                        "total_memory_gb": round(props.total_memory / 1024**3, 2),
                    }
                )
        report["torch"] = {
            "version": torch.__version__,
            "compiled_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_devices": cuda_devices,
        }
    except Exception as exc:  # pragma: no cover - diagnostic path
        report["torch_error"] = repr(exc)

    if load_model:
        try:
            from latent_intent_probe.hf_inference import load_model_and_tokenizer

            model, _tokenizer = load_model_and_tokenizer(config)
            report["model_device_map"] = getattr(model, "hf_device_map", None)
            report["first_parameter_device"] = str(next(model.parameters()).device)
            validate_model_placement(model, config)
        except Exception as exc:  # pragma: no cover - diagnostic path
            report["model_load_error"] = repr(exc)

    return report


def print_diagnostics(config: ExperimentConfig, load_model: bool = False) -> None:
    print(json.dumps(build_diagnostics(config, load_model=load_model), indent=2, sort_keys=True))


def _nvidia_smi() -> dict[str, Any]:
    if not shutil.which("nvidia-smi"):
        return {"available": False}
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,cuda_version,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except Exception as exc:  # pragma: no cover - diagnostic path
        return {"available": True, "error": repr(exc)}

    rows = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 7:
            rows.append(
                {
                    "index": parts[0],
                    "name": parts[1],
                    "driver_version": parts[2],
                    "cuda_version": parts[3],
                    "memory_total_mb": parts[4],
                    "memory_used_mb": parts[5],
                    "utilization_gpu_percent": parts[6],
                }
            )
    return {"available": True, "gpus": rows}
