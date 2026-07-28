"""Shared vLLM runtime configuration for live and offline inference."""

import json
import os
from pathlib import Path

# src/outputs/adapter; the adapter now lives directly at <API_DIR>/adapter.
API_DIR = Path(
    os.environ.get("POSTCALL_API_DIR", "PostCallAPi")
).resolve()
 
MODEL_ID = "unsloth/Qwen3-14B-unsloth-bnb-4bit"
ADAPTER_DIR = API_DIR / "adapter"
ADAPTER_NAME = "postcall-adapter"
ADAPTER_ID = 1
MAX_LORA_RANK = 16

MAX_MODEL_LEN = 16384
MAX_NEW_TOKENS = 16384
GPU_MEMORY_UTILIZATION = float(
    os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.5")
)
ENABLE_THINKING = os.environ.get(
    "POSTCALL_ENABLE_THINKING", "true"
).strip().casefold() in {"1", "true", "yes", "on"}

API_HOST = os.environ.get("POSTCALL_API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("POSTCALL_API_PORT", "8000"))

if not 0 < GPU_MEMORY_UTILIZATION <= 1:
    raise ValueError(
        "VLLM_GPU_MEMORY_UTILIZATION must be greater than 0 and at most 1"
    )
if not 1 <= API_PORT <= 65535:
    raise ValueError("POSTCALL_API_PORT must be between 1 and 65535")


def validate_runtime_paths() -> None:
    """Fail at startup if the adapter is missing or belongs to another base."""
    required_adapter_files = (
        ADAPTER_DIR / "adapter_config.json",
        ADAPTER_DIR / "adapter_model.safetensors",
    )
    missing = [str(path) for path in required_adapter_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "LoRA adapter is incomplete; missing: " + ", ".join(missing)
        )

    adapter_config = json.loads(
        (ADAPTER_DIR / "adapter_config.json").read_text(encoding="utf-8")
    )
    adapter_base = adapter_config.get("base_model_name_or_path")
    if adapter_base != MODEL_ID:
        raise ValueError(
            f"Adapter expects base model {adapter_base!r}, but MODEL_ID is {MODEL_ID!r}"
        )

    adapter_rank = adapter_config.get("r")
    if adapter_rank != MAX_LORA_RANK:
        raise ValueError(
            f"Adapter rank is {adapter_rank!r}, but MAX_LORA_RANK is "
            f"{MAX_LORA_RANK!r}"
        )
