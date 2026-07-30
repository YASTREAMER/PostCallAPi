
import json
import os
from pathlib import Path

MODEL_ID = "unsloth/Qwen3-14B-unsloth-bnb-4bit"
ADAPTER_DIR = Path(__file__).resolve().parent.parent / "adapter"
ADAPTER_NAME = "postcall-adapter"
ADAPTER_ID = 1
MAX_LORA_RANK = 16

MAX_MODEL_LEN = int(os.environ.get("VLLM_MAX_MODEL_LEN", "16384"))
MAX_NEW_TOKENS = int(os.environ.get("VLLM_MAX_NEW_TOKENS", "8192"))
MIN_NEW_TOKENS = int(os.environ.get("VLLM_MIN_NEW_TOKENS", "1024"))
TOKENS_PER_FIELD = int(os.environ.get("VLLM_TOKENS_PER_FIELD", "64"))
MAX_GENERATION_RETRIES = int(os.environ.get("VLLM_MAX_GENERATION_RETRIES", "1"))
GPU_MEMORY_UTILIZATION = float(
    os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.5")
)
ENABLE_THINKING = os.environ.get(
    "POSTCALL_ENABLE_THINKING", "false"
).strip().casefold() in {"1", "true", "yes", "on"}
ENABLE_PREFIX_CACHING = os.environ.get(
    "VLLM_ENABLE_PREFIX_CACHING", "true"
).strip().casefold() in {"1", "true", "yes", "on"}

API_HOST = os.environ.get("POSTCALL_API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("POSTCALL_API_PORT", "8000"))
JOB_TTL_SECONDS = float(os.environ.get("POSTCALL_JOB_TTL_SECONDS", "3600"))
MAX_COMPLETED_JOBS = int(os.environ.get("POSTCALL_MAX_COMPLETED_JOBS", "10000"))
MAX_ACTIVE_JOBS = int(os.environ.get("POSTCALL_MAX_ACTIVE_JOBS", "100"))

if not 0 < GPU_MEMORY_UTILIZATION <= 1:
    raise ValueError(
        "VLLM_GPU_MEMORY_UTILIZATION must be greater than 0 and at most 1"
    )
if not 1 <= API_PORT <= 65535:
    raise ValueError("POSTCALL_API_PORT must be between 1 and 65535")
if not 0 < MIN_NEW_TOKENS <= MAX_NEW_TOKENS:
    raise ValueError("VLLM token limits must satisfy 0 < MIN <= MAX")
if TOKENS_PER_FIELD <= 0:
    raise ValueError("VLLM_TOKENS_PER_FIELD must be greater than zero")
if MAX_GENERATION_RETRIES < 0:
    raise ValueError("VLLM_MAX_GENERATION_RETRIES cannot be negative")
if JOB_TTL_SECONDS <= 0 or MAX_COMPLETED_JOBS <= 0 or MAX_ACTIVE_JOBS <= 0:
    raise ValueError("Job retention and capacity settings must be greater than zero")


def validate_runtime_paths() -> None:
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
