
import json
import os
import subprocess
from pathlib import Path


def _pick_free_gpu() -> str:
    """Return the index of the GPU with the least memory used."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        gpus = [
            line.split(", ")
            for line in result.stdout.strip().splitlines()
            if line.strip()
        ]
        best_index, _ = min(gpus, key=lambda gpu: int(gpu[1]))
        return best_index
    except Exception:
        return "0"


# Select which GPU to run on before vLLM/torch initialize CUDA. Set
# POSTCALL_GPU_DEVICE to pin a specific device (e.g. "1"); otherwise the
# GPU with the least memory currently in use is picked automatically.
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get(
        "POSTCALL_GPU_DEVICE", ""
    ).strip() or _pick_free_gpu()

MODEL_ID = "unsloth/Qwen3-14B-unsloth-bnb-4bit"
ADAPTER_DIR = Path(__file__).resolve().parent.parent / "adapter"
ADAPTER_NAME = "postcall-adapter"
ADAPTER_ID = 1
MAX_LORA_RANK = 16

MAX_MODEL_LEN = int(os.environ.get("VLLM_MAX_MODEL_LEN", "32768"))
MAX_NEW_TOKENS = int(os.environ.get("VLLM_MAX_NEW_TOKENS", "8192"))
MIN_NEW_TOKENS = int(os.environ.get("VLLM_MIN_NEW_TOKENS", "1024"))
TOKENS_PER_FIELD = int(os.environ.get("VLLM_TOKENS_PER_FIELD", "64"))
MAX_GENERATION_RETRIES = int(os.environ.get("VLLM_MAX_GENERATION_RETRIES", "3"))
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
API_PORT = int(os.environ.get("POSTCALL_API_PORT", "8808"))
API_PREFIX = os.environ.get("POSTCALL_API_PREFIX", "/postcall").strip()
MAX_ACTIVE_REQUESTS = int(
    os.environ.get(
        "POSTCALL_MAX_ACTIVE_REQUESTS",
        os.environ.get("POSTCALL_MAX_ACTIVE_JOBS", "100"),
    )
)

if not 0 < GPU_MEMORY_UTILIZATION <= 1:
    raise ValueError(
        "VLLM_GPU_MEMORY_UTILIZATION must be greater than 0 and at most 1"
    )
if not 1 <= API_PORT <= 65535:
    raise ValueError("POSTCALL_API_PORT must be between 1 and 65535")
if (
    not API_PREFIX.startswith("/")
    or API_PREFIX == "/"
    or API_PREFIX.endswith("/")
):
    raise ValueError(
        "POSTCALL_API_PREFIX must start with / and must not end with /"
    )
if not 0 < MIN_NEW_TOKENS <= MAX_NEW_TOKENS:
    raise ValueError("VLLM token limits must satisfy 0 < MIN <= MAX")
if TOKENS_PER_FIELD <= 0:
    raise ValueError("VLLM_TOKENS_PER_FIELD must be greater than zero")
if MAX_GENERATION_RETRIES < 0:
    raise ValueError("VLLM_MAX_GENERATION_RETRIES cannot be negative")
if MAX_ACTIVE_REQUESTS <= 0:
    raise ValueError("POSTCALL_MAX_ACTIVE_REQUESTS must be greater than zero")


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
