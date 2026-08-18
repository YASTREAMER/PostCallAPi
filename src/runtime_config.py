
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

MODEL_ID = str(
    Path(__file__).resolve().parent.parent / "model" / "merged_bf16-fp8"
)
# Canonical caller-facing model name always reported in responses, regardless
# of which gateway alias ("krishna-2.5" or "postcall-qwen3-14b-lora") a
# request came in as (the internal HF model id above should never leak out).
DEFAULT_MODEL_NAME = os.environ.get("POSTCALL_DEFAULT_MODEL_NAME", "krishna-2.5")

MAX_MODEL_LEN = int(os.environ.get("VLLM_MAX_MODEL_LEN", "32768"))
MAX_NEW_TOKENS = int(os.environ.get("VLLM_MAX_NEW_TOKENS", "8192"))
MIN_NEW_TOKENS = int(os.environ.get("VLLM_MIN_NEW_TOKENS", "1024"))
TOKENS_PER_FIELD = int(os.environ.get("VLLM_TOKENS_PER_FIELD", "64"))
MAX_GENERATION_RETRIES = int(os.environ.get("VLLM_MAX_GENERATION_RETRIES", "3"))
# Upper bound injected into each field's "comment" JSON-schema property when
# a caller's response_format doesn't already cap it, so a loosely specified
# schema (e.g. missing "required") can't let the model ramble in a comment
# until it exhausts the whole generation budget.
COMMENT_MAX_CHARS = int(os.environ.get("POSTCALL_COMMENT_MAX_CHARS", "400"))
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
if COMMENT_MAX_CHARS <= 0:
    raise ValueError("POSTCALL_COMMENT_MAX_CHARS must be greater than zero")
if MAX_ACTIVE_REQUESTS <= 0:
    raise ValueError("POSTCALL_MAX_ACTIVE_REQUESTS must be greater than zero")
