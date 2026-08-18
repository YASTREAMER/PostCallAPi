from runtime_config import (
    ENABLE_PREFIX_CACHING,
    ENABLE_THINKING,
    GPU_MEMORY_UTILIZATION,
    MAX_MODEL_LEN,
    MAX_NEW_TOKENS,
    MIN_NEW_TOKENS,
    TOKENS_PER_FIELD,
    MODEL_ID,
)

import inspect
import json
import logging
import re
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

logger = logging.getLogger("model_service")

if "structured_outputs" in inspect.signature(SamplingParams).parameters:
    from vllm.sampling_params import StructuredOutputsParams

    _STRUCTURED_OUTPUTS_PARAMETER = "structured_outputs"
else:
    from vllm.sampling_params import GuidedDecodingParams as StructuredOutputsParams

    _STRUCTURED_OUTPUTS_PARAMETER = "guided_decoding"

_engine: Optional[AsyncLLMEngine] = None
_tokenizer = None


async def load_model() -> None:
    global _engine, _tokenizer
    if _engine is not None:
        return

    logger.info("Starting vLLM AsyncLLMEngine for %s ...", MODEL_ID)
    engine_args = AsyncEngineArgs(
        model=MODEL_ID,
        dtype="bfloat16",
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=MAX_MODEL_LEN,
        enable_prefix_caching=ENABLE_PREFIX_CACHING,
    )
    _engine = AsyncLLMEngine.from_engine_args(engine_args)
    _tokenizer = await _engine.get_tokenizer()
    logger.info("Model loaded.")


def shutdown_model() -> None:
    global _engine, _tokenizer
    engine = _engine
    _engine = None
    _tokenizer = None
    if engine is None:
        return

    shutdown = getattr(engine, "shutdown", None)
    if callable(shutdown):
        shutdown()
        return

    # Compatibility with the legacy V0 async engine.
    shutdown_background_loop = getattr(engine, "shutdown_background_loop", None)
    if callable(shutdown_background_loop):
        shutdown_background_loop()


def _strip_think_block(text: str) -> Tuple[str, Optional[str]]:
    match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    if match:
        thinking = match.group(1).strip()
        remaining = text[match.end() :].strip()
        return remaining, thinking

    open_match = re.search(r"<think>", text)
    if open_match:
        thinking = text[open_match.end() :].strip()
        return "", thinking

    return text.strip(), None


def _extract_json(text: str) -> Dict[str, Any]:
    """Pulls a JSON object out of model output, tolerating stray markdown fences."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in model output")

    return json.loads(cleaned[start : end + 1])


def _build_prompt_text(messages: Sequence[Mapping[str, str]]) -> str:
    return _tokenizer.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=ENABLE_THINKING,
    )


async def _run_generation(
    messages: Sequence[Mapping[str, str]],
    request_id: str,
    requested_max_tokens: int,
    temperature: float,
    json_schema: Optional[Mapping[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    text = _build_prompt_text(messages)
    prompt_token_ids = _tokenizer.encode(text)
    available_tokens = max(1, MAX_MODEL_LEN - len(prompt_token_ids) - 32)
    generation_limit = min(requested_max_tokens, available_tokens)

    structured_output_kwargs = {}
    if json_schema is not None:
        structured_output_kwargs[_STRUCTURED_OUTPUTS_PARAMETER] = (
            StructuredOutputsParams(json=dict(json_schema))
        )
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=generation_limit,
        **structured_output_kwargs,
    )

    started = time.monotonic()
    final_output = None
    async for request_output in _engine.generate(
        text,
        sampling_params,
        request_id,
    ):
        final_output = request_output

    if final_output is None:
        raise RuntimeError("vLLM completed without returning a generation result")

    completion = final_output.outputs[0]
    completion_token_ids = getattr(completion, "token_ids", None) or []
    prompt_tokens = len(prompt_token_ids)
    completion_tokens = len(completion_token_ids)
    usage = {
        "generation_seconds": round(time.monotonic() - started, 3),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "max_tokens": generation_limit,
        "finish_reason": getattr(completion, "finish_reason", None),
    }
    return completion.text, usage


def _normalize_top_level_keys(result: Dict[str, Any]) -> Dict[str, Any]:
    return {(key.strip() if isinstance(key, str) else key): value for key, value in result.items()}


def _validate_commented_result(
    result: dict, expected_fields: Optional[Sequence[str]]
) -> None:
    if expected_fields is None:
        return

    expected = list(expected_fields)
    expected_set = set(expected)
    actual_set = set(result)
    errors = []

    missing = [name for name in expected if name not in result]
    unexpected = sorted(actual_set - expected_set)
    if missing:
        errors.append(f"missing fields: {missing}")
    if unexpected:
        errors.append(f"unexpected top-level keys: {unexpected}")

    for name in expected:
        entry = result.get(name)
        if not isinstance(entry, dict):
            errors.append(f"{name!r} must map to an object")
            continue
        if set(entry) != {"value", "comment"}:
            errors.append(f"{name!r} must contain exactly value and comment")
            continue
        comment = entry["comment"]
        if not isinstance(comment, str) or not comment.strip():
            errors.append(f"{name!r} must contain a non-empty comment")

    if errors:
        raise ValueError("; ".join(errors))


def _corrective_messages(
    messages: Sequence[Mapping[str, str]], expected_fields: Sequence[str]
) -> list[dict[str, str]]:
    shape = {
        name: {
            "value": f"<actual value for {name}>",
            "comment": f"<concise transcript evidence for {name}>",
        }
        for name in expected_fields
    }
    correction = (
        "IMPORTANT CORRECTION: Your previous response had the wrong JSON "
        "shape. Generate a fresh answer. Put each requested field at the top "
        "level, and give every field its own non-empty value and evidence "
        "comment. Do not put value/comment directly at the top level. Use "
        f"exactly this structure: {json.dumps(shape, ensure_ascii=False)}"
    )
    return [dict(message) for message in messages] + [
        {"role": "user", "content": correction}
    ]


async def generate_extraction(
    messages: Sequence[Mapping[str, str]],
    request_id: str,
    max_retries: int = 1,
    expected_fields: Optional[Sequence[str]] = None,
    json_schema: Optional[Mapping[str, Any]] = None,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    last_error = None
    last_raw = None
    attempt_usage = []
    field_count = len(expected_fields or ())
    requested_max_tokens = (
        min(
            MAX_NEW_TOKENS,
            max(MIN_NEW_TOKENS, 512 + TOKENS_PER_FIELD * field_count),
        )
        if field_count
        else MAX_NEW_TOKENS
    )

    for attempt in range(max_retries + 1):
        generation_messages = (
            messages
            if attempt == 0 or expected_fields is None
            else _corrective_messages(messages, expected_fields)
        )
        raw, usage = await _run_generation(
            generation_messages,
            f"{request_id}-attempt{attempt}",
            requested_max_tokens,
            temperature,
            json_schema,
        )
        usage["attempt"] = attempt + 1
        attempt_usage.append(usage)
        last_raw = raw
        answer_text, thinking = _strip_think_block(raw)
        try:
            parsed = _extract_json(answer_text)
            parsed = _normalize_top_level_keys(parsed)
            _validate_commented_result(parsed, expected_fields)
            performance = {
                "attempts": attempt + 1,
                "retried": attempt > 0,
                "generation_seconds": round(
                    sum(item["generation_seconds"] for item in attempt_usage), 3
                ),
                "prompt_tokens": sum(
                    item["prompt_tokens"] for item in attempt_usage
                ),
                "completion_tokens": sum(
                    item["completion_tokens"] for item in attempt_usage
                ),
                "total_tokens": sum(
                    item["total_tokens"] for item in attempt_usage
                ),
                "attempt_details": attempt_usage,
            }
            return {
                "result": parsed,
                "thinking": thinking,
                "performance": performance,
            }
        except (ValueError, json.JSONDecodeError) as e:
            last_error = e
            logger.warning(
                "Attempt %d/%d: failed to parse JSON: %s",
                attempt + 1,
                max_retries + 1,
                e,
            )

    raise ValueError(
        f"Model did not produce valid JSON after {max_retries + 1} attempt(s). "
        f"Last error: {last_error}. Last raw output (truncated): {str(last_raw)[:500]}"
    )
