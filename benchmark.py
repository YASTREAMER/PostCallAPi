#!/usr/bin/env python3
"""Benchmark one remote PostCall API as an external HTTP client."""

import argparse
import csv
import difflib
import json
import math
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = PROJECT_DIR / "data" / "Data_with_outcome_fields.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "output"
CONCURRENCY_LEVELS = (20, 25 ,30, 35,40,45,50, 75,100, 125,150,)
MAX_ROWS = 10_000
DEFAULT_TEMPERATURE = 0.8
DEFAULT_REQUEST_TIMEOUT = 1200.0

CONVERSION_STATUS_FIELD = "conversion_status"
DISPOSITION_REASON_FIELD = "disposition_reason"
DEFAULT_CONVERSION_DESCRIPTION = (
    "Set value to true only if the call satisfies the conversion criteria; "
    "otherwise set value to false. Add a concise evidence comment."
)
DISPOSITION_DESCRIPTION = (
    "State the final call outcome in value and provide a concise evidence "
    "comment explaining why the user did or did not convert."
)


class APIRequestError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        category: str,
        http_status: int | None = None,
        retry_after: str | None = None,
        response_body: str = "",
    ) -> None:
        super().__init__(message)
        self.category = category
        self.http_status = http_status
        self.retry_after = retry_after
        self.response_body = response_body


def _bounded_rows(value: str) -> int:
    parsed = int(value)
    if not 50 <= parsed <= MAX_ROWS:
        raise argparse.ArgumentTypeError(
            f"rows must be between 50 and {MAX_ROWS}; at least 50 rows are "
            "required to exercise concurrency 50"
        )
    return parsed


def _positive_float(name: str):
    def parse(value: str) -> float:
        parsed = float(value)
        if parsed <= 0:
            raise argparse.ArgumentTypeError(f"{name} must be greater than zero")
        return parsed

    return parse


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark one remote PostCall API independently at each configured "
            "concurrency level. Results are saved after every completed request "
            "and summarized before the next level begins."
        )
    )
    parser.add_argument(
        "--api-url",
        required=True,
        help=(
            "Base API URL, including its prefix, for example "
            "http://101.53.137.25:8808/postcall"
        ),
    )
    parser.add_argument(
        "--rows",
        required=True,
        type=_bounded_rows,
        help="Number of identical dataset rows tested at every concurrency level.",
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--request-timeout",
        type=_positive_float("request timeout"),
        default=DEFAULT_REQUEST_TIMEOUT,
        help="Maximum seconds for each extraction request (default: 1200).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature sent to the API (default: 0.8).",
    )
    parser.add_argument(
        "--selection",
        choices=("random", "first"),
        default="random",
        help="Select a seeded random sample or the first N rows.",
    )
    parser.add_argument(
        "--random-state",
        "--seed",
        dest="random_state",
        type=int,
        default=42,
        help=(
            "Seed for reproducible random row selection. --seed remains an alias "
            "(default: 42)."
        ),
    )
    return parser.parse_args()


def _normalize_api_url(value: str) -> str:
    url = value.strip().rstrip("/")
    if url.endswith("/extract"):
        url = url[: -len("/extract")]
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(
            "--api-url must be an absolute HTTP(S) URL such as "
            "http://server:8808/postcall"
        )
    if parsed.query or parsed.fragment:
        raise ValueError("--api-url must not contain a query string or fragment")
    return url


def _read_env_value(path: Path, variable: str) -> str | None:
    if not path.is_file():
        return None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        name, separator, raw_value = line.partition("=")
        if not separator or name.strip() != variable:
            continue
        value = raw_value.strip()
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {"'", '"'}
        ):
            value = value[1:-1]
        return value
    return None


def _load_api_key() -> str:
    api_key = os.environ.get("POSTCALL_API_KEY")
    if api_key is None:
        candidates = [Path.cwd() / ".env", PROJECT_DIR / ".env"]
        for candidate in candidates:
            api_key = _read_env_value(candidate, "POSTCALL_API_KEY")
            if api_key is not None:
                break
    if api_key is None or not api_key.strip():
        raise RuntimeError(
            "POSTCALL_API_KEY is required in the environment or a local .env file"
        )
    return api_key.strip()


def _json_cell(row: dict[str, str], column: str, default: Any) -> Any:
    raw = row.get(column, "")
    if not raw:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in CSV column {column!r}") from exc


def _add_outcome_fields(
    schema: list[dict[str, Any]], conversion_reason: str
) -> list[dict[str, Any]]:
    result = []
    for field in schema:
        if not isinstance(field, dict):
            raise ValueError("every postcall field must be an object")
        normalized_field = dict(field)
        normalized_field["name"] = str(field.get("name", "")).strip()
        if not normalized_field["name"]:
            raise ValueError("postcall field name must not be empty")
        result.append(normalized_field)
    names = {
        str(field.get("name", "")).strip()
        for field in result
        if isinstance(field, dict)
    }
    if CONVERSION_STATUS_FIELD not in names:
        rule = (conversion_reason or "").strip() or DEFAULT_CONVERSION_DESCRIPTION
        result.append(
            {
                "name": CONVERSION_STATUS_FIELD,
                "type": "boolean",
                "description": (
                    "Set value to the JSON boolean true or false and put the "
                    f"supporting evidence in comment. Conversion rule: {rule}"
                ),
                "defaultValue": False,
            }
        )
    if DISPOSITION_REASON_FIELD not in names:
        result.append(
            {
                "name": DISPOSITION_REASON_FIELD,
                "type": "text",
                "description": DISPOSITION_DESCRIPTION,
                "defaultValue": "",
            }
        )
    return result


def _normalize_openai_type(value: Any) -> str:
    normalized = str(value or "string").strip().casefold()
    return {
        "text": "string",
        "string": "string",
        "str": "string",
        "boolean": "boolean",
        "bool": "boolean",
        "integer": "integer",
        "int": "integer",
        "number": "number",
        "float": "number",
        "double": "number",
        "array": "array",
        "object": "object",
    }.get(normalized, "string")


def _build_response_schema(
    fields: list[dict[str, Any]],
    schema_name: str = "post_call_analysis",
) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    for field in fields:
        name = str(field.get("name", "")).strip()
        if not name:
            raise ValueError("postcall field name must not be empty")
        value_type = _normalize_openai_type(field.get("type", "string"))
        value_schema: dict[str, Any] = {"type": value_type}
        if value_type == "array":
            value_schema["items"] = field.get("items") or {"type": "string"}
        elif value_type == "object":
            value_schema["properties"] = field.get("properties") or {}
            value_schema["required"] = field.get("required") or list(
                value_schema["properties"]
            )
            value_schema["additionalProperties"] = False

        field_schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "value": value_schema,
                "comment": {"type": "string"},
            },
            "required": ["value", "comment"],
            "additionalProperties": False,
        }
        description = field.get("description")
        if description:
            field_schema["description"] = str(description)
        properties[name] = field_schema
        required.append(name)

    return {
        "name": schema_name,
        "strict": True,
        "schema": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }


def _compact_json(value: Any) -> str:
    """Match JavaScript JSON.stringify output as closely as practical."""
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _build_messages(
    row: dict[str, str],
    schema: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Build the same system/user message structure used by postcall.js."""
    functions_called = _json_cell(row, "functions_called", [])
    call_metadata = _json_cell(row, "call_metadata", {})
    conversion_reason = (row.get("conversion_reason") or "").strip()
    current_time = datetime.now(ZoneInfo("Asia/Kolkata")).strftime(
        "%Y-%m-%dT%H:%M:%S"
    )

    system_prompt = f"""
<task>
current_time: {current_time},
Given a call transcription and function called(name, parameters, success, timestamp), extract the
specified details and output a single JSON object. Every requested variable name must be a top-level key whose value is an object containing exactly "value" and "comment".
- DONT change the variable and key names even if it is wrong.
- If a string variable is not present, assign it an empty string: "".
- If a boolean variable is not present, assign it the value false.
Return only the JSON object—no extra text, explanation, or markdown.
</task>
<details>
    If in the transcript we have only one agent initial message and no other user message,
    then put desposition reason as "user cut the call after hearing the agent's first message".
    {conversion_reason}
</details>
<variables_to_extract>
{_compact_json(schema)}
</variables_to_extract>
"""

    user_content = f"""
<previous_calls_dispositions></previous_calls_dispositions>

<call_duration>{row.get("call_duration", "")}</call_duration>

<hangup_reason>{row.get("hangup_reason", "")}</hangup_reason>

<transcription>{row.get("transcription", "")}</transcription>

<functions_called>{_compact_json(functions_called)}</functions_called>
<call_metadata>This is the metadata given to the agent before the call. Some parts may have been updated during the call. Always trust the transcription first; only use this metadata for information not found in the transcript.
{_compact_json(call_metadata)}</call_metadata>
"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def _build_payload(
    row: dict[str, str],
    temperature: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw_schema = _json_cell(row, "postcall", [])
    if not isinstance(raw_schema, list):
        raise ValueError("CSV postcall column must contain a JSON array")
    schema = _add_outcome_fields(raw_schema, row.get("conversion_reason", ""))
    payload = {
        "messages": _build_messages(row, schema),
        "temperature": temperature,
        "response_format": {
            "type": "json_schema",
            "json_schema": _build_response_schema(schema),
        },
        "include_performance": True,
    }
    return payload, schema


def _request_json(
    method: str,
    url: str,
    *,
    api_key: str,
    timeout: float,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body = (
        json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if payload is not None
        else None
    )
    request = Request(
        url,
        data=body,
        method=method,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            response_text = response.read().decode("utf-8")
            try:
                decoded = json.loads(response_text)
            except json.JSONDecodeError as exc:
                raise APIRequestError(
                    f"{method} {url} returned invalid JSON",
                    category="invalid_json",
                    http_status=response.status,
                    response_body=response_text[:2000],
                ) from exc
    except HTTPError as exc:
        response_body = exc.read().decode("utf-8", errors="replace")
        raise APIRequestError(
            f"{method} {url} returned HTTP {exc.code}",
            category="http_error",
            http_status=exc.code,
            retry_after=exc.headers.get("Retry-After"),
            response_body=response_body[:5000],
        ) from exc
    except TimeoutError as exc:
        raise APIRequestError(
            f"{method} {url} timed out after {timeout} seconds",
            category="timeout",
        ) from exc
    except URLError as exc:
        reason = exc.reason
        category = "timeout" if isinstance(reason, TimeoutError) else "network_error"
        raise APIRequestError(
            f"Cannot reach API at {url}: {reason}",
            category=category,
        ) from exc

    if not isinstance(decoded, dict):
        raise APIRequestError(
            f"{method} {url} returned a non-object JSON response",
            category="invalid_response",
        )
    return decoded


def _value_matches_json_type(value: Any, schema_type: str) -> bool:
    if schema_type == "boolean":
        return isinstance(value, bool)
    if schema_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if schema_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if schema_type == "string":
        return isinstance(value, str)
    if schema_type == "array":
        return isinstance(value, list)
    if schema_type == "object":
        return isinstance(value, dict)
    return True


def _validate_result(
    result: dict[str, Any],
    schema: list[dict[str, Any]],
) -> list[str]:
    expected = [str(field.get("name", "")).strip() for field in schema]
    errors = []
    missing = [name for name in expected if name not in result]
    unexpected = sorted(set(result) - set(expected))
    if missing:
        errors.append(f"missing fields: {missing}")
    if unexpected:
        errors.append(f"unexpected fields: {unexpected}")

    field_by_name = {
        str(field.get("name", "")).strip(): field for field in schema
    }
    for name in expected:
        entry = result.get(name)
        if not isinstance(entry, dict):
            errors.append(f"{name}: expected an object")
            continue
        if set(entry) != {"value", "comment"}:
            errors.append(f"{name}: expected exactly value and comment")
            continue
        if not isinstance(entry["comment"], str) or not entry["comment"].strip():
            errors.append(f"{name}: comment must be a non-empty string")
        expected_type = _normalize_openai_type(
            field_by_name[name].get("type", "string")
        )
        if not _value_matches_json_type(entry["value"], expected_type):
            errors.append(f"{name}: value does not match {expected_type}")
    return errors


def _extract_value(container: dict[str, Any], name: str) -> Any:
    entry = container.get(name, "<missing>")
    return entry.get("value", "<missing>") if isinstance(entry, dict) else entry


def _normalize_comparison(value: Any) -> str:
    if value is None or value == "<missing>":
        return ""
    text = str(value).strip()
    while len(text) >= 2 and text[0] == text[-1] == '"':
        text = text[1:-1].strip()
    text = re.sub(
        r"(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})",
        r"\1 \2",
        text,
    )
    return text.casefold()


_EMPTY_LABELS = {
    "",
    "false",
    "none",
    "null",
    "n/a",
    "na",
    "nil",
    "not applicable",
    "not available",
    "not provided",
    "not specified",
    "unavailable",
}
TEXT_MATCH_THRESHOLD = 0.6


def _is_meaningful_truth(value: Any) -> bool:
    if value is None or value is False:
        return False
    return _normalize_comparison(value) not in _EMPTY_LABELS


def _score_text(actual: str, expected: str) -> float:
    if actual == expected:
        return 1.0
    if not actual or not expected:
        return 0.0
    character_score = difflib.SequenceMatcher(None, actual, expected).ratio()
    actual_words = set(actual.split())
    expected_words = set(expected.split())
    union = actual_words | expected_words
    token_score = len(actual_words & expected_words) / len(union) if union else 1.0
    return max(character_score, token_score)


def _score_result(
    result: dict[str, Any],
    ground_truth: dict[str, Any],
    schema: list[dict[str, Any]],
) -> dict[str, Any]:
    fields: dict[str, dict[str, Any]] = {}
    for field in schema:
        name = str(field.get("name", "")).strip()
        if name not in ground_truth:
            continue

        schema_type = _normalize_openai_type(field.get("type", "string"))
        model_value = _extract_value(result, name)
        truth_value = _extract_value(ground_truth, name)
        actual = _normalize_comparison(model_value)
        expected = _normalize_comparison(truth_value)

        category = (
            "categorical"
            if schema_type == "boolean"
            or isinstance(truth_value, bool)
            or (schema_type != "string" and len(expected.split()) <= 4)
            else "text"
        )
        if category == "categorical":
            if actual in _EMPTY_LABELS:
                actual = "false"
            if expected in _EMPTY_LABELS:
                expected = "false"
            score = 1.0 if actual == expected else 0.0
            match = score == 1.0
        else:
            score = _score_text(actual, expected)
            match = score >= TEXT_MATCH_THRESHOLD

        entry = result.get(name)
        raw_value = entry.get("value") if isinstance(entry, dict) else entry
        type_valid = _value_matches_json_type(raw_value, schema_type)
        meaningful = _is_meaningful_truth(truth_value)
        fields[name] = {
            "category": category,
            "model_value": model_value,
            "truth_value": truth_value,
            "score": round(score, 3),
            "match": match,
            "schema_type": schema_type,
            "type_valid": type_valid,
            "strict_match": match and type_valid,
            "meaningful_truth": meaningful,
        }

    values = list(fields.values())
    scores = [float(field["score"]) for field in values]
    categorical_scores = [
        float(field["score"])
        for field in values
        if field["category"] == "categorical"
    ]
    text_scores = [
        float(field["score"]) for field in values if field["category"] == "text"
    ]
    strict_matches = [bool(field["strict_match"]) for field in values]
    meaningful_fields = [field for field in values if field["meaningful_truth"]]
    requested_fields = len(schema)
    scored_fields = len(values)

    return {
        "fields": fields,
        "overall_score": _mean(scores),
        "categorical_score": _mean(categorical_scores),
        "text_score": _mean(text_scores),
        "strict_match_rate": (
            round(sum(strict_matches) / len(strict_matches), 3)
            if strict_matches
            else None
        ),
        "meaningful_match_rate": (
            round(
                sum(bool(field["match"]) for field in meaningful_fields)
                / len(meaningful_fields),
                3,
            )
            if meaningful_fields
            else None
        ),
        "meaningful_mean_score": (
            _mean([float(field["score"]) for field in meaningful_fields])
            if meaningful_fields
            else None
        ),
        "requested_fields": requested_fields,
        "scored_fields": scored_fields,
        "ground_truth_coverage": (
            round(scored_fields / requested_fields, 3)
            if requested_fields
            else None
        ),
    }


def _tokens_per_second(tokens: Any, seconds: Any) -> float | None:
    if not isinstance(tokens, (int, float)):
        return None
    if not isinstance(seconds, (int, float)) or seconds <= 0:
        return None
    return round(float(tokens) / float(seconds), 3)


def _process_case(
    *,
    position: int,
    row_index: int,
    row: dict[str, str],
    api_url: str,
    api_key: str,
    request_timeout: float,
    temperature: float,
) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    response: dict[str, Any] | None = None
    schema: list[dict[str, Any]] = []
    ground_truth: dict[str, Any] = {}
    status = "client_error"
    error_category = ""
    error = ""
    http_status: int | None = None
    retry_after: str | None = None
    response_body = ""
    request_seconds: float | None = None
    schema_errors: list[str] = []
    accuracy: dict[str, Any] | None = None

    try:
        payload, schema = _build_payload(row, temperature)
        raw_truth = _json_cell(row, "post_call_detail", {})
        ground_truth = raw_truth if isinstance(raw_truth, dict) else {}
        request_started = time.monotonic()
        try:
            response = _request_json(
                "POST",
                f"{api_url}/extract",
                api_key=api_key,
                timeout=request_timeout,
                payload=payload,
            )
            http_status = 200
        finally:
            request_seconds = round(time.monotonic() - request_started, 3)

        result = response.get("result")
        if not isinstance(result, dict):
            raise APIRequestError(
                "API response is missing a result object",
                category="invalid_response",
                http_status=200,
                response_body=json.dumps(response, ensure_ascii=False)[:5000],
            )
        schema_errors = _validate_result(result, schema)
        accuracy = _score_result(result, ground_truth, schema)
        status = "done"
    except APIRequestError as exc:
        status = exc.category
        error_category = exc.category
        error = str(exc)
        http_status = exc.http_status
        retry_after = exc.retry_after
        response_body = exc.response_body
    except Exception as exc:
        error_category = "client_error"
        error = f"{type(exc).__name__}: {exc}"

    elapsed = round(time.monotonic() - started, 3)
    performance = (response or {}).get("performance") or {}
    generation_seconds = performance.get("generation_seconds")
    prompt_tokens = performance.get("prompt_tokens")
    completion_tokens = performance.get("completion_tokens")

    return {
        "position": position,
        "row_index": row_index,
        "versionId": row.get("versionId", ""),
        "status": status,
        "http_status": http_status,
        "started_at": started_at,
        "api_request_seconds": request_seconds,
        "end_to_end_seconds": elapsed,
        "model_generation_seconds": generation_seconds,
        "queue_and_network_seconds": (
            round(request_seconds - generation_seconds, 3)
            if isinstance(request_seconds, (int, float))
            and isinstance(generation_seconds, (int, float))
            else None
        ),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "completion_tokens_per_second": _tokens_per_second(
            completion_tokens, generation_seconds
        ),
        "generation_attempts": performance.get("attempts"),
        "retried": performance.get("retried"),
        "schema_valid": status == "done" and not schema_errors,
        "schema_errors": schema_errors,
        "accuracy": accuracy,
        "error_category": error_category,
        "error": error,
        "retry_after": retry_after,
        "error_response_body": response_body,
        "ground_truth": ground_truth,
        "api_response": response,
    }


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 3)
    rank = (len(ordered) - 1) * percentile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    value = ordered[lower]
    if lower != upper:
        value += (ordered[upper] - ordered[lower]) * (rank - lower)
    return round(value, 3)


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def _flat_result(result: dict[str, Any]) -> dict[str, Any]:
    evaluation = result.get("accuracy") or {}
    overall_score = evaluation.get("overall_score")
    return {
        "position": result["position"],
        "row_index": result["row_index"],
        "versionId": result["versionId"],
        "status": result["status"],
        "http_status": result["http_status"],
        "started_at": result["started_at"],
        "submit_latency_seconds": result.get("submit_latency_seconds"),
        "api_request_seconds": result["api_request_seconds"],
        "end_to_end_seconds": result["end_to_end_seconds"],
        "accuracy": overall_score,
        "accuracy_percent": (
            round(overall_score * 100, 2)
            if isinstance(overall_score, (int, float))
            else None
        ),
        "categorical_accuracy": evaluation.get("categorical_score"),
        "text_accuracy": evaluation.get("text_score"),
        "strict_accuracy": evaluation.get("strict_match_rate"),
        "meaningful_accuracy": evaluation.get("meaningful_match_rate"),
        "ground_truth_coverage": evaluation.get("ground_truth_coverage"),
        "fields_requested": evaluation.get("requested_fields"),
        "fields_scored": evaluation.get("scored_fields"),
        "model_generation_seconds": result["model_generation_seconds"],
        "queue_and_network_seconds": result["queue_and_network_seconds"],
        "prompt_tokens": result["prompt_tokens"],
        "completion_tokens": result["completion_tokens"],
        "completion_tokens_per_minute": _tokens_per_minute(
            result["completion_tokens"],
            result["model_generation_seconds"],
        ),
        "generation_attempts": result["generation_attempts"],
        "retried": result["retried"],
        "schema_valid": result["schema_valid"],
        "error_category": result["error_category"],
        "error": result["error"],
        "retry_after": result["retry_after"],
    }


ROW_FIELDS = [
    "position", "row_index", "versionId", "status", "http_status",
    "started_at", "submit_latency_seconds", "api_request_seconds",
    "end_to_end_seconds", "accuracy", "accuracy_percent",
    "categorical_accuracy", "text_accuracy", "strict_accuracy",
    "meaningful_accuracy", "ground_truth_coverage", "fields_requested",
    "fields_scored", "model_generation_seconds", "queue_and_network_seconds",
    "prompt_tokens", "completion_tokens", "completion_tokens_per_minute",
    "generation_attempts", "retried", "schema_valid", "error_category",
    "error", "retry_after",
]

FIELD_FIELDS = [
    "position", "row_index", "versionId", "field_name", "category",
    "model_value", "truth_value", "score", "match", "schema_type",
    "type_valid", "strict_match", "meaningful_truth",
]


def _field_reports_for_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    fields = ((result.get("accuracy") or {}).get("fields") or {})
    return [
        {
            "position": result["position"],
            "row_index": result["row_index"],
            "versionId": result["versionId"],
            "field_name": field_name,
            **field_result,
        }
        for field_name, field_result in sorted(fields.items())
    ]


def _tokens_per_minute(tokens: Any, seconds: Any) -> float | None:
    rate = _tokens_per_second(tokens, seconds)
    return round(rate * 60, 3) if rate is not None else None


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _build_summary(
    *,
    concurrency: int,
    requested_rows: int,
    results: list[dict[str, Any]],
    wall_seconds: float,
) -> dict[str, Any]:
    completed = [item for item in results if item["status"] == "done"]
    scored = [
        item
        for item in completed
        if isinstance(
            (item.get("accuracy") or {}).get("overall_score"),
            (int, float),
        )
    ]
    failed = [item for item in results if item["status"] != "done"]

    row_scores = [float(item["accuracy"]["overall_score"]) for item in scored]
    strict_rates = [
        float(item["accuracy"]["strict_match_rate"])
        for item in scored
        if isinstance(item["accuracy"].get("strict_match_rate"), (int, float))
    ]
    meaningful_rates = [
        float(item["accuracy"]["meaningful_match_rate"])
        for item in scored
        if isinstance(item["accuracy"].get("meaningful_match_rate"), (int, float))
    ]
    coverage_rates = [
        float(item["accuracy"]["ground_truth_coverage"])
        for item in scored
        if isinstance(item["accuracy"].get("ground_truth_coverage"), (int, float))
    ]
    all_fields = [
        field
        for item in scored
        for field in (item["accuracy"].get("fields") or {}).values()
    ]
    field_scores = [
        float(field["score"])
        for field in all_fields
        if isinstance(field.get("score"), (int, float))
    ]
    field_matches = [bool(field.get("match")) for field in all_fields]

    generation_times = [
        float(item["model_generation_seconds"])
        for item in completed
        if isinstance(item.get("model_generation_seconds"), (int, float))
    ]
    prompt_tokens = [
        float(item["prompt_tokens"])
        for item in completed
        if isinstance(item.get("prompt_tokens"), (int, float))
    ]
    completion_tokens = [
        float(item["completion_tokens"])
        for item in completed
        if isinstance(item.get("completion_tokens"), (int, float))
    ]
    per_job_completion_tpm = [
        rate
        for item in completed
        if (
            rate := _tokens_per_minute(
                item.get("completion_tokens"),
                item.get("model_generation_seconds"),
            )
        )
        is not None
    ]
    end_to_end = [
        float(item["end_to_end_seconds"])
        for item in completed
        if isinstance(item.get("end_to_end_seconds"), (int, float))
    ]
    submit_latencies = [
        float(item["submit_latency_seconds"])
        for item in results
        if isinstance(item.get("submit_latency_seconds"), (int, float))
    ]
    total_prompt_tokens = sum(prompt_tokens)
    total_completion_tokens = sum(completion_tokens)
    retried_rows = sum(bool(item.get("retried")) for item in completed)

    return {
        "requested_rows": requested_rows,
        "concurrency": concurrency,
        "completed_rows": len(completed),
        "scored_rows": len(scored),
        "failed_rows": len(failed),
        "wall_time_seconds": round(wall_seconds, 3),
        "throughput_rows_per_second": (
            round(len(completed) / wall_seconds, 3) if wall_seconds else None
        ),
        "accuracy": {
            "mean_per_row_score": _mean(row_scores),
            "mean_per_row_percent": (
                round(sum(row_scores) / len(row_scores) * 100, 2)
                if row_scores else None
            ),
            "weighted_mean_field_score": _mean(field_scores),
            "field_match_rate": (
                round(sum(field_matches) / len(field_matches), 3)
                if field_matches else None
            ),
            "fields_scored": len(all_fields),
            "strict_type_and_value_match_rate": _mean(strict_rates),
            "meaningful_nondefault_match_rate": _mean(meaningful_rates),
            "mean_ground_truth_coverage": _mean(coverage_rates),
        },
        "inference": {
            "mean_generation_seconds": _mean(generation_times),
            "mean_prompt_tokens": _mean(prompt_tokens),
            "mean_completion_tokens": _mean(completion_tokens),
            "total_prompt_tokens": int(total_prompt_tokens),
            "total_completion_tokens": int(total_completion_tokens),
            "completion_tokens_per_minute_wall": _tokens_per_minute(
                total_completion_tokens, wall_seconds
            ),
            "total_tokens_per_minute_wall": _tokens_per_minute(
                total_prompt_tokens + total_completion_tokens, wall_seconds
            ),
            "mean_per_job_completion_tokens_per_minute": _mean(
                per_job_completion_tpm
            ),
            "retried_rows": retried_rows,
            "retry_rate": (
                round(retried_rows / len(completed), 3) if completed else None
            ),
        },
        "end_to_end_latency_seconds": {
            "mean": _mean(end_to_end),
            "min": round(min(end_to_end), 3) if end_to_end else None,
            "p50": _percentile(end_to_end, 0.50),
            "p95": _percentile(end_to_end, 0.95),
            "p99": _percentile(end_to_end, 0.99),
            "max": round(max(end_to_end), 3) if end_to_end else None,
        },
        "submit_latency_seconds": {
            "mean": _mean(submit_latencies),
            "p50": _percentile(submit_latencies, 0.50),
            "p95": _percentile(submit_latencies, 0.95),
            "p99": _percentile(submit_latencies, 0.99),
        },
        "failed_row_indices": [item["row_index"] for item in failed],
        "artifacts": {
            "per_row": "rows.csv",
            "per_field": "field_scores.csv",
            "raw_results": "results.jsonl",
            "summary": "summary.json",
        },
    }


def _run_concurrency_level(
    *,
    concurrency: int,
    selected: list[tuple[int, dict[str, str]]],
    api_url: str,
    api_key: str,
    request_timeout: float,
    temperature: float,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json_atomic(
        output_dir / "config.json",
        {
            "api_url": api_url,
            "rows": len(selected),
            "concurrency": concurrency,
            "request_timeout": request_timeout,
            "temperature": temperature,
        },
    )

    results: list[dict[str, Any]] = []
    started = time.monotonic()
    results_path = output_dir / "results.jsonl"
    responses_path = output_dir / "responses.jsonl"
    rows_path = output_dir / "rows.csv"
    field_scores_path = output_dir / "field_scores.csv"

    with (
        results_path.open("w", encoding="utf-8") as results_handle,
        responses_path.open("w", encoding="utf-8") as responses_handle,
        rows_path.open("w", newline="", encoding="utf-8") as rows_handle,
        field_scores_path.open("w", newline="", encoding="utf-8") as fields_handle,
        ThreadPoolExecutor(max_workers=concurrency) as executor,
    ):
        row_writer = csv.DictWriter(rows_handle, fieldnames=ROW_FIELDS)
        field_writer = csv.DictWriter(fields_handle, fieldnames=FIELD_FIELDS)
        row_writer.writeheader()
        field_writer.writeheader()
        rows_handle.flush()
        fields_handle.flush()

        futures: dict[Any, float] = {}
        for position, (row_index, row) in enumerate(selected, start=1):
            submit_started = time.monotonic()
            future = executor.submit(
                _process_case,
                position=position,
                row_index=row_index,
                row=row,
                api_url=api_url,
                api_key=api_key,
                request_timeout=request_timeout,
                temperature=temperature,
            )
            futures[future] = round(time.monotonic() - submit_started, 6)

        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            result["submit_latency_seconds"] = futures[future]
            results.append(result)

            results_handle.write(
                json.dumps(result, ensure_ascii=False) + "\n"
            )
            results_handle.flush()

            response_record = {
                "position": result["position"],
                "row_index": result["row_index"],
                "versionId": result["versionId"],
                "status": result["status"],
                "http_status": result["http_status"],
                "error": result["error"],
                "result": (result.get("api_response") or {}).get("result"),
                "performance": (result.get("api_response") or {}).get(
                    "performance"
                ),
            }
            responses_handle.write(
                json.dumps(response_record, ensure_ascii=False) + "\n"
            )
            responses_handle.flush()

            row_writer.writerow(_flat_result(result))
            rows_handle.flush()

            for field_report in _field_reports_for_result(result):
                field_writer.writerow(field_report)
            fields_handle.flush()

            print(
                f"[concurrency={concurrency}] [{completed}/{len(selected)}] "
                f"row={result['row_index']} status={result['status']} "
                f"http={result['http_status']} "
                f"latency={result['api_request_seconds']}s",
                flush=True,
            )

    wall_seconds = time.monotonic() - started
    results.sort(key=lambda item: item["position"])
    summary = _build_summary(
        concurrency=concurrency,
        requested_rows=len(selected),
        results=results,
        wall_seconds=wall_seconds,
    )
    _write_json_atomic(output_dir / "summary.json", summary)
    return summary


def _server_slug(api_url: str) -> str:
    parsed = urlparse(api_url)
    raw = f"{parsed.hostname or 'server'}_{parsed.port or parsed.scheme}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)


def main() -> None:
    args = _parse_args()
    if not 0 <= args.temperature <= 2:
        raise ValueError("--temperature must be between 0 and 2")
    if not args.csv.is_file():
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    api_url = _normalize_api_url(args.api_url)
    api_key = _load_api_key()

    with args.csv.resolve().open(newline="", encoding="utf-8-sig") as handle:
        dataset = list(csv.DictReader(handle))
    if len(dataset) < args.rows:
        raise ValueError(
            f"CSV has {len(dataset)} rows, fewer than requested {args.rows}"
        )

    if args.selection == "first":
        selected_indices = list(range(args.rows))
    else:
        selected_indices = sorted(
            random.Random(args.random_state).sample(range(len(dataset)), args.rows)
        )
    selected = [(index, dataset[index]) for index in selected_indices]

    print(f"Checking {api_url}/health", flush=True)
    health = _request_json(
        "GET",
        f"{api_url}/health",
        api_key=api_key,
        timeout=min(args.request_timeout, 30.0),
    )
    if health.get("status") != "ok":
        raise RuntimeError(f"API health check was not OK: {health}")
    print(f"Health check passed: {health}", flush=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = (
        args.output_root.resolve()
        / f"benchmark_{run_id}"
        / _server_slug(api_url)
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    run_config = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "api_url": api_url,
        "csv": str(args.csv.resolve()),
        "rows_per_concurrency": args.rows,
        "total_planned_requests": args.rows * len(CONCURRENCY_LEVELS),
        "concurrency_levels": list(CONCURRENCY_LEVELS),
        "selection": args.selection,
        "random_state": args.random_state,
        "selected_indices": selected_indices,
        "request_timeout": args.request_timeout,
        "temperature": args.temperature,
        "health": health,
    }
    _write_json_atomic(output_dir / "run_config.json", run_config)
    print(f"Saving benchmark to {output_dir}", flush=True)

    completed_summaries: list[dict[str, Any]] = []
    for concurrency in CONCURRENCY_LEVELS:
        print(
            f"\nStarting concurrency={concurrency} with {args.rows} rows",
            flush=True,
        )
        level_dir = output_dir / f"concurrency_{concurrency:03d}"
        summary = _run_concurrency_level(
            concurrency=concurrency,
            selected=selected,
            api_url=api_url,
            api_key=api_key,
            request_timeout=args.request_timeout,
            temperature=args.temperature,
            output_dir=level_dir,
        )
        completed_summaries.append(summary)
        _write_json_atomic(
            output_dir / "overall_summary.json",
            {
                "api_url": api_url,
                "rows_per_concurrency": args.rows,
                "completed_concurrency_levels": [
                    item["concurrency"] for item in completed_summaries
                ],
                "summaries": completed_summaries,
            },
        )
        success_rate = (
            summary["completed_rows"] / summary["requested_rows"]
            if summary["requested_rows"]
            else 0.0
        )
        print(
            f"Saved concurrency={concurrency}: "
            f"success_rate={success_rate:.3f} "
            f"throughput={summary['throughput_rows_per_second']} rows/s",
            flush=True,
        )

    print(f"\nBenchmark complete. Results: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
