import argparse
import csv
import json
import math
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen



PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CSV = PROJECT_DIR / "data" / "Data_with_outcome_fields.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "output"
MAX_ROWS = 10000

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

_PRINT_LOCK = threading.Lock()


def _json_cell(row: dict[str, str], column: str, default: Any) -> Any:
    raw = row.get(column, "")
    if not raw:
        return default
    return json.loads(raw)


def _add_outcome_fields(
    schema: list[dict[str, Any]], conversion_reason: str
) -> list[dict[str, Any]]:
    result = list(schema)
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


def _build_payload(row: dict[str, str]) -> dict[str, Any]:
    schema = _add_outcome_fields(
        _json_cell(row, "postcall", []),
        row.get("conversion_reason", ""),
    )
    return {
        "postcall_data": schema,
        "transcription": row.get("transcription", ""),
        "call_duration": (
            float(row["call_duration"]) if row.get("call_duration") else None
        ),
        "hangup_reason": row.get("hangup_reason", ""),
        "functions_called": _json_cell(row, "functions_called", []),
        "call_metadata": _json_cell(row, "call_metadata", {}),
    }


def _is_meaningful_truth(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, str):
        return value.strip().casefold() not in {
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
    return True


def _evaluate_result(
    payload: dict[str, Any],
    result: dict[str, Any],
    ground_truth: dict[str, Any],
) -> dict[str, Any]:
    import evaluation
    from normalize_output import value_matches_type
    from schemas import normalize_schema_type

    schema_names = {
        str(field.get("name", "")).strip() for field in payload["postcall_data"]
    }
    spec_type_by_name = {
        str(field.get("name", "")).strip(): normalize_schema_type(
            field.get("type", "text")
        )
        for field in payload["postcall_data"]
    }
    evaluated = evaluation.evaluate_output(
        schema_names,
        spec_type_by_name,
        result,
        ground_truth,
    )

    fields = evaluated.get("fields") or {}
    strict_matches = []
    meaningful_matches = []
    meaningful_scores = []
    for field_name, field_result in fields.items():
        entry = result.get(field_name)
        value = entry.get("value") if isinstance(entry, dict) else entry
        schema_type = spec_type_by_name.get(field_name, "text")
        type_valid = value_matches_type(value, schema_type)
        strict_match = bool(field_result.get("match")) and type_valid
        meaningful = _is_meaningful_truth(field_result.get("truth_value"))
        field_result.update(
            {
                "schema_type": schema_type,
                "type_valid": type_valid,
                "strict_match": strict_match,
                "meaningful_truth": meaningful,
            }
        )
        strict_matches.append(strict_match)
        if meaningful:
            meaningful_matches.append(bool(field_result.get("match")))
            meaningful_scores.append(float(field_result.get("score", 0)))

    requested_fields = len(payload["postcall_data"])
    scored_fields = len(fields)
    evaluated.update(
        {
            "strict_match_rate": (
                round(sum(strict_matches) / len(strict_matches), 3)
                if strict_matches
                else None
            ),
            "meaningful_match_rate": (
                round(sum(meaningful_matches) / len(meaningful_matches), 3)
                if meaningful_matches
                else None
            ),
            "meaningful_mean_score": (
                round(sum(meaningful_scores) / len(meaningful_scores), 3)
                if meaningful_scores
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
    )
    return evaluated


def _request_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    timeout: float = 30,
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
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        response_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"{method} {url} returned HTTP {exc.code}: {response_body}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(f"Cannot reach API at {url}: {exc.reason}") from exc


def _wait_for_job(
    api_url: str,
    job_id: str,
    poll_interval: float,
    job_timeout: float,
    request_timeout: float,
) -> tuple[dict[str, Any], int]:
    deadline = time.monotonic() + job_timeout
    polls = 0
    while time.monotonic() < deadline:
        response = _request_json(
            "GET",
            f"{api_url}/status/{job_id}",
            timeout=request_timeout,
        )
        polls += 1
        if response.get("status") in {"done", "error"}:
            return response, polls
        time.sleep(poll_interval)
    raise TimeoutError(f"Job {job_id} did not finish within {job_timeout}s")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


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
    if upper != lower:
        value += (ordered[upper] - ordered[lower]) * (rank - lower)
    return round(value, 3)


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def _tokens_per_minute(tokens: Any, seconds: Any) -> float | None:
    if not isinstance(tokens, (int, float)) or not isinstance(
        seconds, (int, float)
    ):
        return None
    if seconds <= 0:
        return None
    return round(float(tokens) / float(seconds) * 60, 2)


def _bounded_int(name: str, minimum: int, maximum: int):
    def parse(value: str) -> int:
        parsed = int(value)
        if not minimum <= parsed <= maximum:
            raise argparse.ArgumentTypeError(
                f"{name} must be between {minimum} and {maximum}"
            )
        return parsed

    return parse


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Submit up to 500 CSV rows to the live API concurrently and save "
            "latency, throughput, accuracy, per-field scores, and raw results."
        )
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument(
        "--count",
        type=_bounded_int("count", 1, MAX_ROWS),
        default=200,
        help="Number of rows to test (default: 200; maximum: 500).",
    )
    parser.add_argument(
        "--concurrency",
        type=_bounded_int("concurrency", 1, MAX_ROWS),
        default=10,
        help="Maximum number of in-flight API jobs (default: 10).",
    )
    parser.add_argument(
        "--selection",
        choices=("random", "first"),
        default="random",
        help="Choose a seeded random sample or the first N rows.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--job-timeout", type=float, default=1200.0)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    return parser.parse_args()


def _process_case(
    *,
    position: int,
    row_index: int,
    row: dict[str, str],
    api_url: str,
    poll_interval: float,
    job_timeout: float,
    request_timeout: float,
) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    job_id = ""
    accepted: dict[str, Any] | None = None
    response: dict[str, Any] | None = None
    payload: dict[str, Any] | None = None
    ground_truth: dict[str, Any] = {}
    evaluation_result: dict[str, Any] | None = None
    submit_seconds: float | None = None
    poll_count = 0
    error = ""
    status = "client_error"

    try:
        payload = _build_payload(row)
        ground_truth = _json_cell(row, "post_call_detail", {})
        submit_started = time.monotonic()
        accepted = _request_json(
            "POST",
            f"{api_url}/extract",
            payload,
            timeout=request_timeout,
        )
        submit_seconds = round(time.monotonic() - submit_started, 3)
        job_id = str(accepted["job_id"])
        response, poll_count = _wait_for_job(
            api_url,
            job_id,
            poll_interval,
            job_timeout,
            request_timeout,
        )
        status = str(response.get("status", "unknown"))
        error = str(response.get("error") or "")
        model_result = response.get("result")
        if status == "done" and isinstance(model_result, dict):
            evaluation_result = _evaluate_result(
                payload,
                model_result,
                ground_truth,
            )
    except Exception as exc:
        error = str(exc)

    elapsed = round(time.monotonic() - started, 3)
    accuracy = (
        evaluation_result.get("overall_score") if evaluation_result else None
    )
    field_count = len((evaluation_result or {}).get("fields") or {})
    performance = (response or {}).get("performance") or {}
    generation_seconds = performance.get("generation_seconds")
    prompt_tokens = performance.get("prompt_tokens")
    completion_tokens = performance.get("completion_tokens")

    return {
        "position": position,
        "row_index": row_index,
        "versionId": row.get("versionId", ""),
        "job_id": job_id,
        "status": status,
        "started_at": started_at,
        "submit_latency_seconds": submit_seconds,
        "end_to_end_seconds": elapsed,
        "poll_count": poll_count,
        "accuracy": accuracy,
        "accuracy_percent": (
            round(accuracy * 100, 2) if isinstance(accuracy, (int, float)) else None
        ),
        "categorical_accuracy": (
            evaluation_result.get("categorical_score")
            if evaluation_result
            else None
        ),
        "text_accuracy": (
            evaluation_result.get("text_score") if evaluation_result else None
        ),
        "strict_accuracy": (
            evaluation_result.get("strict_match_rate")
            if evaluation_result
            else None
        ),
        "meaningful_accuracy": (
            evaluation_result.get("meaningful_match_rate")
            if evaluation_result
            else None
        ),
        "ground_truth_coverage": (
            evaluation_result.get("ground_truth_coverage")
            if evaluation_result
            else None
        ),
        "fields_requested": (
            evaluation_result.get("requested_fields")
            if evaluation_result
            else None
        ),
        "fields_scored": field_count,
        "model_generation_seconds": generation_seconds,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "completion_tokens_per_minute": _tokens_per_minute(
            completion_tokens, generation_seconds
        ),
        "total_tokens_per_minute": _tokens_per_minute(
            (prompt_tokens + completion_tokens)
            if isinstance(prompt_tokens, (int, float))
            and isinstance(completion_tokens, (int, float))
            else None,
            generation_seconds,
        ),
        "generation_attempts": performance.get("attempts"),
        "retried": performance.get("retried"),
        "error": error,
        "ground_truth": ground_truth,
        "model_result": (response or {}).get("result"),
        "evaluation": evaluation_result,
        "accepted_response": accepted,
        "final_response": response,
    }


def _row_report(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: result.get(key)
        for key in (
            "position",
            "row_index",
            "versionId",
            "job_id",
            "status",
            "started_at",
            "submit_latency_seconds",
            "end_to_end_seconds",
            "poll_count",
            "accuracy",
            "accuracy_percent",
            "categorical_accuracy",
            "text_accuracy",
            "strict_accuracy",
            "meaningful_accuracy",
            "ground_truth_coverage",
            "fields_requested",
            "fields_scored",
            "model_generation_seconds",
            "prompt_tokens",
            "completion_tokens",
            "completion_tokens_per_minute",
            "total_tokens_per_minute",
            "generation_attempts",
            "retried",
            "error",
        )
    }


def _field_reports(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        fields = ((result.get("evaluation") or {}).get("fields") or {})
        for field_name, field_result in sorted(fields.items()):
            rows.append(
                {
                    "position": result["position"],
                    "row_index": result["row_index"],
                    "versionId": result["versionId"],
                    "field_name": field_name,
                    **field_result,
                }
            )
    return rows


def _build_summary(
    *,
    args: argparse.Namespace,
    selected_indices: list[int],
    results: list[dict[str, Any]],
    wall_seconds: float,
    output_dir: Path,
) -> dict[str, Any]:
    done = [result for result in results if result["status"] == "done"]
    scored = [
        result
        for result in done
        if isinstance(result.get("accuracy"), (int, float))
    ]
    failed = [result for result in results if result["status"] != "done"]
    end_to_end = [float(result["end_to_end_seconds"]) for result in done]
    submit_latencies = [
        float(result["submit_latency_seconds"])
        for result in results
        if isinstance(result.get("submit_latency_seconds"), (int, float))
    ]
    row_accuracies = [float(result["accuracy"]) for result in scored]
    strict_accuracies = [
        float(result["strict_accuracy"])
        for result in scored
        if isinstance(result.get("strict_accuracy"), (int, float))
    ]
    meaningful_accuracies = [
        float(result["meaningful_accuracy"])
        for result in scored
        if isinstance(result.get("meaningful_accuracy"), (int, float))
    ]
    coverage_values = [
        float(result["ground_truth_coverage"])
        for result in scored
        if isinstance(result.get("ground_truth_coverage"), (int, float))
    ]
    model_seconds = [
        float(result["model_generation_seconds"])
        for result in done
        if isinstance(result.get("model_generation_seconds"), (int, float))
    ]
    prompt_tokens = [
        float(result["prompt_tokens"])
        for result in done
        if isinstance(result.get("prompt_tokens"), (int, float))
    ]
    completion_tokens = [
        float(result["completion_tokens"])
        for result in done
        if isinstance(result.get("completion_tokens"), (int, float))
    ]
    per_job_completion_tpm = [
        float(result["completion_tokens_per_minute"])
        for result in done
        if isinstance(
            result.get("completion_tokens_per_minute"), (int, float)
        )
    ]
    total_prompt_tokens = sum(prompt_tokens)
    total_completion_tokens = sum(completion_tokens)
    retry_count = sum(bool(result.get("retried")) for result in done)
    field_rows = _field_reports(results)
    field_scores = [
        float(row["score"])
        for row in field_rows
        if isinstance(row.get("score"), (int, float))
    ]
    field_matches = [row.get("match") for row in field_rows]

    return {
        "run_started_at": min(
            (result["started_at"] for result in results), default=None
        ),
        "csv": str(args.csv.resolve()),
        "api_url": args.api_url.rstrip("/"),
        "output_directory": str(output_dir),
        "selection": args.selection,
        "seed": args.seed,
        "selected_indices": selected_indices,
        "requested_rows": args.count,
        "concurrency": min(args.concurrency, args.count),
        "completed_rows": len(done),
        "scored_rows": len(scored),
        "failed_rows": len(failed),
        "wall_time_seconds": round(wall_seconds, 3),
        "throughput_rows_per_second": (
            round(len(done) / wall_seconds, 3) if wall_seconds else None
        ),
        "accuracy": {
            "mean_per_row_score": _mean(row_accuracies),
            "mean_per_row_percent": (
                round(sum(row_accuracies) / len(row_accuracies) * 100, 2)
                if row_accuracies
                else None
            ),
            "weighted_mean_field_score": _mean(field_scores),
            "field_match_rate": (
                round(sum(bool(value) for value in field_matches) / len(field_matches), 3)
                if field_matches
                else None
            ),
            "fields_scored": len(field_rows),
            "strict_type_and_value_match_rate": _mean(strict_accuracies),
            "meaningful_nondefault_match_rate": _mean(meaningful_accuracies),
            "mean_ground_truth_coverage": _mean(coverage_values),
        },
        "inference": {
            "mean_generation_seconds": _mean(model_seconds),
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
            "retried_rows": retry_count,
            "retry_rate": (round(retry_count / len(done), 3) if done else None),
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
        "failed_row_indices": [result["row_index"] for result in failed],
        "artifacts": {
            "per_row": "rows.csv",
            "per_field": "field_scores.csv",
            "raw_results": "results.jsonl",
            "summary": "summary.json",
        },
    }


def main() -> None:
    args = _parse_args()
    if args.poll_interval <= 0:
        raise ValueError("--poll-interval must be greater than zero")
    if args.job_timeout <= 0 or args.request_timeout <= 0:
        raise ValueError("timeouts must be greater than zero")
    if not args.csv.is_file():
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    api_url = args.api_url.rstrip("/")
    with args.csv.resolve().open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    if len(rows) < args.count:
        raise ValueError(
            f"CSV has {len(rows)} rows, fewer than requested {args.count}"
        )

    if args.selection == "first":
        selected_indices = list(range(args.count))
    else:
        selected_indices = sorted(
            random.Random(args.seed).sample(range(len(rows)), args.count)
        )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = args.output_root.resolve() / f"api_test_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=False)
    workers = min(args.concurrency, args.count)

    print(
        f"Testing {args.count} rows from {args.csv.resolve()} with "
        f"concurrency={workers}"
    )
    print(f"Saving artifacts to {output_dir}")

    wall_started = time.monotonic()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _process_case,
                position=position,
                row_index=row_index,
                row=rows[row_index],
                api_url=api_url,
                poll_interval=args.poll_interval,
                job_timeout=args.job_timeout,
                request_timeout=args.request_timeout,
            ): (position, row_index)
            for position, row_index in enumerate(selected_indices, start=1)
        }
        completed_count = 0
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed_count += 1
            elapsed_wall = time.monotonic() - wall_started
            observed_completion_tokens = sum(
                item.get("completion_tokens") or 0 for item in results
            )
            wall_tpm = _tokens_per_minute(
                observed_completion_tokens, elapsed_wall
            )
            with _PRINT_LOCK:
                print(
                    f"[{completed_count}/{args.count}] row={result['row_index']} "
                    f"status={result['status']} time={result['end_to_end_seconds']}s "
                    f"accuracy={result['accuracy']} "
                    f"output_tokens={result['completion_tokens']} "
                    f"job_output_tpm={result['completion_tokens_per_minute']} "
                    f"wall_output_tpm={wall_tpm}"
                )

    wall_seconds = time.monotonic() - wall_started
    results.sort(key=lambda result: result["position"])
    row_reports = [_row_report(result) for result in results]
    field_reports = _field_reports(results)
    summary = _build_summary(
        args=args,
        selected_indices=selected_indices,
        results=results,
        wall_seconds=wall_seconds,
        output_dir=output_dir,
    )

    _write_csv(
        output_dir / "rows.csv",
        row_reports,
        fieldnames=[
            "position",
            "row_index",
            "versionId",
            "job_id",
            "status",
            "started_at",
            "submit_latency_seconds",
            "end_to_end_seconds",
            "poll_count",
            "accuracy",
            "accuracy_percent",
            "categorical_accuracy",
            "text_accuracy",
            "strict_accuracy",
            "meaningful_accuracy",
            "ground_truth_coverage",
            "fields_requested",
            "fields_scored",
            "model_generation_seconds",
            "prompt_tokens",
            "completion_tokens",
            "completion_tokens_per_minute",
            "total_tokens_per_minute",
            "generation_attempts",
            "retried",
            "error",
        ],
    )
    _write_csv(
        output_dir / "field_scores.csv",
        field_reports,
        fieldnames=[
            "position",
            "row_index",
            "versionId",
            "field_name",
            "category",
            "model_value",
            "truth_value",
            "score",
            "match",
            "schema_type",
            "type_valid",
            "strict_match",
            "meaningful_truth",
        ],
    )
    _write_jsonl(output_dir / "results.jsonl", results)
    _write_json(output_dir / "summary.json", summary)

    print(json.dumps(summary, indent=2))
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
