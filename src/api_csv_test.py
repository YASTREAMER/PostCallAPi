"""Submit a small, reproducible CSV sample to the live extraction API."""

import argparse
import csv
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CSV = PROJECT_DIR / "data" / "Data.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_DIR / "output"

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
        "ground_truth": _json_cell(row, "post_call_detail", {}),
    }


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
) -> dict[str, Any]:
    deadline = time.monotonic() + job_timeout
    while time.monotonic() < deadline:
        response = _request_json(
            "GET", f"{api_url}/status/{job_id}", timeout=30
        )
        if response.get("status") in {"done", "error"}:
            return response
        time.sleep(poll_interval)
    raise TimeoutError(f"Job {job_id} did not finish within {job_timeout}s")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--count", type=int, default=2, choices=range(2, 6))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--poll-interval", type=float, default=2)
    parser.add_argument("--job-timeout", type=float, default=600)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    api_url = args.api_url.rstrip("/")
    with args.csv.resolve().open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        rows = list(csv.DictReader(handle))

    if len(rows) < args.count:
        raise ValueError(
            f"CSV has {len(rows)} rows, fewer than requested {args.count}"
        )

    selected_indices = sorted(
        random.Random(args.seed).sample(range(len(rows)), args.count)
    )
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root.resolve() / f"api_test_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=False)

    summary_rows: list[dict[str, Any]] = []
    field_rows: list[dict[str, Any]] = []

    print(f"Testing {args.count} rows: {selected_indices}")
    print(f"Saving artifacts to {output_dir}")

    for position, row_index in enumerate(selected_indices, start=1):
        row = rows[row_index]
        payload = _build_payload(row)
        case_dir = output_dir / f"row_{row_index}"
        case_dir.mkdir()
        _write_json(case_dir / "request.json", payload)

        print(f"[{position}/{args.count}] Submitting row {row_index} ...")
        started = time.monotonic()
        try:
            accepted = _request_json(
                "POST", f"{api_url}/extract", payload, timeout=60
            )
            _write_json(case_dir / "accepted.json", accepted)
            job_id = accepted["job_id"]
            response = _wait_for_job(
                api_url,
                job_id,
                args.poll_interval,
                args.job_timeout,
            )
            elapsed = round(time.monotonic() - started, 3)
            _write_json(case_dir / "response.json", response)

            evaluation = response.get("eval_result") or {}
            summary_rows.append(
                {
                    "row_index": row_index,
                    "versionId": row.get("versionId", ""),
                    "job_id": job_id,
                    "status": response.get("status"),
                    "latency_seconds": elapsed,
                    "overall_score": evaluation.get("overall_score"),
                    "categorical_score": evaluation.get(
                        "categorical_score"
                    ),
                    "text_score": evaluation.get("text_score"),
                    "error": response.get("error"),
                }
            )
            for field_name, field_result in (
                evaluation.get("fields") or {}
            ).items():
                field_rows.append(
                    {
                        "row_index": row_index,
                        "versionId": row.get("versionId", ""),
                        "field_name": field_name,
                        **field_result,
                    }
                )
            print(
                f"[{position}/{args.count}] {response.get('status')} in "
                f"{elapsed}s; score={evaluation.get('overall_score')}"
            )
        except Exception as exc:
            elapsed = round(time.monotonic() - started, 3)
            error = str(exc)
            _write_json(case_dir / "error.json", {"error": error})
            summary_rows.append(
                {
                    "row_index": row_index,
                    "versionId": row.get("versionId", ""),
                    "job_id": "",
                    "status": "client_error",
                    "latency_seconds": elapsed,
                    "overall_score": "",
                    "categorical_score": "",
                    "text_score": "",
                    "error": error,
                }
            )
            print(f"[{position}/{args.count}] ERROR: {error}")

    _write_csv(output_dir / "summary.csv", summary_rows)
    _write_csv(output_dir / "field_scores.csv", field_rows)

    completed = [row for row in summary_rows if row["status"] == "done"]
    numeric_scores = [
        row["overall_score"]
        for row in completed
        if isinstance(row["overall_score"], (int, float))
    ]
    numeric_latencies = [
        row["latency_seconds"] for row in completed
    ]
    summary = {
        "csv": str(args.csv.resolve()),
        "api_url": api_url,
        "seed": args.seed,
        "selected_indices": selected_indices,
        "requested_rows": args.count,
        "completed_rows": len(completed),
        "failed_rows": args.count - len(completed),
        "average_overall_score": (
            round(sum(numeric_scores) / len(numeric_scores), 3)
            if numeric_scores
            else None
        ),
        "average_latency_seconds": (
            round(sum(numeric_latencies) / len(numeric_latencies), 3)
            if numeric_latencies
            else None
        ),
    }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
