import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "output"


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
            "Submit one production-format request, wait for completion, and "
            "print/save the complete normalized API response."
        )
    )
    parser.add_argument(
        "--payload",
        type=Path,
        required=True,
        help="Path to a JSON object matching the POST /postcall/extract request schema.",
    )
    parser.add_argument("--api-url", default="http://127.0.0.1:8088/postcall")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON path; defaults to output/smoke_test_<timestamp>.json.",
    )
    parser.add_argument(
        "--poll-interval",
        type=_positive_float("poll interval"),
        default=1.0,
    )
    parser.add_argument(
        "--job-timeout",
        type=_positive_float("job timeout"),
        default=1200.0,
    )
    parser.add_argument(
        "--request-timeout",
        type=_positive_float("request timeout"),
        default=60.0,
    )
    return parser.parse_args()


def _request_json(
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float,
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
            decoded = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        response_body = exc.read().decode("utf-8", errors="replace")
        retry_after = exc.headers.get("Retry-After")
        retry_hint = f" Retry-After: {retry_after}s." if retry_after else ""
        raise RuntimeError(
            f"{method} {url} returned HTTP {exc.code}: "
            f"{response_body}.{retry_hint}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(f"Cannot reach API at {url}: {exc.reason}") from exc

    if not isinstance(decoded, dict):
        raise RuntimeError(f"{method} {url} returned a non-object JSON response")
    return decoded


def _load_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Payload file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Payload JSON must be an object")
    if "ground_truth" in payload:
        raise ValueError(
            "Production payload must not contain ground_truth; evaluation is "
            "kept outside the live API"
        )
    return payload


def _default_output_path() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_OUTPUT_DIR / f"smoke_test_{timestamp}.json"


def main() -> None:
    args = _parse_args()
    payload = _load_payload(args.payload)
    api_url = args.api_url.rstrip("/")
    output_path = args.output or _default_output_path()

    health = _request_json(
        "GET",
        f"{api_url}/health",
        timeout=args.request_timeout,
    )
    if health.get("status") != "ok":
        raise RuntimeError(f"API health check was not OK: {health}")

    started = time.monotonic()
    accepted = _request_json(
        "POST",
        f"{api_url}/extract",
        payload=payload,
        timeout=args.request_timeout,
    )
    job_id = accepted.get("job_id")
    if not isinstance(job_id, str) or not job_id:
        raise RuntimeError(f"API did not return a valid job_id: {accepted}")

    print(f"Accepted job {job_id}", file=sys.stderr)
    deadline = started + args.job_timeout
    last_status = None
    polls = 0
    final_response: dict[str, Any] | None = None

    while time.monotonic() < deadline:
        response = _request_json(
            "GET",
            f"{api_url}/status/{job_id}",
            timeout=args.request_timeout,
        )
        polls += 1
        status = response.get("status")
        if status != last_status:
            print(f"Job status: {status}", file=sys.stderr)
            last_status = status
        if status in {"done", "error"}:
            final_response = response
            break
        time.sleep(args.poll_interval)

    if final_response is None:
        raise TimeoutError(
            f"Job {job_id} did not finish within {args.job_timeout} seconds"
        )

    report = {
        "test": {
            "api_url": api_url,
            "payload_file": str(args.payload.resolve()),
            "poll_count": polls,
            "end_to_end_seconds": round(time.monotonic() - started, 3),
        },
        "response": final_response,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(final_response, ensure_ascii=False, indent=2))
    print(f"Saved complete response to {output_path}", file=sys.stderr)

    if final_response.get("status") != "done":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

