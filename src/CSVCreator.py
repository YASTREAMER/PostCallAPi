"""Add Node-server outcome fields to every row's `postcall` CSV schema."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_DIR / "data" / "Data.csv"
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "Data_with_outcome_fields.csv"


def _outcome_fields(conversion_reason: str) -> list[dict[str, Any]]:
    conversion_rule = conversion_reason.strip() or "conversion status"
    return [
        {
            "type": "boolean",
            "name": "conversion_status",
            "description": conversion_rule,
        },
        {
            "type": "string",
            "name": "disposition_reason",
            "description": (
                "why the user didn't convert, according to conversion reason: "
                f"{conversion_rule}, if user converted then provide a brief "
                "important information about conversion. This must never be "
                "empty — if the call was too short or no clear reason was "
                'found, use "Call too short / no response" or a brief summary '
                "of what happened."
            ),
        },
    ]


def _upsert_outcome_fields(
    postcall_data: list[dict[str, Any]], conversion_reason: str
) -> list[dict[str, Any]]:
    """Mirror JavaScript Map.set by replacing matching names in place."""
    result = list(postcall_data)
    index_by_name = {
        str(field.get("name", "")).strip(): index
        for index, field in enumerate(result)
        if isinstance(field, dict) and str(field.get("name", "")).strip()
    }
    for field in _outcome_fields(conversion_reason):
        name = field["name"]
        if name in index_by_name:
            result[index_by_name[name]] = field
        else:
            index_by_name[name] = len(result)
            result.append(field)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add conversion_status and disposition_reason to every postcall "
            "schema without changing post_call_detail ground truth."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    if input_path == output_path:
        raise ValueError(
            "Refusing to overwrite the source CSV; choose a different --output"
        )

    with input_path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or "postcall" not in reader.fieldnames:
            raise ValueError("Input CSV must contain a 'postcall' column")
        rows = list(reader)
        fieldnames = reader.fieldnames

    for row_index, row in enumerate(rows):
        raw_schema = row.get("postcall", "")
        try:
            postcall_data = json.loads(raw_schema) if raw_schema else []
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Row {row_index} has invalid JSON in 'postcall': {exc}"
            ) from exc
        if not isinstance(postcall_data, list):
            raise ValueError(
                f"Row {row_index} 'postcall' must contain a JSON list"
            )

        row["postcall"] = json.dumps(
            _upsert_outcome_fields(
                postcall_data,
                row.get("conversion_reason", ""),
            ),
            ensure_ascii=False,
            separators=(",", ":"),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Updated {len(rows)} rows")
    print(f"Source preserved: {input_path}")
    print(f"Output written: {output_path}")


if __name__ == "__main__":
    main()
