
import json
import math
import re
from typing import Any, Dict, Iterable, Union

from schemas import normalize_schema_type

DEFAULTED_COMMENT = "<auto-defaulted: field not mentioned in call>"
MISSING_COMMENT = "Model did not provide supporting evidence for this value."
MISSING_FIELD_COMMENT = "Model did not return this field."

_TRUE_STRINGS = {"1", "true", "yes", "y", "on"}
_FALSE_STRINGS = {
    "0",
    "0.0",
    "false",
    "no",
    "n",
    "off",
    "none",
    "null",
    "n/a",
    "na",
    "",
}
_NUMBER_PATTERN = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")


def _is_empty(value: Any) -> bool:
    """Return whether a model value represents missing output."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str):
        normalized = value.strip()
        while (
            len(normalized) >= 2
            and normalized[0] == '"'
            and normalized[-1] == '"'
        ):
            normalized = normalized[1:-1].strip()
        return normalized in {"", "<missing>"}
    return value == "<missing>"


def _field_type_map(
    schema_fields: Union[Iterable[Any], Dict[str, str]],
) -> Dict[str, str]:
    if isinstance(schema_fields, dict):
        return {
            str(name).strip(): normalize_schema_type(spec_type)
            for name, spec_type in schema_fields.items()
        }

    result = {}
    for field in schema_fields:
        if isinstance(field, dict):
            name = str(field["name"]).strip()
            spec_type = field.get("type", "text")
        else:
            name = field.name.strip()
            spec_type = getattr(field, "type", "text")
        result[name] = normalize_schema_type(spec_type)
    return result


def _field_default_map(
    schema_fields: Union[Iterable[Any], Dict[str, str]],
) -> Dict[str, Any]:
    if isinstance(schema_fields, dict):
        return {}

    defaults = {}
    for field in schema_fields:
        if isinstance(field, dict):
            if "defaultValue" in field:
                defaults[str(field["name"]).strip()] = field["defaultValue"]
            continue

        if "defaultValue" in getattr(field, "model_fields_set", set()):
            defaults[field.name.strip()] = getattr(field, "defaultValue", None)
    return defaults


def _default_for_type(spec_type: str, configured_default: Any) -> Any:
    if configured_default is not None:
        return configured_default
    if spec_type == "boolean":
        return False
    if spec_type == "number":
        return 0
    if spec_type in {"text", "string", "selector", "categorical"}:
        return ""
    return configured_default


def _coerce_boolean(value: Any, default: Any = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value == 1:
            return True
        if value == 0 or (isinstance(value, float) and math.isnan(value)):
            return False
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in _TRUE_STRINGS:
            return True
        if normalized in _FALSE_STRINGS:
            return False
    return default if isinstance(default, bool) else False


def _coerce_number(value: Any, default: Any = 0) -> int | float:
    if isinstance(value, bool):
        value = int(value)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            value = default
        else:
            return value
    if isinstance(value, str):
        normalized = value.strip().replace(",", "")
        if _NUMBER_PATTERN.fullmatch(normalized):
            parsed = float(normalized) if "." in normalized else int(normalized)
            return parsed
    if isinstance(default, (int, float)) and not isinstance(default, bool):
        return default
    return 0


def _coerce_string(value: Any, default: Any = "") -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default if isinstance(default, str) else ""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def coerce_value_for_type(value: Any, spec_type: str, default: Any = None) -> Any:
    """Return a JSON value that conforms to the requested schema type."""
    spec_type = normalize_schema_type(spec_type)
    effective_default = _default_for_type(spec_type, default)
    if spec_type == "boolean":
        return _coerce_boolean(value, effective_default)
    if spec_type == "number":
        return _coerce_number(value, effective_default)
    if spec_type in {"text", "string", "selector", "categorical"}:
        return _coerce_string(value, effective_default)
    return value if value is not None else effective_default


def value_matches_type(value: Any, spec_type: str) -> bool:
    """Strict JSON type check used by the benchmark."""
    spec_type = normalize_schema_type(spec_type)
    if spec_type == "boolean":
        return isinstance(value, bool)
    if spec_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if spec_type in {"text", "string", "selector", "categorical"}:
        return isinstance(value, str)
    return True


def normalize_boolean_defaults(
    model_output: dict,
    schema_fields: Union[Iterable[Any], Dict[str, str]],
) -> dict:
    type_by_name = _field_type_map(schema_fields)
    default_by_name = _field_default_map(schema_fields)
    result = dict(model_output)

    for name, spec_type in type_by_name.items():
        if spec_type != "boolean":
            continue
        entry = result.get(name)
        value = entry.get("value") if isinstance(entry, dict) else entry
        coerced = _coerce_boolean(value, default_by_name.get(name, False))
        if isinstance(entry, dict):
            normalized_entry = dict(entry)
            normalized_entry["value"] = coerced
            if _is_empty(normalized_entry.get("comment")):
                normalized_entry["comment"] = DEFAULTED_COMMENT
        else:
            normalized_entry = {
                "value": coerced,
                "comment": DEFAULTED_COMMENT,
            }
        result[name] = normalized_entry

    return result


def normalize_model_output(
    model_output: dict,
    schema_fields: Union[Iterable[Any], Dict[str, str]],
) -> dict:
    type_by_name = _field_type_map(schema_fields)
    default_by_name = _field_default_map(schema_fields)
    source = model_output if isinstance(model_output, dict) else {}
    normalized: dict[str, dict[str, Any]] = {}

    for name, spec_type in type_by_name.items():
        configured_default = default_by_name.get(name)
        entry = source.get(name)
        missing = name not in source

        if isinstance(entry, dict):
            value = entry.get("value")
            comment = entry.get("comment")
        else:
            value = entry
            comment = None

        coerced_value = coerce_value_for_type(
            value,
            spec_type,
            configured_default,
        )
        if _is_empty(comment):
            comment = MISSING_FIELD_COMMENT if missing else MISSING_COMMENT
        elif not isinstance(comment, str):
            comment = _coerce_string(comment)

        normalized[name] = {
            "value": coerced_value,
            "comment": comment,
        }

    return normalized
