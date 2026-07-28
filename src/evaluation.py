from fuzzywuzzy import fuzz
import difflib
import json
import re as _re
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Iterable

EVAL_LOG_PATH = "eval_log/eval_logs.jsonl"

CATEGORICAL_WORD_LIMIT = 4  # ground-truth values with <= this many words -> categorical
TEXT_MATCH_THRESHOLD = 0.6  # fuzzy score at/above this counts as a "match" for text fields

_LOG_LOCK = threading.Lock()

# Values that count as "nothing extracted" on either side of a comparison.
_EMPTY_VALUES = (None, "<missing>", "")

# Values a categorical ground-truth field uses as a "0"/null sentinel in some
# source data (e.g. alternate_contact_number = "0" meaning none given).
_ZERO_SENTINEL_STRINGS = ("0", "0.0")

def _normalize_value(v):
    """Strips accidental double-JSON-encoding and normalizes common formatting
    noise before comparison — e.g. '"neutral"' -> 'neutral', '2026-07-24T08:00:00'
    and '2026-07-24 08:00:00' treated as equal."""
    if v in (None, "<missing>"):
        return v
    s = str(v).strip()
    # strip one layer of surrounding quotes if the model double-encoded the string
    while len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1].strip()
    # normalize ISO-ish timestamps: swap T for space so both forms compare equal
    s = _re.sub(r"(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})", r"\1 \2", s)
    return s


def _is_empty(v) -> bool:
    """True if v represents 'nothing extracted' — None, the missing sentinel,
    empty string, or a quoted empty string like '""' that normalizes to ''."""
    if v in _EMPTY_VALUES:
        return True
    return _normalize_value(v) == ""


def extract_value(d: dict, key: str):
    key = key.strip()
    if key in d:
        entry = d[key]
    else:
        stripped_map = {k.strip(): v for k, v in d.items()}
        entry = stripped_map.get(key, "<missing>")
    if isinstance(entry, dict):
        return entry.get("value", "<missing>")
    return entry


def classify_field(spec_type: str, ground_truth_value) -> str:
    if isinstance(ground_truth_value, bool):
        return "categorical"
    if spec_type == "boolean":
        return "categorical"
    if spec_type in {"text", "string"}:
        return "text"
    text_val = str(ground_truth_value)
    if len(text_val.split()) <= CATEGORICAL_WORD_LIMIT:
        return "categorical"
    return "text"


def zero_ground_truth_to_false(spec_type: str, truth_val):
    category = classify_field(spec_type, truth_val)
    if category != "categorical":
        return truth_val
    if isinstance(truth_val, bool):
        return truth_val  # already boolean, nothing to do
    if isinstance(truth_val, (int, float)) and truth_val == 0:
        return False
    if isinstance(truth_val, str) and truth_val.strip() in _ZERO_SENTINEL_STRINGS:
        return False
    return truth_val


def score_categorical(model_val, truth_val) -> float:
    # Normalize empty values before calling string methods: _normalize_value
    # deliberately preserves None and the missing sentinel.
    if _is_empty(model_val) and _is_empty(truth_val):
        return 1.0
    if _is_empty(model_val) or _is_empty(truth_val):
        return 0.0

    return (
        1.0
        if _normalize_value(model_val).strip().casefold()
        == _normalize_value(truth_val).strip().casefold()
        else 0.0
    )


def _token_overlap_ratio(a: str, b: str) -> float:
    """Jaccard overlap of words — cares about shared meaning, not word order/exact phrasing."""
    a_tokens = set(str(a).lower().split())
    b_tokens = set(str(b).lower().split())
    if not a_tokens and not b_tokens:
        return 1.0
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def score_text(model_val, truth_val) -> float:
    # Both sides agree there's nothing here — that's a match, not a miss.
    if _is_empty(model_val) and _is_empty(truth_val):
        return 1.0
    # Exactly one side is empty — that's a real miss.
    if _is_empty(model_val) or _is_empty(truth_val):
        return 0.0

    model_str = _normalize_value(model_val)
    truth_str = _normalize_value(truth_val)
    char_ratio = difflib.SequenceMatcher(None, model_str.lower(), truth_str.lower()).ratio()
    token_ratio = _token_overlap_ratio(model_str, truth_str)
    fuzzy_set_ratio = fuzz.token_set_ratio(model_str.lower(), truth_str.lower()) / 100.0
    return max(char_ratio, token_ratio, fuzzy_set_ratio)


def score_field(spec_type: str, model_val, truth_val) -> Dict[str, Any]:
    truth_val = zero_ground_truth_to_false(spec_type, truth_val)
    category = classify_field(spec_type, truth_val)

    # Empty categorical ground truths use the same negative/false convention
    # as numeric zero sentinels.
    if category == "categorical" and _is_empty(truth_val):
        truth_val = False

    # Empty categorical predictions represent a negative/false value. Do this
    # before scoring so the score, match flag, and reported model value agree.
    if category == "categorical" and _is_empty(model_val):
        model_val = False

    score = (
        score_categorical(model_val, truth_val)
        if category == "categorical"
        else score_text(model_val, truth_val)
    )
    is_match = (
        score >= 1.0 if category == "categorical" else score >= TEXT_MATCH_THRESHOLD
    )
    return {
        "category": category,
        "model_value": model_val,
        "truth_value": truth_val,
        "score": round(score, 3),
        "match": is_match,
    }


def evaluate_output(
    schema_names: Iterable[str],
    spec_type_by_name: Dict[str, str],
    model_output: dict,
    ground_truth: dict,
) -> Dict[str, Any]:
    """Scores model_output against ground_truth for every field in the schema
    that ground_truth actually has a value for."""
    field_results = {}
    for field_name in schema_names:
        truth_val = extract_value(ground_truth, field_name)
        if truth_val == "<missing>":
            continue  # node didn't provide ground truth for this field — skip, don't penalize
        model_val = extract_value(model_output, field_name)
        field_results[field_name] = score_field(
            spec_type_by_name.get(field_name, "text"), model_val, truth_val
        )

    if not field_results:
        return {
            "fields": {},
            "overall_score": None,
            "categorical_score": None,
            "text_score": None,
        }

    scores = [r["score"] for r in field_results.values()]
    cat_scores = [
        r["score"] for r in field_results.values() if r["category"] == "categorical"
    ]
    text_scores = [
        r["score"] for r in field_results.values() if r["category"] == "text"
    ]

    return {
        "fields": field_results,
        "overall_score": round(sum(scores) / len(scores), 3),
        "categorical_score": (
            round(sum(cat_scores) / len(cat_scores), 3) if cat_scores else None
        ),
        "text_score": (
            round(sum(text_scores) / len(text_scores), 3) if text_scores else None
        ),
    }


def log_evaluation(
    job_id: str, model_output: dict, eval_result: dict, log_path: str = EVAL_LOG_PATH
) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "job_id": job_id,
        "model_output": model_output,
        "eval_result": eval_result,
    }
    with _LOG_LOCK:
        with open(f"eval_csv/{log_path}", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
