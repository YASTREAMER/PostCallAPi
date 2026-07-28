"""
Post-processing normalizer: coerces empty/missing boolean-field values to
an explicit `false`, matching your own TASK_INSTRUCTIONS contract
("If a boolean variable is not present, assign it the value false.").

The model already knows this rule -- it's in the prompt every single call
-- it just doesn't reliably execute it for low-salience / rarely-true
fields (things like Address_verification_required, alternate_contact_number,
etc., which are almost always false in your data). This closes that gap
deterministically instead of requiring a retrain.

Two places to use it:

  1. evaluator.py -- right after _extract_json(...) and before scoring,
     so eval numbers reflect real prediction accuracy instead of the
     empty-vs-false formatting artifact. See wiring notes at the bottom
     of this file.

  2. The live serving path (model_service.py) -- right before the
     response is returned to the node server, so production output
     matches the documented contract even on the calls where the model
     itself doesn't fill it in. This is the one that actually matters
     for users -- the eval-side fix just lets you measure correctly.

Deliberately reuses evaluation._is_empty / evaluation._normalize_value
instead of redefining "what counts as empty" here. If you ever change
that definition in evaluation.py, this file picks it up automatically --
there's no second copy of that logic to drift out of sync.
"""

from typing import Any, Dict, Iterable, Union

from evaluation import _is_empty  # reuse -- see module docstring

DEFAULTED_COMMENT = "<auto-defaulted: field not mentioned in call>"
MISSING_COMMENT = "Model did not provide supporting evidence for this value."
MISSING_FIELD_COMMENT = "Model did not return this field."


def _field_type_map(schema_fields: Union[Iterable[Any], Dict[str, str]]) -> Dict[str, str]:
    """schema_fields can be req.postcall_data (objects with .name/.type,
    as used in evaluator.py), a list of dicts with 'name'/'type' keys
    (as parsed straight from the postcall CSV column), or an existing
    {name: type} map (e.g. evaluator.py's spec_type_by_name)."""
    if isinstance(schema_fields, dict):
        return schema_fields
    out = {}
    for f in schema_fields:
        if isinstance(f, dict):
            out[f["name"]] = f.get("type", "text")
        else:
            out[f.name] = getattr(f, "type", "text")
    return out


def normalize_boolean_defaults(
    model_output: dict, schema_fields: Union[Iterable[Any], Dict[str, str]]
) -> dict:
    """Returns a NEW dict (does not mutate model_output) where every
    schema field declared type=='boolean' whose value is empty or
    missing gets coerced to explicit False, preserving the
    {"value": ..., "comment": ...} shape your contract expects.

    Non-boolean fields, and boolean fields the model already gave an
    explicit (non-empty) value for, pass through completely unchanged --
    this only touches the specific empty-boolean case."""
    type_by_name = _field_type_map(schema_fields)
    result = dict(model_output)  # shallow copy -- only top-level keys we touch are replaced

    for name, spec_type in type_by_name.items():
        if spec_type != "boolean":
            continue

        entry = result.get(name, "<missing>")
        current_value = entry.get("value", "<missing>") if isinstance(entry, dict) else entry

        if not _is_empty(current_value):
            continue  # model already gave an explicit value here -- leave it alone

        if isinstance(entry, dict):
            new_entry = dict(entry)
            new_entry["value"] = False
            if _is_empty(new_entry.get("comment")):
                new_entry["comment"] = DEFAULTED_COMMENT
        else:
            new_entry = {"value": False, "comment": DEFAULTED_COMMENT}

        result[name] = new_entry

    return result


def normalize_model_output(
    model_output: dict, schema_fields: Union[Iterable[Any], Dict[str, str]]
) -> dict:
    """Apply all schema-aware normalization and enforce value/comment objects."""
    type_by_name = _field_type_map(schema_fields)
    result = normalize_boolean_defaults(model_output, type_by_name)

    # Match Data.csv's post_call_detail contract for every requested field:
    # {"field": {"value": ..., "comment": "supporting call evidence"}}.
    for name, spec_type in type_by_name.items():
        if name not in result:
            result[name] = {
                "value": False if spec_type == "boolean" else "",
                "comment": MISSING_FIELD_COMMENT,
            }
            continue

        entry = result[name]
        if not isinstance(entry, dict):
            result[name] = {
                "value": entry,
                "comment": MISSING_COMMENT,
            }
            continue

        new_entry = dict(entry)
        if "value" not in new_entry:
            new_entry["value"] = False if spec_type == "boolean" else ""
        if _is_empty(new_entry.get("comment")):
            new_entry["comment"] = MISSING_COMMENT
        result[name] = new_entry

    # The output contract allows only requested schema fields at the top level.
    return {name: result[name] for name in type_by_name}


# --------------------------------------------------------------------------
# Wiring into evaluator.py: in the per-row loop, right after
# `model_output = _extract_json(answer_text)` succeeds, add:
#
#     from normalize_output import normalize_boolean_defaults
#     model_output = normalize_boolean_defaults(model_output, req.postcall_data)
#
# Everything downstream (schema_names, spec_type_by_name, the scoring loop)
# stays exactly as-is -- normalize_boolean_defaults just cleans model_output
# before evaluation.score_field ever sees it.
#
# Wiring into the live server (model_service.py): apply the same call
# right before the response dict is returned/serialized, using whatever
# variable holds the parsed model JSON and the request's postcall_data.
# --------------------------------------------------------------------------
