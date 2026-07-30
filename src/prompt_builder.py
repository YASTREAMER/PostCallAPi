import json
import re

from schemas import ExtractRequest

SYSTEM_INSTRUCTION = (
    "You are a call-analysis extraction engine. You are given a call "
    "transcript and context, plus a list of fields to extract. Respond "
    "with ONLY a single valid JSON object mapping each field name to an "
    "object with exactly two keys: \"value\" and \"comment\". The comment "
    "must be a short evidence phrase of at most 12 words, not hidden "
    "reasoning or a step-by-step analysis. No prose "
    "outside the JSON and no markdown fences."
)

DESCRIPTION_MAX_CHARS = 360
DESCRIPTION_HEAD_CHARS = 230


def _shorten_description(description: str) -> str:
    """Compact a rule without deleting its later conditions/exceptions."""
    description = re.sub(r"\s+", " ", description or "").strip()
    if not description:
        return "(no description provided)"
    if len(description) <= DESCRIPTION_MAX_CHARS:
        return description

    tail_chars = DESCRIPTION_MAX_CHARS - DESCRIPTION_HEAD_CHARS - 3
    return (
        description[:DESCRIPTION_HEAD_CHARS].rstrip()
        + " … "
        + description[-tail_chars:].lstrip()
    )


def build_prompt(req: ExtractRequest) -> str:
    functions_called_json = json.dumps(
        [fc.model_dump(exclude_none=True) for fc in req.functions_called], ensure_ascii=False
    )
    call_metadata_json = json.dumps(req.call_metadata or {}, ensure_ascii=False)
    context = [
        f"### transcription\n{req.transcription}",
        f"### call_metadata\n{call_metadata_json}",
        f"### hangup_reason\n{req.hangup_reason or ''}",
        (
            "### call_duration\n"
            f"{req.call_duration if req.call_duration is not None else ''}"
        ),
        f"### functions_called\n{functions_called_json}",
    ]

    schema_lines = []
    for field in req.postcall_data:
        metadata = [f"type={field.type}"]
        if "defaultValue" in field.model_fields_set:
            metadata.append(
                "default=" + json.dumps(field.defaultValue, ensure_ascii=False)
            )
        schema_lines.append(
            f"- {field.name} ({', '.join(metadata)}): "
            f"{_shorten_description(field.description)}"
        )
    field_names = [field.name for field in req.postcall_data]
    context_block = "\n\n".join(context)
    schema_block = "\n".join(schema_lines)
    output_shape = {
        name: {"value": "<value>", "comment": "<brief evidence>"}
        for name in field_names
    }

    consistency_rules = ""
    field_name_set = set(field_names)
    if {
        "call_unknown_disconnect",
        "user_cut_the_call_after_hearing_first_message",
    }.issubset(field_name_set):
        consistency_rules = (
            "\n\n### Cross-field consistency\n"
            "- user_cut_the_call_after_hearing_first_message is true only "
            "when the user gave no meaningful response after the opening.\n"
            "- call_unknown_disconnect is true only when the user had already "
            "shown sell intent and then disconnected without a known reason.\n"
            "- These two fields cannot both be true for the same call."
        )

    return (
        f"{context_block}\n\n"
        f"### Fields to extract\n{schema_block}"
        f"{consistency_rules}\n\n"
        "### Value type rules\n"
        "- boolean: use the JSON literals true or false.\n"
        "- number: use a JSON number, including the configured numeric default "
        "when the value is absent.\n"
        "- selector/categorical: return only the requested label/value in "
        "value; put the explanation in comment.\n"
        "- text/string: use a JSON string.\n"
        "- Every comment must be an evidence phrase of at most 12 words.\n\n"
        "Return a single JSON object with exactly these field keys: "
        f"{json.dumps(field_names, ensure_ascii=False)}. "
        "Each field key must map to its own object containing both value and "
        "comment. The top level must contain field names only; never put "
        "value or comment directly at the top level. Follow this exact layout "
        "(replace every placeholder with an actual prediction and evidence): "
        f"{json.dumps(output_shape, ensure_ascii=False)}"
    )
