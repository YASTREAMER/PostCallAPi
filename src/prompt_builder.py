import json
import re

from schemas import ExtractRequest

SYSTEM_INSTRUCTION = (
    "You are a call-analysis extraction engine. You are given a call "
    "transcript and context, plus a list of fields to extract. Respond "
    "with ONLY a single valid JSON object mapping each field name to an "
    "object with exactly two keys: \"value\" and \"comment\". The comment "
    "must be a concise explanation citing the relevant call evidence for "
    "that value, not hidden reasoning or a step-by-step analysis. No prose "
    "outside the JSON and no markdown fences."
)

SHORT_DESC_MAX_CHARS = 220


def _shorten_description(description: str) -> str:
    """Match the description shortening used to build the fine-tuning data."""
    description = (description or "").strip()
    first_sentence = (
        re.split(r"(?<=[.!?])\s+", description)[0] if description else ""
    )
    shortened = first_sentence[:SHORT_DESC_MAX_CHARS].strip()
    if not shortened:
        shortened = description[:SHORT_DESC_MAX_CHARS].strip()
    return shortened or "(no description provided)"


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

    schema_lines = [
        f"- {field.name}: {_shorten_description(field.description)}"
        for field in req.postcall_data
    ]
    field_names = [field.name for field in req.postcall_data]
    context_block = "\n\n".join(context)
    schema_block = "\n".join(schema_lines)
    output_shape = {
        name: {
            "value": f"<extracted value for {name}>",
            "comment": f"<concise transcript evidence for {name}>",
        }
        for name in field_names
    }

    return (
        f"{context_block}\n\n"
        f"### Fields to extract\n{schema_block}\n\n"
        "Return a single JSON object with exactly these field keys: "
        f"{json.dumps(field_names, ensure_ascii=False)}. "
        "Each field key must map to its own object containing both value and "
        "comment. The top level must contain field names only; never put "
        "value or comment directly at the top level. Follow this exact layout "
        "(replace every placeholder with an actual prediction and evidence): "
        f"{json.dumps(output_shape, ensure_ascii=False)}"
    )
