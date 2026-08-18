from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from runtime_config import COMMENT_MAX_CHARS


SCHEMA_TYPE_ALIASES = {
    "bool": "boolean",
    "integer": "number",
    "float": "number",
    "str": "string",
    "call_drop_off_reason": "boolean",
}


def normalize_schema_type(value: Any) -> str:
    normalized = str(value or "text").strip().casefold()
    return SCHEMA_TYPE_ALIASES.get(normalized, normalized)


class FunctionCall(BaseModel):
    name: str
    parameters: Optional[Dict[str, Any]] = None
    response: Optional[Any] = None
    success: bool = False
    timestamp: Optional[str] = None


class VariableSpec(BaseModel):
    name: str
    description: str
    type: str = "text"
    defaultValue: Optional[Any] = None
    defaultValueConfig: Optional[Dict[str, Any]] = None

    @field_validator("name")
    @classmethod
    def strip_field_name(cls, value: str) -> str:
        return value.strip()

    @field_validator("type")
    @classmethod
    def normalize_field_type(cls, value: str) -> str:
        return normalize_schema_type(value)


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def reject_blank_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message content must not be blank")
        return value


class ResponseFormat(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["json_schema"]
    json_schema: Dict[str, Any]


def _variables_from_response_format(
    response_format: ResponseFormat,
) -> List[VariableSpec]:
    schema = response_format.json_schema.get("schema")
    if not isinstance(schema, dict):
        raise ValueError("response_format.json_schema.schema must be an object")
    if schema.get("type") != "object":
        raise ValueError(
            "response_format.json_schema.schema.type must be 'object'"
        )

    properties = schema.get("properties")
    if not isinstance(properties, dict) or not properties:
        raise ValueError(
            "response_format.json_schema.schema.properties must be a "
            "non-empty object"
        )

    variables = []
    for name, field_schema in properties.items():
        if not isinstance(field_schema, dict):
            raise ValueError(f"JSON schema field {name!r} must be an object")
        nested_properties = field_schema.get("properties")
        if not isinstance(nested_properties, dict):
            raise ValueError(
                f"JSON schema field {name!r} must define properties"
            )
        value_schema = nested_properties.get("value")
        comment_schema = nested_properties.get("comment")
        if not isinstance(value_schema, dict):
            raise ValueError(
                f"JSON schema field {name!r} must define value"
            )
        if not isinstance(comment_schema, dict):
            raise ValueError(
                f"JSON schema field {name!r} must define comment"
            )

        raw_type = value_schema.get("type", "string")
        if isinstance(raw_type, list):
            raw_type = next(
                (item for item in raw_type if item != "null"),
                "string",
            )
        variable_data = {
            "name": str(name),
            "description": str(field_schema.get("description") or ""),
            "type": raw_type,
        }
        if "default" in value_schema:
            variable_data["defaultValue"] = value_schema["default"]
        variables.append(VariableSpec.model_validate(variable_data))

    return variables


def normalize_response_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Repairs structural gaps callers commonly leave out of a value/comment
    JSON schema (missing "type": "object", missing "required") before it is
    handed to the model's structured-output grammar. Without "required",
    the grammar treats "value"/"comment" as optional, letting the model
    satisfy the schema with a partial (or unboundedly long) field object.
    """
    schema = dict(schema)
    schema["type"] = "object"

    properties = dict(schema.get("properties") or {})
    for name, field_schema in properties.items():
        if not isinstance(field_schema, dict):
            continue
        field_schema = dict(field_schema)
        field_schema["type"] = "object"

        nested = dict(field_schema.get("properties") or {})
        value_schema = dict(nested.get("value") or {"type": "string"})
        comment_schema = dict(nested.get("comment") or {"type": "string"})
        comment_schema.setdefault("maxLength", COMMENT_MAX_CHARS)
        nested["value"] = value_schema
        nested["comment"] = comment_schema

        field_schema["properties"] = nested
        field_schema["required"] = ["value", "comment"]
        field_schema["additionalProperties"] = False
        properties[name] = field_schema

    schema["properties"] = properties
    schema["required"] = list(properties.keys())
    schema["additionalProperties"] = False
    return schema


class ExtractRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messages: List[ChatMessage] = Field(min_length=1)
    model: Optional[str] = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    response_format: Optional[ResponseFormat] = None
    postcall_data: Optional[List[VariableSpec]] = None
    include_performance: bool = False

    @model_validator(mode="after")
    def validate_extraction_schema(self) -> "ExtractRequest":
        if self.response_format is None and self.postcall_data is None:
            raise ValueError(
                "either response_format or postcall_data must be provided"
            )
        if self.response_format is not None:
            schema_fields = _variables_from_response_format(
                self.response_format
            )
            if self.postcall_data is not None:
                schema_names = [field.name for field in schema_fields]
                legacy_names = [field.name for field in self.postcall_data]
                if schema_names != legacy_names:
                    raise ValueError(
                        "response_format and postcall_data field names "
                        "must match"
                    )
        return self

    def resolved_postcall_data(self) -> List[VariableSpec]:
        if self.response_format is not None:
            return _variables_from_response_format(self.response_format)
        return list(self.postcall_data or [])


class PromptBuildRequest(BaseModel):
    """Structured context used by offline tools to build their own messages."""

    postcall_data: List[VariableSpec]
    transcription: str
    call_duration: Optional[float] = None
    hangup_reason: Optional[str] = ""
    functions_called: List[FunctionCall] = Field(default_factory=list)
    call_metadata: Dict[str, Any] = Field(default_factory=dict)
    timezone: str = "Asia/Kolkata"


class ChatCompletionMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str
    refusal: Optional[str] = None
    annotations: List[Any] = Field(default_factory=list)


class Choice(BaseModel):
    index: int = 0
    finish_reason: Optional[str] = None
    logprobs: Optional[Any] = None
    content_filter_results: Dict[str, Any] = Field(default_factory=dict)
    message: ChatCompletionMessage


class PromptTokensDetails(BaseModel):
    audio_tokens: int = 0
    cached_tokens: int = 0


class CompletionTokensDetails(BaseModel):
    accepted_prediction_tokens: int = 0
    audio_tokens: int = 0
    reasoning_tokens: int = 0
    rejected_prediction_tokens: int = 0


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: PromptTokensDetails = Field(
        default_factory=PromptTokensDetails
    )
    completion_tokens_details: CompletionTokensDetails = Field(
        default_factory=CompletionTokensDetails
    )
    latency_checkpoint: Optional[Dict[str, Any]] = None


class PromptFilterResult(BaseModel):
    prompt_index: int = 0
    content_filter_results: Dict[str, Any] = Field(default_factory=dict)


class ExtractResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage
    system_fingerprint: Optional[str] = None
    service_tier: str = "default"
    prompt_filter_results: List[PromptFilterResult] = Field(default_factory=list)
