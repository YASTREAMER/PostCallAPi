from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


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


class ExtractRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messages: List[ChatMessage] = Field(min_length=1)
    postcall_data: List[VariableSpec]
    include_performance: bool = False


class PromptBuildRequest(BaseModel):
    """Structured context used by offline tools to build their own messages."""

    postcall_data: List[VariableSpec]
    transcription: str
    call_duration: Optional[float] = None
    hangup_reason: Optional[str] = ""
    functions_called: List[FunctionCall] = Field(default_factory=list)
    call_metadata: Dict[str, Any] = Field(default_factory=dict)
    timezone: str = "Asia/Kolkata"


class ExtractResponse(BaseModel):
    result: Dict[str, Any]
    performance: Optional[Dict[str, Any]] = None
