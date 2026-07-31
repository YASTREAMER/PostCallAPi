from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


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


class ExtractRequest(BaseModel):
    postcall_data: List[VariableSpec]

    transcription: str
    call_duration: Optional[float] = None
    hangup_reason: Optional[str] = ""
    functions_called: List[FunctionCall] = Field(default_factory=list)
    call_metadata: Dict[str, Any] = Field(default_factory=dict)

    timezone: str = "Asia/Kolkata"


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    ERROR = "error"


class JobRecord(BaseModel):
    job_id: str
    status: JobStatus
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    raw_thinking: Optional[str] = None
    performance: Optional[Dict[str, Any]] = None


class ExtractAcceptedResponse(BaseModel):
    job_id: str
    status: JobStatus


class StatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    performance: Optional[Dict[str, Any]] = None
