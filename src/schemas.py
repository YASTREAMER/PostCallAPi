from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


SCHEMA_TYPE_ALIASES = {
    "bool": "boolean",
    "integer": "number",
    "float": "number",
    "str": "string",
    # This dataset uses call_drop_off_reason for one-of-many boolean flags.
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
    """One field requested from the post-call extraction model."""

    name: str
    description: str
    type: str = "text"
    defaultValue: Optional[Any] = None
    defaultValueConfig: Optional[Dict[str, Any]] = None

    @field_validator("name")
    @classmethod
    def strip_field_name(cls, value: str) -> str:
        """Prevent invisible CSV/schema whitespace from creating new fields."""
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

    # Optional and intended only for evaluation clients.
    ground_truth: Optional[Dict[str, Any]] = None


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
    eval_result: Optional[Dict[str, Any]] = None
    performance: Optional[Dict[str, Any]] = None


class ExtractAcceptedResponse(BaseModel):
    job_id: str
    status: JobStatus


class StatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    eval_result: Optional[Dict[str, Any]] = None
    performance: Optional[Dict[str, Any]] = None
