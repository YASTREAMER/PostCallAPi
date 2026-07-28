from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class FunctionCall(BaseModel):
    name: str
    parameters: Optional[Dict[str, Any]] = None
    response: Optional[Any] = None
    success: bool = False
    timestamp: Optional[str] = None


class VariableSpec(BaseModel):
    """One entry in the postcall_data schema, e.g.
    {"name": "Salary_amount", "description": "...", "type": "text"}
    """
    name: str
    description: str
    type: str = "text"
    defaultValue: Optional[Any] = None
    defaultValueConfig: Optional[Dict[str, Any]] = None


class ExtractRequest(BaseModel):
    postcall_data: List[VariableSpec]

    transcription: str
    call_duration: Optional[float] = None
    hangup_reason: Optional[str] = ""
    functions_called: List[FunctionCall] = Field(default_factory=list)
    call_metadata: Dict[str, Any] = Field(default_factory=dict)

    timezone: str = "Asia/Kolkata"

    # Optional — only sent during testing. If present, the job scores
    # its own output against this and logs the result.
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


class ExtractAcceptedResponse(BaseModel):
    job_id: str
    status: JobStatus


class StatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    eval_result: Optional[Dict[str, Any]] = None
