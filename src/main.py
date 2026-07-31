import asyncio
import logging
import time
import uuid
from collections import Counter
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI, HTTPException

import model_service
from normalize_output import normalize_model_output
from prompt_builder import build_prompt
from runtime_config import (
    API_HOST,
    API_PORT,
    API_PREFIX,
    JOB_TTL_SECONDS,
    MAX_ACTIVE_JOBS,
    MAX_COMPLETED_JOBS,
    MAX_GENERATION_RETRIES,
)
from schemas import (
    ExtractAcceptedResponse,
    ExtractRequest,
    JobRecord,
    JobStatus,
    StatusResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await model_service.load_model()
        yield
    finally:
        model_service.shutdown_model()


app = FastAPI(
    title="Postcall Extraction API",
    lifespan=lifespan,
    docs_url=f"{API_PREFIX}/docs",
    openapi_url=f"{API_PREFIX}/openapi.json",
    redoc_url=f"{API_PREFIX}/redoc",
    swagger_ui_oauth2_redirect_url=f"{API_PREFIX}/docs/oauth2-redirect",
)
api = APIRouter(prefix=API_PREFIX)

JOBS: dict[str, JobRecord] = {}
JOB_FINISHED_AT: dict[str, float] = {}


def _remove_job(job_id: str) -> None:
    JOBS.pop(job_id, None)
    JOB_FINISHED_AT.pop(job_id, None)


def _cleanup_jobs() -> None:
    now = time.monotonic()
    for job_id, finished_at in list(JOB_FINISHED_AT.items()):
        if now - finished_at >= JOB_TTL_SECONDS:
            _remove_job(job_id)

    overflow = len(JOB_FINISHED_AT) - MAX_COMPLETED_JOBS
    if overflow > 0:
        oldest = sorted(JOB_FINISHED_AT, key=JOB_FINISHED_AT.get)[:overflow]
        for job_id in oldest:
            _remove_job(job_id)


@api.get("")
def api_root() -> dict:
    return {
        "name": "Postcall Extraction API",
        "health": f"{API_PREFIX}/health",
        "docs": f"{API_PREFIX}/docs",
    }


@api.get("/health")
def health() -> dict:
    _cleanup_jobs()
    counts = Counter(job.status.value for job in JOBS.values())
    return {
        "status": "ok",
        "jobs": dict(counts),
        "retained_jobs": len(JOBS),
        "max_active_jobs": MAX_ACTIVE_JOBS,
    }


@api.post("/extract", response_model=ExtractAcceptedResponse)
async def extract(req: ExtractRequest):
    _cleanup_jobs()
    active_jobs = sum(
        job.status in {JobStatus.PENDING, JobStatus.PROCESSING}
        for job in JOBS.values()
    )
    if active_jobs >= MAX_ACTIVE_JOBS:
        raise HTTPException(
            status_code=429,
            detail="Server inference queue is full; retry later",
            headers={"Retry-After": "5"},
        )
    job_id = str(uuid.uuid4())
    JOBS[job_id] = JobRecord(job_id=job_id, status=JobStatus.PENDING)
    asyncio.create_task(_process_job(job_id, req))
    return ExtractAcceptedResponse(job_id=job_id, status=JobStatus.PENDING)


async def _process_job(job_id: str, req: ExtractRequest) -> None:
    job = JOBS[job_id]
    job.status = JobStatus.PROCESSING
    try:
        prompt = build_prompt(req)
        output = await model_service.generate_extraction(
            prompt,
            request_id=job_id,
            max_retries=MAX_GENERATION_RETRIES,
            expected_fields=[field.name for field in req.postcall_data],
        )
        result = normalize_model_output(output["result"], req.postcall_data)
        job.result = result
        job.raw_thinking = output.get("thinking")
        job.performance = output.get("performance")

        job.status = JobStatus.DONE
    except Exception as exc:
        logger.exception("Job %s failed", job_id)
        job.status = JobStatus.ERROR
        job.error = str(exc)
    finally:
        JOB_FINISHED_AT[job_id] = time.monotonic()
        _cleanup_jobs()


@api.get("/status/{job_id}", response_model=StatusResponse)
def get_status(job_id: str):
    _cleanup_jobs()
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job_id not found or expired")

    return StatusResponse(
        job_id=job.job_id,
        status=job.status,
        result=job.result,
        error=job.error,
        performance=job.performance,
    )


app.include_router(api)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=API_HOST, port=API_PORT)
