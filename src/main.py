import asyncio
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

import evaluation
import model_service
from normalize_output import normalize_model_output
from prompt_builder import build_prompt
from runtime_config import API_HOST, API_PORT
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
    """Initialize the shared model engine when the API server starts."""
    try:
        await model_service.load_model()
        yield
    finally:
        model_service.shutdown_model()


app = FastAPI(title="Postcall Extraction API", lifespan=lifespan)

JOBS: dict[str, JobRecord] = {}


@app.post("/extract", response_model=ExtractAcceptedResponse)
async def extract(req: ExtractRequest):
    job_id = str(uuid.uuid4())
    JOBS[job_id] = JobRecord(job_id=job_id, status=JobStatus.PENDING)

    asyncio.create_task(_process_job(job_id, req))

    return ExtractAcceptedResponse(job_id=job_id, status=JobStatus.PENDING)


async def _process_job(job_id: str, req: ExtractRequest):
    JOBS[job_id].status = JobStatus.PROCESSING
    try:
        prompt = build_prompt(req)
        output = await model_service.generate_extraction(
            prompt,
            request_id=job_id,
            max_retries=2,
            expected_fields=[field.name for field in req.postcall_data],
        )
        result = normalize_model_output(output["result"], req.postcall_data)
        JOBS[job_id].result = result
        JOBS[job_id].raw_thinking = output["thinking"]

        if req.ground_truth:
            schema_names = {v.name for v in req.postcall_data}
            spec_type_by_name = {v.name: v.type for v in req.postcall_data}
            eval_result = evaluation.evaluate_output(
                schema_names, spec_type_by_name, result, req.ground_truth
            )
            JOBS[job_id].eval_result = eval_result
            evaluation.log_evaluation(job_id, result, eval_result)

        JOBS[job_id].status = JobStatus.DONE
    except Exception as e:
        logger.exception("Job %s failed", job_id)
        JOBS[job_id].status = JobStatus.ERROR
        JOBS[job_id].error = str(e)


@app.get("/status/{job_id}", response_model=StatusResponse)
def get_status(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job_id not found")

    return StatusResponse(
        job_id=job.job_id,
        status=job.status,
        result=job.result,
        error=job.error,
        eval_result=job.eval_result,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=API_HOST, port=API_PORT)
