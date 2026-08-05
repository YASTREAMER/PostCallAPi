import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, FastAPI, HTTPException

from api_auth import require_api_key
import model_service
from normalize_output import normalize_model_output
from runtime_config import (
    API_HOST,
    API_PORT,
    API_PREFIX,
    MAX_ACTIVE_REQUESTS,
    MAX_GENERATION_RETRIES,
)
from schemas import (
    ExtractRequest,
    ExtractResponse,
    Usage,
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
api = APIRouter(
    prefix=API_PREFIX,
    dependencies=[Depends(require_api_key)],
)

ACTIVE_REQUESTS = 0


@api.get("")
def api_root() -> dict:
    return {
        "name": "Postcall Extraction API",
        "health": f"{API_PREFIX}/health",
        "docs": f"{API_PREFIX}/docs",
    }


@api.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "active_requests": ACTIVE_REQUESTS,
        "max_active_requests": MAX_ACTIVE_REQUESTS,
    }


@api.post(
    "/extract",
    response_model=ExtractResponse,
    response_model_exclude_none=True,
)
async def extract(req: ExtractRequest) -> ExtractResponse:
    global ACTIVE_REQUESTS

    if ACTIVE_REQUESTS >= MAX_ACTIVE_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail="Server inference queue is full; retry later",
            headers={"Retry-After": "5"},
        )
    ACTIVE_REQUESTS += 1
    request_id = str(uuid.uuid4())
    try:
        postcall_data = req.resolved_postcall_data()
        messages = [message.model_dump() for message in req.messages]
        output = await model_service.generate_extraction(
            messages,
            request_id=request_id,
            max_retries=MAX_GENERATION_RETRIES,
            expected_fields=[field.name for field in postcall_data],
            json_schema=(
                req.response_format.json_schema["schema"]
                if req.response_format is not None
                else None
            ),
            temperature=req.temperature,
        )
        result = normalize_model_output(output["result"], postcall_data)
        performance = output.get("performance") or {}
        usage = Usage(
            prompt_tokens=performance.get("prompt_tokens", 0),
            completion_tokens=performance.get("completion_tokens", 0),
            total_tokens=performance.get("total_tokens", 0),
        )
        return ExtractResponse(
            result=result,
            usage=usage,
            performance=(performance if req.include_performance else None),
        )
    except Exception as exc:
        logger.exception("Request %s failed", request_id)
        raise HTTPException(
            status_code=500,
            detail="Model generation failed",
        ) from exc
    finally:
        ACTIVE_REQUESTS -= 1


app.include_router(api)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=API_HOST, port=API_PORT)
