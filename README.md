# PostCall API

PostCall API is an asynchronous post-call extraction service backed by vLLM,
Qwen3, and a fine-tuned LoRA adapter. It accepts a call transcript plus a
caller-defined extraction schema and returns normalized, typed JSON values with
short evidence comments.

This document covers installation, model setup, running the API on a private
company server, calling the endpoints, concurrency and batching, configuration,
benchmarking, offline evaluation, monitoring, and troubleshooting.

## Table of contents

- [How the service works](#how-the-service-works)
- [Requirements](#requirements)
- [Project layout](#project-layout)
- [Installation](#installation)
- [Model and adapter setup](#model-and-adapter-setup)
- [Configuration](#configuration)
- [Run the API](#run-the-api)
- [Private production deployment](#private-production-deployment)
- [API reference](#api-reference)
- [Batching, concurrency, and capacity](#batching-concurrency-and-capacity)
- [Smoke test](#smoke-test)
- [Benchmark the live API](#benchmark-the-live-api)
- [Offline batch evaluation](#offline-batch-evaluation)
- [Prepare the evaluation CSV](#prepare-the-evaluation-csv)
- [Performance tuning](#performance-tuning)
- [Operations and monitoring](#operations-and-monitoring)
- [Security and data handling](#security-and-data-handling)
- [Troubleshooting](#troubleshooting)

## How the service works

The service uses one shared `AsyncLLMEngine` and submits every accepted
extraction job to that engine:

```text
Company client
    |
    | POST /extract
    v
FastAPI validates the request and returns a job_id
    |
    | background task calls AsyncLLMEngine.generate(...)
    v
vLLM continuously batches active requests on the GPU
    |
    | result is parsed, validated, and normalized
    v
Client polls GET /status/{job_id}
```

Important characteristics:

- `POST /extract` is asynchronous. It returns a job ID instead of waiting for
  inference to finish.
- vLLM performs continuous dynamic batching. The API does not create fixed
  request batches and does not wait for a batch to fill.
- Multiple concurrent requests can be processed together by vLLM.
- Each requested output field is normalized according to its configured type.
- Jobs and results are stored in process memory. Restarting the API removes all
  pending and completed job records.
- The service is intended to run as one Uvicorn worker per GPU. Multiple workers
  on the same GPU each load their own model and can exhaust GPU memory.

The model configuration currently uses:

- Base model: `unsloth/Qwen3-14B-unsloth-bnb-4bit`
- Quantization/load format: BitsAndBytes
- Compute dtype: `bfloat16`
- LoRA adapter name: `postcall-adapter`
- Maximum LoRA rank: `16`
- Default model context length: `16,384` tokens
- Thinking mode: disabled
- Prefix caching: enabled

## Requirements

The pinned project environment expects:

- Linux
- Python `>=3.12,<3.13`
- Poetry
- An NVIDIA GPU supported by the installed PyTorch, CUDA, BitsAndBytes, and
  vLLM versions
- Enough GPU memory for the 14B 4-bit base model, LoRA adapter, and vLLM KV
  cache
- Network access to Hugging Face on the first model download, unless the model
  is already present in the local Hugging Face cache
- The fine-tuned adapter files described below

The project pins vLLM `0.10.1` and Transformers `4.55.0`. Treat the lock file as
the source of truth when creating the environment.

## Project layout

```text
PostCallAPi/
├── adapter/                    # LoRA adapter; required and gitignored
├── data/                       # Evaluation CSV files; gitignored
├── output/                     # Live API benchmark reports; gitignored
├── src/
│   ├── main.py                 # FastAPI application and in-memory job store
│   ├── model_service.py        # vLLM engine and generation logic
│   ├── runtime_config.py       # Environment-based runtime configuration
│   ├── schemas.py              # Request, response, and field schemas
│   ├── prompt_builder.py       # Extraction prompt construction
│   ├── normalize_output.py     # Typed output normalization
│   ├── evaluation.py           # Shared scoring functions
│   ├── api_csv_test.py         # Concurrent live API benchmark
│   ├── evaluator.py            # Offline vLLM batch evaluator
│   └── CSVCreator.py           # Evaluation CSV schema preparation
├── pyproject.toml
├── poetry.lock
└── requirements.txt
```

## Installation

Run the following commands from the project root:

```bash
poetry env use python3.12
poetry install
```

Confirm that Poetry selected a Python 3.12 environment:

```bash
poetry run python --version
```

The expected output is a Python 3.12 release.

Confirm that PyTorch can see the GPU:

```bash
poetry run python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

The first value should be `True`. If it is `False`, verify the NVIDIA driver,
CUDA/PyTorch compatibility, and the environment in which the service is
started.

## Model and adapter setup

The adapter directory is required at startup and must contain:

```text
adapter/
├── adapter_config.json
└── adapter_model.safetensors
```

The startup validation checks that:

- both files exist;
- `base_model_name_or_path` in `adapter_config.json` equals
  `unsloth/Qwen3-14B-unsloth-bnb-4bit`; and
- the adapter rank is `16`.

The `adapter/` directory is intentionally gitignored. Copy the approved adapter
artifact onto the server as part of deployment.

The base model is loaded from Hugging Face by model ID. On first startup, allow
time for the download. For a controlled production environment, pre-download
the approved model artifact into the server cache so a restart does not depend
on external network availability.

## Configuration

Runtime settings are read from environment variables when the Python process
starts. Restart the API after changing them.

### vLLM and generation settings

| Variable | Default | Meaning |
|---|---:|---|
| `VLLM_MAX_MODEL_LEN` | `16384` | Maximum combined prompt and generation context used by the engine. |
| `VLLM_MAX_NEW_TOKENS` | `8192` | Absolute ceiling for generated tokens per attempt. |
| `VLLM_MIN_NEW_TOKENS` | `1024` | Floor used when calculating the per-request generation ceiling. It does not force the model to generate this many tokens. |
| `VLLM_TOKENS_PER_FIELD` | `64` | Additional generation allowance assigned per requested field. |
| `VLLM_MAX_GENERATION_RETRIES` | `1` | Additional generation attempts after invalid JSON or an invalid output shape. A value of `1` allows at most two total attempts. |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.5` | Fraction of GPU memory vLLM may reserve. Must be greater than `0` and at most `1`. |
| `VLLM_ENABLE_PREFIX_CACHING` | `true` | Enables vLLM prefix caching. Accepted true values are `1`, `true`, `yes`, and `on`. |
| `POSTCALL_ENABLE_THINKING` | `false` | Enables the Qwen chat template's thinking mode. Keep disabled for extraction unless accuracy tests demonstrate a benefit. |

For a request containing fields, the generation ceiling is calculated as:

```text
min(
    VLLM_MAX_NEW_TOKENS,
    max(
        VLLM_MIN_NEW_TOKENS,
        512 + VLLM_TOKENS_PER_FIELD * number_of_fields
    )
)
```

The actual generation can end earlier when the model emits its end token.

### API and job settings

| Variable | Default | Meaning |
|---|---:|---|
| `POSTCALL_API_HOST` | `0.0.0.0` | Bind address used by `python src/main.py`. |
| `POSTCALL_API_PORT` | `8000` | Port used by `python src/main.py`. |
| `POSTCALL_MAX_ACTIVE_JOBS` | `100` | Maximum total `pending` plus `processing` jobs accepted by this process. |
| `POSTCALL_JOB_TTL_SECONDS` | `3600` | Time a completed job remains eligible for retrieval. |
| `POSTCALL_MAX_COMPLETED_JOBS` | `10000` | Maximum completed job records retained in process memory. |

When the active-job limit is reached, `POST /extract` returns HTTP `429` with a
`Retry-After: 5` header.

`POSTCALL_API_HOST` and `POSTCALL_API_PORT` apply when starting
`src/main.py`. When using the Uvicorn CLI, the CLI `--host` and `--port`
arguments control the listener.

### Example environment

For a dedicated internal GPU server:

```bash
export VLLM_GPU_MEMORY_UTILIZATION=0.85
export VLLM_MAX_MODEL_LEN=16384
export VLLM_MAX_NEW_TOKENS=4096
export VLLM_MIN_NEW_TOKENS=512
export VLLM_TOKENS_PER_FIELD=64
export VLLM_MAX_GENERATION_RETRIES=1
export VLLM_ENABLE_PREFIX_CACHING=true
export POSTCALL_ENABLE_THINKING=false
export POSTCALL_MAX_ACTIVE_JOBS=100
export POSTCALL_JOB_TTL_SECONDS=3600
export POSTCALL_MAX_COMPLETED_JOBS=10000
```

These are starting values, not universal optimal values. Measure throughput,
latency, output quality, GPU memory, and preemption under representative load
before selecting production values.

## Run the API

### Recommended development command

From the project root:

```bash
poetry run uvicorn main:app \
  --app-dir src \
  --host 127.0.0.1 \
  --port 8000
```

### Listen on the private server interface

Use this only when the server is protected by the company's private network,
VPN, firewall, or internal load balancer:

```bash
poetry run uvicorn main:app \
  --app-dir src \
  --host 0.0.0.0 \
  --port 8000
```

Alternatively:

```bash
POSTCALL_API_HOST=0.0.0.0 \
POSTCALL_API_PORT=8000 \
poetry run python src/main.py
```

Startup is complete only after the log contains:

```text
Model loaded.
```

Model loading can take several minutes on the first run because the base model
may need to be downloaded and loaded onto the GPU. Do not begin a benchmark
until startup has completed.

Check health:

```bash
curl http://127.0.0.1:8000/health
```

FastAPI also exposes interactive documentation at `/docs` and the OpenAPI
schema at `/openapi.json`. Keep these endpoints on the private network.

## Private production deployment

This API currently has no built-in authentication. Because only the company
should access it, deploy it on a private subnet or behind a company-controlled
gateway. Do not expose port 8000 directly to the public Internet.

Recommended topology:

```text
Company network or VPN
        |
        v
Internal gateway / load balancer
  - company authentication
  - TLS
  - request-size limit
  - per-client rate limit
        |
        v
PostCall API on private GPU server
  - one Uvicorn worker
  - one vLLM engine
  - one GPU
```

Production rules:

- Run exactly one Uvicorn worker for each model/GPU instance.
- Do not use `--workers 2` or a process manager that silently creates multiple
  application workers on one GPU.
- Restrict inbound traffic to the company network, VPN, service mesh, or
  internal load balancer.
- Apply authentication and authorization at the internal gateway even if the
  server has no public address.
- Set request-body and rate limits at the gateway.
- Use TLS because call transcripts and extracted results can contain sensitive
  company or customer data.
- Do not enable Uvicorn `--reload` in production.
- Keep the model cache and adapter readable only by the service account.
- Ensure logs, evaluation artifacts, and benchmark output follow the company's
  data-retention rules.

### Example systemd service

The following is a template. Adjust user names and paths for the server:

```ini
[Unit]
Description=PostCall vLLM API
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=postcall
Group=postcall
WorkingDirectory=/opt/PostCallAPi
EnvironmentFile=/etc/postcall-api.env
Environment=PYTHONUNBUFFERED=1
ExecStart=/usr/local/bin/poetry run uvicorn main:app --app-dir src --host 0.0.0.0 --port 8000
Restart=on-failure
RestartSec=5
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
```

Example `/etc/postcall-api.env`:

```bash
VLLM_GPU_MEMORY_UTILIZATION=0.85
VLLM_MAX_MODEL_LEN=16384
VLLM_MAX_NEW_TOKENS=4096
VLLM_MIN_NEW_TOKENS=512
VLLM_TOKENS_PER_FIELD=64
VLLM_MAX_GENERATION_RETRIES=1
VLLM_ENABLE_PREFIX_CACHING=true
POSTCALL_ENABLE_THINKING=false
POSTCALL_MAX_ACTIVE_JOBS=100
POSTCALL_JOB_TTL_SECONDS=3600
POSTCALL_MAX_COMPLETED_JOBS=10000
```

After installing the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now postcall-api
sudo systemctl status postcall-api
sudo journalctl -u postcall-api -f
```

If the `poetry` binary is installed elsewhere, replace
`/usr/local/bin/poetry` with the result of `command -v poetry`.

### Multiple GPUs or replicas

Run one API process per GPU and place an internal load balancer in front of
them. The current job store is local process memory, so the status request for a
job must return to the same replica that accepted it. Use sticky routing based
on a suitable routing mechanism, or move the job store to shared infrastructure
before deploying multiple replicas.

Assign a single visible GPU and a different port to each service instance, for
example:

```ini
Environment=CUDA_VISIBLE_DEVICES=0
ExecStart=/usr/local/bin/poetry run uvicorn main:app --app-dir src --host 0.0.0.0 --port 8000
```

The next instance can use physical GPU `1` and another private port. Because
`CUDA_VISIBLE_DEVICES` remaps visible device numbering, each isolated process
still sees its assigned GPU as device `0`.

For strong production durability, use an external job store such as Redis or a
database. The current implementation loses job state during process restarts
and deployments.

## API reference

The base URL in the examples is:

```text
http://127.0.0.1:8000
```

Replace it with the internal company endpoint when calling the deployed API.

### `GET /health`

Returns service and in-memory job counts.

Example:

```bash
curl http://127.0.0.1:8000/health
```

Response:

```json
{
  "status": "ok",
  "jobs": {
    "processing": 3,
    "done": 27
  },
  "retained_jobs": 30,
  "max_active_jobs": 100
}
```

The health response confirms that the application is running. It does not
perform a new model generation.

### `POST /extract`

Accepts an extraction request, creates a background job, and returns its ID.

Example:

```bash
curl -X POST http://127.0.0.1:8000/extract \
  -H 'Content-Type: application/json' \
  -d '{
    "postcall_data": [
      {
        "name": "customer_interested",
        "description": "True when the customer clearly expresses interest in the offer.",
        "type": "boolean",
        "defaultValue": false
      },
      {
        "name": "disposition_reason",
        "description": "Briefly state the final call outcome.",
        "type": "text",
        "defaultValue": ""
      }
    ],
    "transcription": "Agent: Would you like a demo tomorrow? Customer: Yes, please schedule it for 3 PM.",
    "call_duration": 42.5,
    "hangup_reason": "customer-ended-call",
    "functions_called": [
      {
        "name": "schedule_demo",
        "parameters": {
          "time": "15:00"
        },
        "response": {
          "scheduled": true
        },
        "success": true,
        "timestamp": "2026-07-30T10:00:00Z"
      }
    ],
    "call_metadata": {
      "call_id": "company-call-123",
      "campaign": "demo-campaign"
    },
    "timezone": "Asia/Kolkata"
  }'
```

Accepted response:

```json
{
  "job_id": "dc37a7a1-d24b-41b8-b0f9-82a0ad6eb25f",
  "status": "pending"
}
```

The current endpoint returns HTTP `200` when the job is accepted.

#### Request fields

| Field | Required | Description |
|---|---|---|
| `postcall_data` | Yes | List of fields the model must extract. |
| `transcription` | Yes | Call transcript supplied to the model. |
| `call_duration` | No | Call duration as a number. |
| `hangup_reason` | No | Call termination reason. Defaults to an empty string. |
| `functions_called` | No | Functions or tools called during the conversation. Defaults to an empty list. |
| `call_metadata` | No | Additional JSON context. Defaults to an empty object. |
| `timezone` | No | Context timezone. Defaults to `Asia/Kolkata`. |
| `ground_truth` | No | Evaluation-only expected output. Omit in normal production traffic. |

Each `postcall_data` item supports:

| Field | Required | Description |
|---|---|---|
| `name` | Yes | Output key. Leading and trailing whitespace is removed. |
| `description` | Yes | Extraction instructions for the field. |
| `type` | No | Expected output type. Defaults to `text`. |
| `defaultValue` | No | Value used when an appropriate model value is unavailable. |
| `defaultValueConfig` | No | Optional metadata accepted by the request schema. |

Supported normalized type behavior:

- `boolean`: output is normalized to JSON `true` or `false`.
- `number`: output is normalized to an integer or floating-point number.
- `text`, `string`, `selector`, and `categorical`: output is normalized to a
  JSON string.
- Aliases: `bool` becomes `boolean`; `integer` and `float` become `number`;
  `str` becomes `string`.

Every result field has this shape:

```json
{
  "value": "typed value",
  "comment": "short supporting evidence"
}
```

### `GET /status/{job_id}`

Returns the current state and, when available, the result.

Example:

```bash
curl http://127.0.0.1:8000/status/dc37a7a1-d24b-41b8-b0f9-82a0ad6eb25f
```

Possible statuses:

- `pending`: accepted but the background task has not started processing.
- `processing`: submitted to the inference path; it may be running or waiting
  in the vLLM scheduler.
- `done`: extraction completed successfully.
- `error`: generation, parsing, validation, or normalization failed.

Completed response:

```json
{
  "job_id": "dc37a7a1-d24b-41b8-b0f9-82a0ad6eb25f",
  "status": "done",
  "result": {
    "customer_interested": {
      "value": true,
      "comment": "Customer agreed to schedule a demo."
    },
    "disposition_reason": {
      "value": "Demo requested for 3 PM",
      "comment": "Customer accepted the proposed demo."
    }
  },
  "error": null,
  "eval_result": null,
  "performance": {
    "attempts": 1,
    "retried": false,
    "generation_seconds": 2.417,
    "prompt_tokens": 486,
    "completion_tokens": 61,
    "attempt_details": [
      {
        "generation_seconds": 2.417,
        "prompt_tokens": 486,
        "completion_tokens": 61,
        "max_tokens": 1024,
        "finish_reason": "stop",
        "attempt": 1
      }
    ]
  }
}
```

Error response:

```json
{
  "job_id": "dc37a7a1-d24b-41b8-b0f9-82a0ad6eb25f",
  "status": "error",
  "result": null,
  "error": "Error description",
  "eval_result": null,
  "performance": null
}
```

An unknown or expired job ID returns HTTP `404`.

### Evaluation-only `ground_truth`

When `ground_truth` is supplied, the server evaluates the completed result and
returns `eval_result`. It also appends evaluation output to:

```text
src/eval_log/eval_logs.jsonl
```

This feature is intended for controlled evaluation, not normal production
requests. The live benchmark evaluates ground truth on the benchmark client and
does not need to send it to the API.

## Batching, concurrency, and capacity

### Continuous batching

The API uses `AsyncLLMEngine`. Each accepted request calls `generate`
independently, and vLLM combines active sequences dynamically during inference.
There is no fixed API batch size.

For example, with 50 concurrent client requests:

- the API creates up to 50 active generation jobs;
- vLLM schedules those sequences according to available GPU compute, KV cache,
  prompt lengths, and generation lengths;
- new requests can join later scheduler iterations;
- completed requests leave immediately; and
- requests that cannot run in the current iteration remain queued.

The engine does not wait for all 50 requests before starting.

### The three different concurrency controls

Do not confuse these settings:

| Control | What it controls |
|---|---|
| Benchmark `--concurrency` | How many requests the benchmark client keeps in flight. |
| `POSTCALL_MAX_ACTIVE_JOBS` | How many pending plus processing jobs this API process accepts. |
| vLLM scheduler | How many sequences/tokens are actually processed during each GPU iteration. |

The current engine configuration does not set `max_num_seqs` or
`max_num_batched_tokens`; vLLM selects its version- and context-dependent
defaults.

This is usually the right production starting point because traffic is not
constant. Add a scheduler ceiling only if load testing shows GPU
out-of-memory errors, excessive KV-cache preemption, or unacceptable tail
latency. A ceiling still preserves dynamic batching; it does not create a fixed
batch.

### Behavior under overload

If active jobs reach `POSTCALL_MAX_ACTIVE_JOBS`, new submissions receive:

```text
HTTP 429 Too Many Requests
Retry-After: 5
```

Company clients should retry with exponential backoff and jitter. Do not retry
immediately in a tight loop.

## Smoke test

Start with one small request before running a concurrent benchmark.

Submit:

```bash
RESPONSE=$(curl -sS -X POST http://127.0.0.1:8000/extract \
  -H 'Content-Type: application/json' \
  -d '{
    "postcall_data": [
      {
        "name": "interested",
        "description": "Whether the customer expressed interest.",
        "type": "boolean",
        "defaultValue": false
      }
    ],
    "transcription": "Customer: Yes, I am interested. Please call tomorrow."
  }')

echo "$RESPONSE"
```

Copy the returned `job_id`, then poll:

```bash
curl http://127.0.0.1:8000/status/JOB_ID
```

Confirm that the job reaches `done`, the result contains every requested field,
and the performance object contains prompt and completion token counts.

## Benchmark the live API

`src/api_csv_test.py` sends production-shaped requests to a running API,
maintains configurable client concurrency, polls each job, compares results
with CSV ground truth locally, and saves detailed reports.

The API and benchmark may run on different machines. The benchmark machine does
not need a GPU when it is testing a remote API.

### Required CSV data

The benchmark uses these columns:

- `postcall`: JSON list containing the extraction field schema.
- `transcription`: call transcript.
- `post_call_detail`: JSON object containing ground truth.
- `call_duration`: optional numeric duration.
- `hangup_reason`: optional text.
- `functions_called`: optional JSON list.
- `call_metadata`: optional JSON object.
- `conversion_reason`: optional rule used when adding the conversion field.

The benchmark adds `conversion_status` and `disposition_reason` to the request
schema when they are missing.

### Basic benchmark

Test 200 reproducibly sampled rows with at most 10 jobs in flight:

```bash
poetry run python src/api_csv_test.py \
  --csv data/Data_with_outcome_fields.csv \
  --count 200 \
  --concurrency 10 \
  --api-url http://127.0.0.1:8000
```

Benchmark a private remote server:

```bash
poetry run python src/api_csv_test.py \
  --csv data/Data_with_outcome_fields.csv \
  --count 200 \
  --concurrency 10 \
  --api-url https://postcall-api.internal.company
```

The current benchmark client does not attach authentication headers. If the
company gateway requires authentication, extend `_request_json` in
`src/api_csv_test.py` or run the benchmark from an already authorized internal
network path.

### Load-test progression

Increase load gradually rather than starting with maximum concurrency:

```bash
# Baseline latency
poetry run python src/api_csv_test.py --count 50 --concurrency 1

# Light concurrency
poetry run python src/api_csv_test.py --count 100 --concurrency 5

# Moderate concurrency
poetry run python src/api_csv_test.py --count 200 --concurrency 10

# Higher production-style concurrency
poetry run python src/api_csv_test.py --count 500 --concurrency 25

# Stress test
poetry run python src/api_csv_test.py --count 1000 --concurrency 50
```

Keep `--concurrency` at or below `POSTCALL_MAX_ACTIVE_JOBS`, leaving capacity
for health checks and other clients.

### Benchmark options

| Option | Default | Description |
|---|---:|---|
| `--csv` | `data/Data_with_outcome_fields.csv` | Source CSV. |
| `--count` | `200` | Number of rows to test. The current code accepts `1` through `10000`. |
| `--concurrency` | `10` | Maximum number of in-flight API jobs. |
| `--selection` | `random` | `random` for seeded sampling or `first` for the first N rows. |
| `--seed` | `42` | Seed used for reproducible random sampling. |
| `--api-url` | `http://127.0.0.1:8000` | API base URL. |
| `--output-root` | `output` | Parent directory for reports. |
| `--poll-interval` | `1.0` | Seconds between status polls. |
| `--job-timeout` | `1200.0` | Maximum seconds to wait for a job. |
| `--request-timeout` | `60.0` | HTTP timeout per submit or poll request. |

Display the command help:

```bash
poetry run python src/api_csv_test.py --help
```

### Benchmark output

Each run creates:

```text
output/api_test_<timestamp>/
├── summary.json
├── rows.csv
├── field_scores.csv
└── results.jsonl
```

- `summary.json`: wall time, completed and failed counts, throughput, latency
  percentiles, retry rate, token totals, tokens per minute, strict typed
  accuracy, meaningful-value accuracy, and ground-truth coverage.
- `rows.csv`: one row per test case with status, timing, polling, token usage,
  and accuracy.
- `field_scores.csv`: per-field predictions, ground truth, schema type, type
  validity, match status, and score.
- `results.jsonl`: complete request/result/evaluation details for debugging.

During execution, the benchmark prints:

- completed row count;
- job status;
- end-to-end time;
- completion tokens;
- per-job completion tokens per minute; and
- rolling wall-clock completion tokens per minute.

### Metrics to compare

When tuning concurrency or GPU settings, compare:

- requests per minute;
- completion tokens per minute;
- p50, p95, and p99 end-to-end latency;
- generation time;
- completion token count;
- retry rate;
- failed/error jobs;
- strict type accuracy;
- meaningful-value accuracy;
- GPU utilization and memory use; and
- vLLM preemption or out-of-memory warnings.

Throughput normally increases with concurrency until the GPU is saturated.
Past that point, queueing and p95/p99 latency increase while throughput improves
little or can decline.

## Offline batch evaluation

`src/evaluator.py` loads its own offline vLLM `LLM` instance, builds prompts for
a train/test split, generates them in chunks, and produces accuracy reports.

Do not run the offline evaluator at the same time as the live API on the same
GPU. Each process creates a separate vLLM engine and independently reserves GPU
memory.

Example:

```bash
poetry run python src/evaluator.py \
  --csv data/Data_with_outcome_fields.csv \
  --test-size 0.2 \
  --random-state 42 \
  --batch-size 8 \
  --gpu-memory-utilization 0.85 \
  --out output/eval_report.csv
```

Options:

| Option | Default | Description |
|---|---:|---|
| `--csv` | Required | Evaluation CSV path. |
| `--test-size` | `0.2` | Fraction of rows selected for evaluation. |
| `--random-state` | `42` | Reproducible split seed. |
| `--out` | `eval_report.csv` | Detailed field-level report. |
| `--batch-size` | `8` | Number of prompts passed to each offline `llm.generate` call. |
| `--gpu-memory-utilization` | Shared runtime default | Fraction of GPU memory reserved by the offline engine. |
| `--log-file` | Same stem as `--out`, with `.log` | Summary and mismatch log path. |

Unlike the live API's continuous batching, `--batch-size` here controls how the
prepared prompt list is divided into explicit offline chunks.

The evaluator writes:

- the main field-level CSV report;
- a boolean metrics CSV beside the main report;
- a summary and mismatch log; and
- `evaluator_errors.log` for rows whose model output could not be parsed.

## Prepare the evaluation CSV

`src/CSVCreator.py` adds or replaces `conversion_status` and
`disposition_reason` in each row's `postcall` schema while preserving the
ground-truth column.

Using the default paths:

```bash
poetry run python src/CSVCreator.py
```

This reads:

```text
data/Data.csv
```

and writes:

```text
data/Data_with_outcome_fields.csv
```

Specify custom files:

```bash
poetry run python src/CSVCreator.py \
  --input data/source.csv \
  --output data/prepared.csv
```

The script refuses to overwrite the input file.

## Performance tuning

Tune one variable at a time and compare benchmark artifacts.

### 1. GPU memory utilization

The default `VLLM_GPU_MEMORY_UTILIZATION=0.5` is conservative. If the API owns
the GPU exclusively, test:

```text
0.70 -> 0.80 -> 0.85 -> 0.90
```

More reserved memory gives vLLM more KV-cache capacity and can improve
concurrent throughput. Leave headroom for CUDA, model loading, and temporary
allocations. Reduce the setting if startup or generation produces GPU
out-of-memory errors.

### 2. Client concurrency

Test concurrency at:

```text
1 -> 5 -> 10 -> 25 -> 50
```

Select the point where throughput is strong while p95/p99 latency remains
acceptable. Higher concurrency is not automatically faster after the GPU is
saturated.

### 3. Generation limits

Large token ceilings protect against truncated JSON but allow malformed
generations to run longer. Use benchmark completion-token distributions to
choose `VLLM_MIN_NEW_TOKENS`, `VLLM_TOKENS_PER_FIELD`, and
`VLLM_MAX_NEW_TOKENS`.

After changing these settings, verify:

- the JSON parse failure rate;
- retries;
- truncation/length finish reasons;
- missing fields; and
- extraction accuracy.

### 4. Retries

`VLLM_MAX_GENERATION_RETRIES=1` permits one corrective regeneration. Retries
can improve successful JSON completion but consume additional GPU time.

If the retry rate is consistently near zero, keep the setting as a safety net
or test `0` for lower worst-case latency. If retries are frequent, investigate
prompt/output formatting rather than increasing retries indefinitely.

### 5. Prefix caching

Keep `VLLM_ENABLE_PREFIX_CACHING=true` for repeated extraction workloads.
Benefits are largest when requests share long prompt prefixes.

### 6. Thinking mode

Thinking is disabled by default because the endpoint needs compact structured
JSON. Enabling it can increase output length and latency. Enable it only after
an accuracy benchmark shows that the gain is worth the cost.

### 7. Scheduler limits

The current code allows vLLM to choose scheduler defaults. If production load
tests show memory pressure or excessive tail latency, `max_num_seqs` and
`max_num_batched_tokens` can be added to `AsyncEngineArgs` in
`src/model_service.py`.

- `max_num_seqs` caps the number of sequences processed in one scheduler
  iteration.
- `max_num_batched_tokens` caps the tokens processed in one scheduler
  iteration.

These are safety and latency controls, not fixed batch sizes. Setting them too
low underutilizes the GPU; setting them too high can increase KV-cache pressure,
preemption, and tail latency.

## Operations and monitoring

### Health monitoring

Monitor:

```bash
curl -fsS http://127.0.0.1:8000/health
```

Alert when:

- the endpoint cannot be reached;
- processing jobs remain high for an extended period;
- HTTP `429` responses increase;
- error jobs increase; or
- GPU utilization unexpectedly falls to zero while jobs are active.

### GPU monitoring

On the server:

```bash
watch -n 1 nvidia-smi
```

Watch GPU utilization, memory use, temperature, and competing processes during
benchmarks.

### Application logs

For systemd:

```bash
sudo journalctl -u postcall-api -f
```

Useful log events include:

- model startup and completion;
- missing or incompatible adapter files;
- JSON parse/shape failures;
- retries;
- job failures;
- CUDA out-of-memory errors; and
- vLLM KV-cache preemption warnings.

### Job retention

Completed jobs are retained until they exceed the configured TTL or the
completed-job count limit. Cleanup occurs during job completion and API access,
not through a durable external scheduler.

Because records are held in memory:

- higher retention consumes more host RAM;
- process restarts remove all records;
- status polling must reach the process that owns the job; and
- this storage design is suitable for a single-instance internal deployment,
  but should be replaced for durable multi-replica operation.

## Security and data handling

The expected deployment is private and company-only, but the application itself
does not currently enforce authentication or request-size limits.

Before production use:

- keep the server off the public Internet;
- allow inbound traffic only from approved company networks and services;
- enforce company authentication at an internal gateway;
- use TLS in transit;
- configure per-client rate and concurrency limits;
- configure maximum HTTP body and header sizes;
- keep `POSTCALL_MAX_ACTIVE_JOBS` bounded;
- omit `ground_truth` from normal production requests;
- protect model, adapter, log, CSV, and benchmark files with service-account
  permissions;
- avoid logging complete transcripts unless explicitly required;
- define retention and deletion policies for results and evaluation artifacts;
  and
- review dependency and model-artifact updates before deployment.

Treat `job_id` as sensitive operational data. Anyone who can reach this
unauthenticated API and obtains a valid job ID can request that job's status and
result.

## Troubleshooting

### Python version is unsupported

Symptom:

```text
The currently activated Python version is not supported by the project
```

Fix:

```bash
poetry env use python3.12
poetry install
```

### Adapter is missing

Symptom:

```text
LoRA adapter is incomplete; missing: ...
```

Fix: ensure both files exist:

```text
adapter/adapter_config.json
adapter/adapter_model.safetensors
```

### Adapter base model or rank does not match

Symptom:

```text
Adapter expects base model ...
```

or:

```text
Adapter rank is ..., but MAX_LORA_RANK is 16
```

Fix: deploy the adapter trained for
`unsloth/Qwen3-14B-unsloth-bnb-4bit` with rank `16`, or deliberately update the
runtime model/adapter configuration together.

### Model startup is slow

Possible causes:

- first-time Hugging Face download;
- slow model storage;
- insufficient host RAM;
- another process using the GPU; or
- CUDA kernel initialization/compilation.

Pre-download the approved model, verify disk throughput, and inspect
`nvidia-smi`.

### CUDA out of memory

Try:

- lower `VLLM_GPU_MEMORY_UTILIZATION`;
- reduce `VLLM_MAX_MODEL_LEN`;
- reduce client concurrency;
- reduce `POSTCALL_MAX_ACTIVE_JOBS`;
- ensure only one API worker is using the GPU; and
- stop the offline evaluator or other GPU processes.

### HTTP 429

The active-job limit is full. Respect `Retry-After`, use exponential backoff,
reduce client concurrency, or increase capacity only after confirming that GPU
and host memory can support it.

### HTTP 404 for a job

The job ID is invalid, expired, removed by the completed-job cap, belonged to a
different replica, or was lost during a process restart.

### Jobs remain in `processing`

`processing` includes requests waiting inside vLLM. Check:

- current active-job count;
- GPU utilization;
- prompt and output lengths;
- client concurrency;
- vLLM preemption warnings;
- other GPU processes; and
- whether one request is producing an unusually long generation.

### Invalid JSON or repeated generation retries

Check:

- whether thinking mode was enabled;
- whether output was truncated by the context or generation limit;
- field descriptions that contain conflicting instructions;
- very large numbers of requested fields;
- completion finish reasons in the performance metadata; and
- whether lowering concurrency changes the error rate.

### API appears slow but the GPU is not busy

Check:

- that multiple requests are actually arriving concurrently;
- network and status-polling latency;
- CPU tokenization load;
- prompt lengths;
- whether the benchmark is using `--concurrency 1`;
- whether another process is limiting CPU or disk access; and
- whether the API and benchmark are running through a slow proxy path.

Start with a concurrency-1 baseline, then increase gradually and compare the
generated `summary.json` reports.
