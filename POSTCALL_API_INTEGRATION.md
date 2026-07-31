# Postcall Extraction API Integration Guide

## Connection details

The Postcall Extraction API is publicly reachable over the internet.

```text
Base URL: http://101.53.137.25:8088/postcall
```

No SSH tunnel is required. The API currently uses plain HTTP and does not
require an API key or other authentication.

## Available endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Check whether the API is available. |
| `POST` | `/extract` | Submit a call transcript for asynchronous extraction. |
| `GET` | `/status/{job_id}` | Check a job and retrieve its completed result. |
| `GET` | `/docs` | Open the interactive FastAPI documentation. |

The complete URLs are formed by appending an endpoint to the base URL. For
example:

```text
http://101.53.137.25:8088/postcall/health
```

## Processing workflow

Extraction is asynchronous. Integrators must use the following workflow:

1. Optionally call `GET /health` to confirm that the service is available.
2. Send the extraction payload to `POST /extract`.
3. Read the `job_id` from the accepted response.
4. Poll `GET /status/{job_id}` while the status is `pending` or `processing`.
5. Stop polling when the status becomes `done` or `error`.
6. When the status is `done`, read the extracted fields from `result`.

Do not expect `POST /extract` to return the completed LLM result immediately.

## Check service health

Request:

```bash
curl --fail-with-body --max-time 10 \
  http://101.53.137.25:8088/postcall/health
```

Example response:

```json
{
  "status": "ok",
  "jobs": {
    "processing": 2,
    "done": 15
  },
  "retained_jobs": 17,
  "max_active_jobs": 60
}
```

The health endpoint confirms that the web application is running. It does not
perform an LLM generation.

## Submit an extraction

Request:

```bash
curl --fail-with-body --max-time 60 \
  -X POST \
  http://101.53.137.25:8088/postcall/extract \
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
        "description": "Briefly describe the final outcome of the call.",
        "type": "text",
        "defaultValue": ""
      },
      {
        "name": "follow_up_required",
        "description": "True when the customer or agent requests another contact.",
        "type": "boolean",
        "defaultValue": false
      }
    ],
    "transcription": "Agent: Would you like a product demonstration tomorrow? Customer: Yes, please schedule it for 3 PM.",
    "call_duration": 42.5,
    "hangup_reason": "customer-ended-call",
    "functions_called": [],
    "call_metadata": {
      "call_id": "call-123",
      "source": "backend"
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

The accepted response means that the job was queued. It does not mean that LLM
processing has finished.

## Extraction request schema

### Top-level fields

| Field | Type | Required | Description |
|---|---|---:|---|
| `postcall_data` | array | Yes | Fields that the LLM must extract. |
| `transcription` | string | Yes | Complete call transcript. |
| `call_duration` | number or `null` | No | Call duration, normally in seconds. |
| `hangup_reason` | string | No | Reason that the call ended. Defaults to an empty string. |
| `functions_called` | array | No | Tools or functions called during the conversation. Defaults to `[]`. |
| `call_metadata` | object | No | Caller-provided metadata such as call ID or campaign. Defaults to `{}`. |
| `timezone` | string | No | Timezone used to interpret the call. Defaults to `Asia/Kolkata`. |

### `postcall_data` fields

Each item describes one value that should appear in the LLM result.

| Field | Type | Required | Description |
|---|---|---:|---|
| `name` | string | Yes | Unique output field name. |
| `description` | string | Yes | Clear instructions defining what the model should extract. |
| `type` | string | No | Output type. Defaults to `text`. |
| `defaultValue` | any | No | Fallback value when the transcript does not provide an answer. |
| `defaultValueConfig` | object | No | Optional metadata associated with the default value. |

Supported output types include:

- `boolean`
- `number`
- `text`
- `string`
- `selector`
- `categorical`

Aliases are also accepted: `bool` becomes `boolean`, `integer` and `float`
become `number`, and `str` becomes `string`.

Descriptions should be explicit about the evidence required for a positive or
non-default result. Use a `defaultValue` with the same JSON type requested by
`type`.

### `functions_called` fields

Each function record can contain:

| Field | Type | Required |
|---|---|---:|
| `name` | string | Yes |
| `parameters` | object or `null` | No |
| `response` | any | No |
| `success` | boolean | No; defaults to `false` |
| `timestamp` | string or `null` | No |

## Poll a submitted job

Replace the example UUID with the `job_id` returned by `POST /extract`.

```bash
curl --fail-with-body --max-time 60 \
  http://101.53.137.25:8088/postcall/status/dc37a7a1-d24b-41b8-b0f9-82a0ad6eb25f
```

Possible job statuses:

| Status | Meaning | Client action |
|---|---|---|
| `pending` | The job was accepted but has not started. | Wait and poll again. |
| `processing` | The job is running or waiting in the model scheduler. | Wait and poll again. |
| `done` | Extraction completed successfully. | Read `result`. |
| `error` | Extraction failed. | Stop polling and record `error`. |

Poll approximately every 1–2 seconds. Avoid continuous polling without a
delay. Use an overall job deadline appropriate for LLM processing, such as
10–20 minutes, rather than an unlimited polling loop.

### Completed response

```json
{
  "job_id": "dc37a7a1-d24b-41b8-b0f9-82a0ad6eb25f",
  "status": "done",
  "result": {
    "customer_interested": {
      "value": true,
      "comment": "The customer agreed to schedule a demonstration."
    },
    "disposition_reason": {
      "value": "Customer requested a product demonstration.",
      "comment": "The customer asked for the demonstration at 3 PM."
    },
    "follow_up_required": {
      "value": true,
      "comment": "A demonstration must be scheduled."
    }
  },
  "error": null,
  "performance": {
    "attempts": 1,
    "retried": false
  }
}
```

Every requested output field is returned as an object containing:

```json
{
  "value": "the normalized extracted value",
  "comment": "supporting explanation or evidence"
}
```

The exact keys inside `performance` may vary. Backend integrations should not
depend on every performance field being present.

### Error response

```json
{
  "job_id": "dc37a7a1-d24b-41b8-b0f9-82a0ad6eb25f",
  "status": "error",
  "result": null,
  "error": "Error description",
  "performance": null
}
```

## HTTP errors and retry behavior

| HTTP status | Meaning | Recommended action |
|---:|---|---|
| `200` | Request succeeded or extraction job was accepted. | Process the JSON response. |
| `404` | The job ID is unknown or has expired. | Stop polling and verify the job ID. |
| `422` | The request payload does not match the required schema. | Correct the request; do not retry unchanged. |
| `429` | The inference queue has reached its active-job limit. | Respect `Retry-After`, then retry submission. |
| `500` | Unexpected server error. | Log the response and retry with bounded backoff if appropriate. |

Clients should:

- use a connection/request timeout;
- retry temporary network failures and HTTP `429`/`5xx` responses with bounded
  exponential backoff;
- avoid retrying HTTP `422` without correcting the payload;
- store the returned `job_id` so processing can resume after a client restart;
- make sure duplicate submission is acceptable before retrying a submission
  whose response was lost, because the API does not currently accept an
  idempotency key;
- limit client concurrency so the public inference queue is not overwhelmed.

Completed jobs are retained temporarily in server memory. Poll and save the
result promptly rather than treating the API as permanent job storage.

## Interactive documentation

FastAPI documentation is available at:

```text
http://101.53.137.25:8088/postcall/docs
```

## Security notice

This deployment is publicly reachable, has no authentication, and uses
unencrypted HTTP. Anyone who knows or discovers the address can submit work,
consume GPU capacity, and observe any data they send. Call transcripts may
contain sensitive or personal information.

For production use, the service owner should add HTTPS, authentication, request
size limits, rate limiting, and network allowlisting where possible. Backend
clients should not send secrets or regulated data over the current public HTTP
endpoint.
