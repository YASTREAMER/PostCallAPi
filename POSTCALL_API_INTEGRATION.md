# Postcall Extraction API — Caller-Supplied Prompt Guide

## Overview

The backend server now builds the complete LLM conversation. The Postcall API
does not build a prompt from a transcript, metadata, or function calls.

The backend sends:

1. `messages`: the ordered chat messages given to the model;
2. OpenAI-style `response_format.json_schema` (recommended), or legacy
   `postcall_data`, to define the expected output fields;
3. optional `temperature` in the OpenAI range from `0` through `2`; and
4. optional `include_performance`: whether diagnostic timing/token data should
   be returned.

The optional `model` field is accepted for client compatibility but ignored;
the service always uses its configured model and LoRA adapter.

The API applies the local model chat template, runs vLLM with the Postcall LoRA
adapter, validates the generated JSON, normalizes field values, and returns the
completed result in the same HTTP response. There are no job IDs and no status
polling.

## Connection and authentication

```bash
POSTCALL_API_URL=http://101.53.137.25:8808/postcall
POSTCALL_API_KEY=replace-with-the-key-provided-by-the-api-owner
```

The API root, health, and extraction endpoints require:

```text
Authorization: Bearer <POSTCALL_API_KEY>
```

The current public URL uses plain HTTP. HTTP does not encrypt API keys or call
content. Use HTTPS or a trusted private network before sending sensitive data.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Check availability and active request count. |
| `POST` | `/extract` | Send messages and wait for the complete LLM extraction. |
| `GET` | `/docs` | OpenAPI/Swagger documentation (currently public). |

## Health check

```bash
curl --fail-with-body --max-time 10 \
  -H "Authorization: Bearer $POSTCALL_API_KEY" \
  "$POSTCALL_API_URL/health"
```

Example response:

```json
{
  "status": "ok",
  "active_requests": 0,
  "max_active_requests": 60
}
```

## Extraction request

```bash
curl --fail-with-body --max-time 1200 \
  -X POST "$POSTCALL_API_URL/extract" \
  -H "Authorization: Bearer $POSTCALL_API_KEY" \
  -H "Content-Type: application/json" \
  -d @- <<JSON
{
  "messages": [
    {
      "role": "system",
      "content": "You are a call-analysis extraction engine. Return only one JSON object. Each requested field must contain exactly value and comment. Do not rename fields and do not return markdown."
    },
    {
      "role": "user",
      "content": "<variables_to_extract>\n[{\"name\":\"intrested\",\"type\":\"boolean\",\"description\":\"True if the user showed genuine interest.\"},{\"name\":\"callback_time\",\"type\":\"text\",\"description\":\"Requested callback time, otherwise an empty string.\"}]\n</variables_to_extract>\n<current_time>2026-08-01T12:00:00+05:30</current_time>\n<transcription>Agent: Would you like to know more? User: This sounds interesting. Call me tomorrow at 11 AM.</transcription>\n<functions_called>[]</functions_called>\n<call_metadata>{\"customer_name\":\"Rohan Sharma\"}</call_metadata>"
    }
  ],
  "postcall_data": [
    {
      "name": "intrested",
      "type": "boolean",
      "description": "True if the user showed genuine interest.",
      "defaultValue": false
    },
    {
      "name": "callback_time",
      "type": "text",
      "description": "Requested callback time, otherwise an empty string.",
      "defaultValue": ""
    }
  ],
  "include_performance": false
}
JSON
```

The HTTP connection remains open while inference runs. A successful request
returns HTTP `200` with the completed result.

## Request schema

### `messages`

`messages` is required and must contain at least one item.

| Field | Required | Values | Description |
|---|---:|---|---|
| `role` | Yes | `system`, `user`, or `assistant` | Chat role passed to the model. |
| `content` | Yes | Non-empty string | Exact caller-provided prompt content. |

The usual production shape is:

```json
{
  "messages": [
    { "role": "system", "content": "task and output instructions" },
    { "role": "user", "content": "transcript and call context" }
  ]
}
```

Message order is preserved. The API does not prepend its own system prompt and
does not rewrite the supplied content.

### `response_format` and `postcall_data`

Send either OpenAI-style `response_format` or the legacy `postcall_data`
array. When `response_format` is present, the API derives field names,
descriptions, value types, and defaults from
`response_format.json_schema.schema.properties`. The schema produced by
`build_response_schema_for_openai()` is accepted directly and is enforced by
vLLM structured-output decoding during token generation.

If both forms are sent, their field names and ordering must match.
`postcall_data` remains supported for backward compatibility. The API uses
the resolved schema outside the LLM for deterministic validation and
normalization as an additional safeguard. Legacy requests that provide only
`postcall_data` retain the earlier validation-and-retry behavior.

| Field | Required | Description |
|---|---:|---|
| `name` | Yes | Exact top-level result key. |
| `description` | Yes | Field meaning; also useful when embedding the same object in the prompt. |
| `type` | No | Expected value type; defaults to `text`. |
| `defaultValue` | No | Fallback used when the model omits an appropriate value. |
| `defaultValueConfig` | No | Accepted compatibility metadata. |

Supported normalized types are `boolean`, `number`, `text`, `string`,
`selector`, and `categorical`. Aliases include `bool`, `integer`, `float`, and
`str`.

The spelling and capitalization of every `name` must match the keys requested
inside the prompt. For example, the supplied JavaScript intentionally uses
`intrested`; the API preserves that spelling rather than correcting it.

### Additional fields

| Field | Default | Description |
|---|---:|---|
| `temperature` | `0.0` | vLLM sampling temperature from `0` through `2`. |
| `model` | `null` | Accepted for compatibility and currently ignored. |
| `include_performance` | `false` | Include model timing and token diagnostics. |

Unknown top-level request fields are rejected with HTTP `422`. At least one of
`response_format` and `postcall_data` is required. Do not send top-level
fields such as `transcription`, `call_duration`, `hangup_reason`,
`functions_called`, `call_metadata`, or `timezone`; put that information
inside a caller-built message instead.

## Successful response

```json
{
  "result": {
    "intrested": {
      "value": true,
      "comment": "User explicitly said the product sounds interesting."
    },
    "callback_time": {
      "value": "2026-08-02T11:00:00",
      "comment": "User requested a callback tomorrow at 11 AM."
    }
  },
  "usage": {
    "prompt_tokens": 486,
    "completion_tokens": 61,
    "total_tokens": 547
  }
}
```

Every response includes a top-level `usage` object with `prompt_tokens`,
`completion_tokens`, and `total_tokens`. This is returned unconditionally,
whether or not `include_performance` is set — no request field is needed to
get token counts. The naming matches OpenAI's `usage` shape
(`prompt_tokens`/`completion_tokens`/`total_tokens`).

Every expected field is returned as:

```json
{
  "value": "normalized value",
  "comment": "supporting evidence"
}
```

The API checks that:

- every field resolved from `response_format` or `postcall_data` exists at
  the top level;
- no unexpected top-level result keys are present;
- each field contains exactly `value` and `comment`; and
- each comment is a non-empty string.

If the first model response has the wrong shape, the API retries with an
additional corrective user message. It then normalizes values using
`postcall_data`.

With `include_performance: true`, the response additionally contains a
`performance` object with retry and timing detail on top of the `usage` field
described above:

```json
{
  "result": {},
  "usage": {
    "prompt_tokens": 486,
    "completion_tokens": 61,
    "total_tokens": 547
  },
  "performance": {
    "attempts": 1,
    "retried": false,
    "generation_seconds": 2.417,
    "prompt_tokens": 486,
    "completion_tokens": 61,
    "total_tokens": 547,
    "attempt_details": []
  }
}
```

`performance` is omitted entirely when `include_performance` is `false` or
unset. `usage` is present either way.

## Node.js 18+ integration

The backend can reuse the existing `prompt`, `userContent`, and `postcall_data`
variables from the Azure implementation. Node.js 18 and newer provide `fetch`
and `AbortSignal.timeout` globally.

Set these variables on the Node server:

```env
POSTCALL_API_URL=http://101.53.137.25:8808/postcall
POSTCALL_API_KEY=replace-with-the-key-provided-by-the-api-owner
```

Use this request helper:

```javascript
require("dotenv").config();

const POSTCALL_API_URL = (
  process.env.POSTCALL_API_URL ||
  "http://101.53.137.25:8808/postcall"
).replace(/\/$/, "");

const POSTCALL_API_KEY = process.env.POSTCALL_API_KEY;

async function callPostcallAPI({
  prompt,
  userContent,
  responseFormat,
  temperature = 0,
  includePerformance = false,
}) {
  if (!POSTCALL_API_KEY) {
    throw new Error("POSTCALL_API_KEY is not configured");
  }

  const response = await fetch(`${POSTCALL_API_URL}/extract`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${POSTCALL_API_KEY}`,
    },
    body: JSON.stringify({
      messages: [
        { role: "system", content: prompt },
        { role: "user", content: userContent },
      ],
      temperature,
      response_format: responseFormat,
      include_performance: includePerformance,
    }),
    signal: AbortSignal.timeout(20 * 60 * 1000),
  });

  const responseText = await response.text();
  let responseBody;

  try {
    responseBody = JSON.parse(responseText);
  } catch {
    throw new Error(
      `Postcall API returned invalid JSON (${response.status}): ${responseText}`
    );
  }

  if (!response.ok) {
    const error = new Error(
      `Postcall API returned HTTP ${response.status}: ${JSON.stringify(responseBody)}`
    );
    error.status = response.status;
    error.retryAfter = response.headers.get("retry-after");
    error.response = responseBody;
    throw error;
  }

  if (!responseBody.result || typeof responseBody.result !== "object") {
    throw new Error(
      `Postcall API response is missing result: ${JSON.stringify(responseBody)}`
    );
  }

  return responseBody;
}
```

Replace the Azure completion call with:

```javascript
try {
  const response = await callPostcallAPI({
    prompt,
    userContent,
    temperature: 0.8,
    responseFormat: {
      type: "json_schema",
      json_schema: build_response_schema_for_openai(postcall_data),
    },
    includePerformance: true,
  });

  const result = response.result;

  result.x_model_used = {
    value: "postcall-qwen3-14b-lora",
    comment: "Model used for post-call analysis.",
  };

  console.log(
    "Post-call analysis result:\n",
    JSON.stringify(result, null, 2)
  );

  // response.usage is always present, regardless of includePerformance.
  console.log("Input tokens:", response.usage.prompt_tokens);
  console.log("Output tokens:", response.usage.completion_tokens);
  console.log("Total tokens:", response.usage.total_tokens);

  if (response.performance) {
    console.log("Performance:", response.performance);
  }
} catch (error) {
  console.error("Postcall API request failed:", {
    message: error.message,
    status: error.status,
    retryAfter: error.retryAfter,
    response: error.response,
  });
}
```

The request remains open until inference finishes. Do not poll a status route.
On HTTP `429`, wait for the `retryAfter` duration and retry with bounded backoff.
Do not retry HTTP `401` or `422` without correcting the credentials or payload.

## Error handling

| HTTP status | Meaning | Client action |
|---:|---|---|
| `200` | Complete extraction returned. | Read `result`. |
| `401` | Bearer key missing or invalid. | Fix the `Authorization` header. |
| `422` | Request does not match the schema. | Correct the payload; do not retry unchanged. |
| `429` | Active-request limit reached. | Respect `Retry-After` and retry with backoff. |
| `500` | Model generation or output validation failed. | Log the response and retry with bounded backoff. |

Use a request timeout long enough for model generation, such as 10–20 minutes.
Do not automatically retry a timed-out POST indefinitely; the original request
may still have consumed inference work.

## How the supplied Azure JavaScript works

The supplied script performs these steps:

1. Loads Azure credentials from environment variables with `dotenv`.
2. Creates one `AzureOpenAI` client using the endpoint, API key, and API
   version.
3. Defines the Azure deployment name (`gpt-4.1-mini-1`) and temperature (`0.8`).
4. Converts `postcall_data` field types into an Azure/OpenAI strict JSON Schema.
   Each result field must be an object containing `value` and `comment`.
5. Builds `prompt`, the system message containing task instructions, field
   definitions, defaults, and output rules.
6. Builds `userContent`, the user message containing previous dispositions,
   duration, hangup reason, transcript, function calls, and metadata.
7. Calls Azure Chat Completions with those two messages and the strict
   `response_format` schema.
8. Reads token usage, separates cached from billed input tokens, and estimates
   cost using the configured per-token prices.
9. Parses the assistant content as JSON and adds an `x_model_used` field only
   in the JavaScript result after generation.

For the H200 API, reuse the same `prompt` and `userContent` as the two
entries in `messages`, and send the existing
`build_response_schema_for_openai(postcall_data)` output as
`response_format.json_schema`. The API accepts `temperature` and applies it
to vLLM sampling, while the JSON Schema constrains generation itself. It also
accepts `model` for compatibility but ignores its value because the H200 model
and LoRA adapter are fixed by the service.

Azure endpoint, API version, token pricing, and Azure SDK credentials are not
part of the H200 request.

The H200 API returns `result` directly rather than an Azure `choices` object,
so there is no `JSON.parse(response.choices[0].message.content)` step. If the
backend still needs `x_model_used`, it should add it after receiving the H200
response, just as the supplied JavaScript currently does.

Token usage works the same way in both: the H200 API's top-level `usage`
object uses the same field names as Azure's (`prompt_tokens`,
`completion_tokens`, `total_tokens`), so the existing cost-estimation logic
can be reused as-is. The one difference: the H200 API does not track
cached-token counts, so treat `cached_tokens` as always `0` when reusing that
logic.

## Interactive documentation

```text
http://101.53.137.25:8808/postcall/docs
```
