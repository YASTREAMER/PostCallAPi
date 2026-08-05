# Postcall H200 API — Benchmark Findings for Server-Side Diagnosis

This file summarizes a client-side load test of the Postcall extraction API
(`vLLM` + Qwen3-14B + LoRA adapter, per `adapter/adapter_config.json`) and is
meant to be handed to Claude Code running **on the H200 server** so it can
correlate these symptoms with server logs, vLLM config, and process state.

## Companion files this refers to (bring them along)

- `benchmark.py` — the load-test client that produced these results.
- `Data_with_outcome_fields.csv` — the 2000-row dataset it samples from.
- `Doc.md` — the documented API contract.
- `output/benchmark_20260802_232237_680452/101.53.137.25_8808/` — full run
  against server A (this is the server referenced by `Doc.md` and
  `test.js`'s default URL — treat this as "the H200").
- `output/benchmark_20260802_232437_701220/164.52.192.77_8808/` — full run
  against server B, run in parallel against a different host.
- Each concurrency subfolder has `results.jsonl` (one row per request, with
  timings/errors), `responses.jsonl` (raw API responses), and `summary.json`.

Both runs used the **same 200-row random sample** (`seed=42`) at concurrency
levels 20/30/40/50/75/100/125/150, 1200s (20 min) client timeout per request.

---

## Finding 1 (highest priority): a fixed set of rows deterministically return HTTP 500

**Symptom:** The same 9 row indices — `1774, 1780, 1793, 1794, 1811, 1812,
1813, 1827, 1838` — fail with `HTTP 500 {"detail":"Model generation failed"}`
on **both** servers, at **every** concurrency level (20 through 125), every
time. They fail fast: `api_request_seconds` ≈ 0.22–0.3s, so this is failing
before/during request setup, not during actual generation or a timeout.

Evidence (from `results.jsonl`, e.g.
`output/.../101.53.137.25_8808/concurrency_020/results.jsonl`, row_index 1774):
```json
{"status":"http_error","http_status":500,"api_request_seconds":0.287,
 "error_response_body":"{\"detail\":\"Model generation failed\"}"}
```

**Root cause candidate — confirmed correlation:** every one of these 9 rows
has an unusually large `postcall` field-definition list in the CSV: **40
output fields** (~59KB of JSON), versus the typical row's **9 fields**
(~18–19KB). Checked against the full dataset: rows with a 40-field schema
sit in a size bucket (`59402`–`59901` chars) that's completely disjoint from
the normal ~18–40K bucket. All 9 sampled rows that fall in that bucket fail;
none of the normal-size rows fail this way.

```python
# Verified: bucket of 40-field-schema rows == the 9 that always fail
import csv, json
with open("Data_with_outcome_fields.csv", newline="", encoding="utf-8") as f:
    rows = list(csv.reader(f))
header, rows = rows[0], rows[1:]
pcol = header.index("postcall")
big = [i for i, r in enumerate(rows) if len(r[pcol]) >= 59000]
print(len(big), "rows in the full 2000-row dataset have a 40-field schema")
```

**Leading hypothesis:** `benchmark.py` sends the field list as a **strict
JSON Schema** via `response_format` (see `_build_response_schema` /
`_build_payload` in `benchmark.py`, lines ~268–394) — `strict: true`, one
`object` property per output field, each requiring `additionalProperties:
false` and a nested `value`/`comment` object. For a 40-field schema this is a
large, deeply-nested strict schema. vLLM's structured-output / guided-decoding
backend (outlines / xgrammar / lm-format-enforcer, whichever is configured)
compiles a schema into a token-level grammar or FSM before generation starts,
and compile cost for `strict` JSON Schema can blow up non-linearly with
property count — a common failure mode is the compiler exceeding time/memory
and the request erroring out server-side, exactly matching "fails instantly,
every time, only for large schemas."

**What to check on the server:**
1. `grep`/`journalctl` the API/vLLM process logs at the timestamps of any of
   these row IDs (cross-reference `started_at` in `results.jsonl`) for the
   actual traceback behind `"Model generation failed"` — the client only sees
   the generic message.
2. Which guided-decoding backend is configured (`--guided-decoding-backend`
   in vLLM args, or equivalent) and whether it has a known issue/limit with
   large `strict` schemas or many `required` properties.
3. Whether `max_model_len` / prompt+schema token budget is being exceeded —
   the *schema* itself (not just the transcript) counts against context in
   some structured-output implementations.
4. Whether this reproduces with a schema that's large but *not* deeply
   nested (e.g. 40 flat string fields vs 40 object-wrapped fields) — that
   isolates "too many fields" from "schema shape/nesting."

**Repro script** (run on the server, against `localhost` or the real port;
adjust `API_URL`/`API_KEY`):

```python
#!/usr/bin/env python3
"""Reproduce the 40-field-schema 500 using row 1774 from the CSV.
Mirrors benchmark.py's exact request-building logic."""
import csv, json, os, urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo

CSV_PATH = "Data_with_outcome_fields.csv"
API_URL = os.environ.get("POSTCALL_API_URL", "http://127.0.0.1:8808/postcall")
API_KEY = os.environ["POSTCALL_API_KEY"]
ROW_INDEX = 1774  # one of the 9 rows that always fails; try 0 as a control

with open(CSV_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
row = rows[ROW_INDEX]

def jcell(col, default):
    raw = row.get(col, "")
    return json.loads(raw) if raw else default

schema = jcell("postcall", [])
print(f"row {ROW_INDEX}: {len(schema)} output fields")

functions_called = jcell("functions_called", [])
call_metadata = jcell("call_metadata", {})
current_time = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%dT%H:%M:%S")

def compact(v):
    return json.dumps(v, ensure_ascii=False, separators=(",", ":"))

system_prompt = f"""
<task>
current_time: {current_time},
Given a call transcription and function called(name, parameters, success, timestamp), extract the
specified details and output a single JSON object with keys corresponding to the variable names and their values as an object of value and comment e.g "variable_name": {{"value": "value here", "comment": "explanation about how you got this value"}}.
- DONT change the variable and key names even if it is wrong.
- If a string variable is not present, assign it an empty string: "".
- If a boolean variable is not present, assign it the value false.
Return only the JSON object—no extra text, explanation, or markdown.
</task>
<details>
    {row.get("conversion_reason", "")}
</details>
<variables_to_extract>
{compact(schema)}
</variables_to_extract>
"""

user_content = f"""
<previous_calls_dispositions></previous_calls_dispositions>
<call_duration>{row.get("call_duration", "")}</call_duration>
<hangup_reason>{row.get("hangup_reason", "")}</hangup_reason>
<transcription>{row.get("transcription", "")}</transcription>
<functions_called>{compact(functions_called)}</functions_called>
<call_metadata>{compact(call_metadata)}</call_metadata>
"""

def to_openai_type(t):
    return {"text": "string", "str": "string", "boolean": "boolean", "bool": "boolean",
            "integer": "integer", "int": "integer", "number": "number", "float": "number"
            }.get(str(t or "string").casefold(), "string")

properties, required = {}, []
for field in schema:
    name = field["name"]
    vtype = to_openai_type(field.get("type"))
    properties[name] = {
        "type": "object",
        "properties": {"value": {"type": vtype}, "comment": {"type": "string"}},
        "required": ["value", "comment"],
        "additionalProperties": False,
    }
    required.append(name)

payload_variant_A_legacy_schema = {
    "messages": [{"role": "system", "content": system_prompt},
                 {"role": "user", "content": user_content}],
    "temperature": 0.8,
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "post_call_analysis", "strict": True,
            "schema": {"type": "object", "properties": properties,
                       "required": required, "additionalProperties": False},
        },
    },
    "include_performance": True,
}

payload_variant_B_documented_contract = {
    "messages": [{"role": "system", "content": system_prompt},
                 {"role": "user", "content": user_content}],
    "postcall_data": schema,
    "include_performance": True,
}

for label, payload in [("A (legacy: temperature+response_format, what benchmark.py actually sends)",
                         payload_variant_A_legacy_schema),
                        ("B (Doc.md contract: postcall_data, no temperature/response_format)",
                         payload_variant_B_documented_contract)]:
    print(f"\n--- sending variant {label} ---")
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(f"{API_URL}/extract", data=body, method="POST",
                                  headers={"Content-Type": "application/json",
                                           "Authorization": f"Bearer {API_KEY}"})
    try:
        with urllib.request.urlopen(req, timeout=1200) as resp:
            print(resp.status, resp.read().decode()[:2000])
    except urllib.error.HTTPError as e:
        print(e.code, e.read().decode()[:2000])
```

Also worth a quick bisection: re-run with `schema[:20]`, `schema[:10]` sliced
from the 40-field list to find the exact field count where it starts
failing — that pinpoints whether it's a hard limit (e.g. a token/size cap)
or a slow-but-eventually-crashing compile (which would show increasing
`api_request_seconds` as field count grows, right up to the point of failure).

---

## Finding 2: the benchmark client's request body doesn't match `Doc.md`'s documented contract

`Doc.md` states the request must be `{messages, postcall_data,
include_performance}` and explicitly says **not** to send `temperature` or
`response_format` ("Unknown top-level request fields are rejected with HTTP
422"). But the actual code path that fires requests, `_build_payload` in
`benchmark.py` (lines 377–394, wired to the request at line 643-654), sends:

```json
{"messages": [...], "temperature": 0.8, "response_format": {...}, "include_performance": true}
```

— i.e. the **old Azure-style** shape, with **no `postcall_data` key at all**.

Despite that, ~93-95% of requests to server A returned `200` with well-formed
results. That means one of two things, and it matters for interpreting every
other finding here:

- (a) the deployed server is more lenient than `Doc.md` describes (it
  doesn't actually reject unknown fields / doesn't actually require
  `postcall_data`, and derives the field list from
  `response_format.json_schema` when present) — in which case `Doc.md` is
  describing a contract that isn't fully enforced yet, or
- (b) the server has two code paths (an old Azure-compatible one and a new
  `postcall_data` one) and this benchmark has only ever exercised the old
  one — meaning **all these results describe the legacy path**, and the new
  documented path (`postcall_data`) is untested.

**What to check on the server:** find the request-handling code for
`/postcall/extract` and see whether `postcall_data` is actually required, and
whether `temperature`/`response_format` are actually rejected as `Doc.md`
claims. This also matters for Finding 1: if the *documented* contract
(`postcall_data`, no schema) skips the strict-JSON-Schema/guided-decoding
path entirely, the 40-field failure may be specific to the legacy
`response_format` path and might not reproduce via `postcall_data` — the
repro script above sends both variants for exactly this reason.

---

## Finding 3: server B (`164.52.192.77`) can't sustain throughput at any concurrency tested

| concurrency | success rate | timeouts | p50 latency (s) | mean gen time (s) | successful req/s |
|---|---|---|---|---|---|
| 20  | 53.5% | 84  | 232.1  | 185.0 | 0.017 |
| 30  | 38.5% | 114 | 1200.1 | 154.3 | 0.015 |
| 40  | 33.0% | 125 | 1200.1 | 171.8 | 0.014 |
| 50  | 20.5% | 150 | 1200.1 | 173.5 | 0.010 |
| 75  | 20.5% | 150 | 1200.1 | 188.3 | 0.015 |
| 100 | 16.0% | 159 | 1200.1 | 189.1 | 0.013 |
| 125 | 17.0% | 157 | 1200.2 | 177.5 | 0.014 |
| 150 | 3.5%  | 150 (+43 network_error) | 1200.2 | 131.3 | 0.005 |

(Source: `output/benchmark_20260802_232437_701220/164.52.192.77_8808/overall_summary.json`)

**The key anomaly:** `successful_requests_per_second` is flat at ~0.01-0.017
**across every concurrency level from 20 to 125** — dialing client
concurrency up doesn't move completed-request throughput at all, it just
converts more requests into 1200s timeouts. `mean_generation_seconds` for
requests that *do* finish stays a normal ~150-190s throughout, so the model
itself isn't the bottleneck. This pattern — flat throughput regardless of
offered load, individual completions look fine, everything else queues to
the timeout — points at **something serializing request handling far below
the concurrency being offered**: a single-worker process, a global lock
around the vLLM call, an `asyncio` loop being blocked by sync work, or a
reverse proxy/gateway in front with its own low concurrency limit.

**What to check on the server (for whichever process is behind
`164.52.192.77:8808`):**
1. `nvidia-smi dmon` / GPU utilization while running load at concurrency 20-30
   — if the GPU sits mostly idle while requests queue, this is an
   application-level serialization bug, not a compute limit.
2. How many worker processes/threads the API server is running (uvicorn
   `--workers`, gunicorn worker count, etc.) — a single worker with a
   synchronous call into vLLM would exactly produce "throughput doesn't
   scale with concurrency."
3. Whether vLLM's engine is configured for continuous batching
   (`max_num_seqs`, `max_num_batched_tokens`) at a value that actually allows
   ~20+ concurrent sequences, or whether it's effectively capped much lower.
4. Compare this server's process/config against server A's — server A (see
   Finding 4 below) shows throughput *scaling with* concurrency (0.104 →
   0.156 successful req/s from 20→125), so whatever's different between the
   two deployments is the likely cause.
5. At concurrency=150 this server also starts returning `network_error`
   (43 of 200) — check for connection/socket exhaustion or the process
   restarting under load.

---

## Finding 4: server A (`101.53.137.25`, "the H200") looks healthy through concurrency 125, but concurrency=150 hit `Connection refused`

| concurrency | success rate | http_error(500) | timeouts | p50 latency (s) | mean gen time (s) | successful req/s | ground truth score |
|---|---|---|---|---|---|---|---|
| 20  | 94.5% | 9  | 2 | 84.4  | 83.2  | 0.104 | 0.931 |
| 30  | 94.5% | 9  | 2 | 93.7  | 92.3  | 0.157 | 0.931 |
| 40  | 94.0% | 9  | 3 | 104.9 | 104.2 | 0.136 | 0.938 |
| 50  | 94.0% | 10 | 2 | 109.3 | 110.6 | 0.131 | 0.933 |
| 75  | 93.5% | 9  | 4 | 128.5 | 127.4 | 0.156 | 0.938 |
| 100 | 93.0% | 10 | 4 | 139.8 | 141.7 | 0.155 | 0.936 |
| 125 | 93.5% | 9  | 4 | 153.4 | 151.6 | 0.156 | 0.934 |
| 150 | 0%    | -  | - | -     | -     | -     | - |

(Source: `output/benchmark_20260802_232237_680452/101.53.137.25_8808/overall_summary.json`)

Up through concurrency 125: success rate is flat at ~93-95%, latency and
generation time scale gracefully with load, throughput actually *increases*
with concurrency (real batching, unlike server B above), and quality is
untouched (`schema_valid_rate` 1.0 throughout, ground-truth score stable at
0.93-0.94). The only recurring failures in this range are the 9 fixed rows
from Finding 1.

At **concurrency=150**, all 200 requests failed instantly (0.357s wall time
total) with:
```
Cannot reach API at http://101.53.137.25:8808/postcall/extract: [Errno 111] Connection refused
```
This is not a timeout or a graceful rejection — nothing was listening /
actively refusing on that port at that moment
(`2026-08-02T20:29Z` / `2026-08-03 01:59 IST`).

The user running this benchmark reports a possible local network issue
around this time, which could explain this in isolation. One data point
against a *blanket* network outage: a second benchmark process was
simultaneously hitting `164.52.192.77:8808` at the same wall-clock moment
and its requests kept completing normally through that window (see
`164.52.192.77_8808/concurrency_030/results.jsonl`, `done` events at
`2026-08-02T20:27-20:39Z`) — so if this was a client-side network drop, it
was either very short-lived or specific to the route/connection to server A.

**What to check on the server:**
1. Process uptime / restart timestamp for the API process around
   `2026-08-02 20:25-20:35 UTC` (`01:55-02:05 IST`) — did it crash/restart?
2. `dmesg` / OOM killer logs around that timestamp — sustained load across
   concurrency 20→125 for ~2.5 hours beforehand is a plausible lead-up to a
   GPU or host OOM.
3. **Rerun concurrency=150 in isolation** (just that one level, nothing else
   running) to see if it reproduces. If it fails the same way again, it's a
   real capacity ceiling around 125-150 concurrent requests, not a network
   fluke. If it passes clean, treat the original result as inconclusive.

---

## Finding 5: `/health`'s `max_active_requests` reports 600000 — admission control isn't real

`Doc.md` documents `max_active_requests` as a meaningful cap that should
trigger `429` + `Retry-After` responses under overload (with an example
value of `60`). But the actual health snapshot captured at the start of both
runs (`run_config.json` in each output folder) shows:
```json
"health": {"status": "ok", "active_requests": 0, "max_active_requests": 600000}
```
600000 is effectively "no limit." This matters because it's very likely why
overload shows up as **silent 1200s client timeouts** (Finding 3) instead of
fast `429`s the client could back off and retry on, as `Doc.md` promises.

**What to check on the server:** find where `max_active_requests` is
configured/returned by `/health` and confirm whether real admission control
(a semaphore/queue cap that returns 429 past some threshold) is wired up at
all, or whether that field is just a placeholder/default that was never set
to a real value.

---

## Suggested order of investigation

1. Finding 2 first (contract mismatch) — it changes how to interpret
   everything else, and is a 10-minute code read.
2. Finding 1 (40-field-schema 500s) — deterministic, cheap to reproduce with
   the script above, likely a guided-decoding/schema-size limit.
3. Finding 3 (server B throughput ceiling) — compare its process/engine
   config against server A's; check GPU utilization under load.
4. Finding 4 (server A's concurrency=150 refusal) — rerun in isolation to
   settle network-vs-crash; correlate with process logs if it reproduces.
5. Finding 5 (fake admission control) — implement or fix real backpressure
   so overload degrades as fast 429s instead of 20-minute hangs.
