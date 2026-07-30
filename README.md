# PostCallAPi

Post-call extraction API backed by vLLM and a LoRA adapter.

## Run the API

From the project root:

```bash
poetry run uvicorn main:app --app-dir src --host 0.0.0.0 --port 8000
```

Or run:

```bash
poetry run python3 src/main.py
```

Wait until the model has loaded before starting a benchmark. Do not use
multiple Uvicorn workers on one GPU because each worker loads another model.

The server defaults are optimized for structured extraction: model thinking is
disabled, prefix caching is enabled, generation is bounded by field count, and
completed jobs are retained for one hour with a 10,000-job cap. These can be
overridden with environment variables such as `POSTCALL_ENABLE_THINKING`,
`VLLM_GPU_MEMORY_UTILIZATION`, `VLLM_MAX_NEW_TOKENS`,
`POSTCALL_JOB_TTL_SECONDS`, `POSTCALL_MAX_COMPLETED_JOBS`, and
`POSTCALL_MAX_ACTIVE_JOBS`. Requests beyond the active-job limit receive HTTP
429 instead of creating an unbounded inference queue.

Use `GET /health` to inspect retained and active job counts. Completed job
responses also include generation time, prompt/completion tokens, and retries.

## Benchmark the live API

The benchmark samples CSV rows, sends production-shaped requests concurrently,
polls each asynchronous job, scores the response against CSV ground truth, and
writes reports beneath `output/`.

Test 200 random rows with at most 10 jobs in flight:

```bash
poetry run python3 src/api_csv_test.py \
  --csv data/Data_with_outcome_fields.csv \
  --count 200 \
  --concurrency 10 \
  --api-url http://127.0.0.1:8000
```

For a 500-row run, change `--count` to `500`. Increase `--concurrency`
gradually to test heavier traffic without immediately overwhelming the server.
Use `--selection first` for the first N rows or the default seeded random sample
for reproducible runs.

Each run creates `output/api_test_<timestamp>/` containing:

- `summary.json`: wall time, throughput, latency percentiles, token usage,
  retry rate, strict typed accuracy, meaningful-value accuracy, and coverage.
- `rows.csv`: status, timing, token usage, strict accuracy, and accuracy per row.
- `field_scores.csv`: values, truth, schema type, type validity, and field scores.
- `results.jsonl`: complete model results and evaluation details for each row.

During a run, each completed row prints its output token count, per-job output
tokens/minute, and rolling wall-clock output tokens/minute. `summary.json` also
contains final completion-token and total-token throughput per minute.

Use `--api-url https://your-production-host` to benchmark a deployed instance.
Run `poetry run python3 src/api_csv_test.py --help` for all options.
