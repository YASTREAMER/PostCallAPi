"""
Offline batch evaluation using vLLM's batch generate — processes ALL prompts
in one call instead of one row at a time, using vLLM's continuous batching
for much higher throughput than a one-row-at-a-time loop.

Uses vLLM's offline LLM class (not AsyncLLMEngine, which model_service.py
uses for the live server) — this is the right tool for a one-shot batch job
where all prompts are known up front.

NOTE: don't run this at the same time as `uvicorn main:app` — each starts
its own vLLM engine instance and claims GPU memory independently.

Usage:
    python evaluator.py --csv path/to/maybe.csv --test-size 0.2 --out report.csv
"""

from runtime_config import (
    ADAPTER_DIR,
    ADAPTER_ID,
    ADAPTER_NAME,
    ENABLE_THINKING,
    GPU_MEMORY_UTILIZATION,
    HF_HUB_CACHE,
    MAX_LORA_RANK,
    MAX_MODEL_LEN,
    MAX_NEW_TOKENS,
    MODEL_ID,
    configure_runtime_cache,
    validate_runtime_paths,
)

configure_runtime_cache()

import argparse
import json
import re
from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

import evaluation
from normalize_output import normalize_model_output
from prompt_builder import SYSTEM_INSTRUCTION, build_prompt
from schemas import ExtractRequest


CONVERSION_STATUS_FIELD = "conversion_status"
DISPOSITION_REASON_FIELD = "disposition_reason"

DEFAULT_CONVERSION_STATUS_DESCRIPTION = (
    "Return true only when the call satisfies the conversion criteria and the "
    "user successfully completes or clearly commits to the required outcome; "
    "otherwise return false."
)
CONVERSION_STATUS_OUTPUT_INSTRUCTION = (
    "Set this field's value to the JSON boolean true or false, never a status "
    "label; put the concise supporting explanation in its comment; determine "
    "conversion using this rule: "
)
DISPOSITION_REASON_DESCRIPTION = (
    "Briefly state the primary call outcome and why the user did or did not "
    "convert, using evidence from the transcript and call context."
)


def _add_evaluation_fields(postcall_data: list, row: pd.Series) -> list:
    """Add fields present in ground truth but omitted from the source schema."""
    result = list(postcall_data)
    existing_names = {
        str(field.get("name", "")).strip()
        for field in result
        if isinstance(field, dict)
    }

    if CONVERSION_STATUS_FIELD not in existing_names:
        conversion_reason = row.get("conversion_reason")
        conversion_description = (
            str(conversion_reason).strip()
            if pd.notna(conversion_reason) and str(conversion_reason).strip()
            else DEFAULT_CONVERSION_STATUS_DESCRIPTION
        )
        conversion_description = (
            CONVERSION_STATUS_OUTPUT_INSTRUCTION + conversion_description
        )
        result.append(
            {
                "name": CONVERSION_STATUS_FIELD,
                "description": conversion_description,
                "type": "boolean",
                "defaultValue": False,
            }
        )

    if DISPOSITION_REASON_FIELD not in existing_names:
        result.append(
            {
                "name": DISPOSITION_REASON_FIELD,
                "description": DISPOSITION_REASON_DESCRIPTION,
                "type": "text",
                "defaultValue": "",
            }
        )

    return result


def build_request(row: pd.Series) -> ExtractRequest:
    functions_called = (
        json.loads(row["functions_called"]) if pd.notna(row["functions_called"]) else []
    )
    call_metadata = (
        json.loads(row["call_metadata"]) if pd.notna(row["call_metadata"]) else {}
    )
    postcall_data = json.loads(row["postcall"]) if pd.notna(row["postcall"]) else []
    postcall_data = _add_evaluation_fields(postcall_data, row)

    return ExtractRequest(
        postcall_data=postcall_data,
        transcription=row["transcription"] if pd.notna(row["transcription"]) else "",
        call_duration=(
            float(row["call_duration"]) if pd.notna(row["call_duration"]) else None
        ),
        hangup_reason=row["hangup_reason"] if pd.notna(row["hangup_reason"]) else "",
        functions_called=functions_called,
        call_metadata=call_metadata,
    )


def _strip_think_block(text: str) -> str:
    """Removes a leading <think>...</think> block, matching model_service.py's
    logic — if generation got cut off mid-thought, returns empty string
    rather than misparsing the reasoning text as the answer."""
    match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    if match:
        return text[match.end():].strip()

    open_match = re.search(r"<think>", text)
    if open_match:
        return ""  # unclosed think block, no answer recoverable

    return text.strip()


def _extract_json(text: str) -> dict:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in model output")
    return json.loads(cleaned[start:end + 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--out", type=str, default="eval_report.csv")
    parser.add_argument("--batch-size", type=int, default=8,
                     help="If set, process prompts in chunks of this size instead of all at once")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=GPU_MEMORY_UTILIZATION,
        help=(
            "Fraction of total GPU memory vLLM may reserve (default: shared "
            "VLLM_GPU_MEMORY_UTILIZATION setting, normally 0.5). "
            "Lower this if other processes are using the GPU."
        ),
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help=(
            "Path for this run's summary log. Defaults to a timestamped "
            "filename (eval_summary_<YYYYMMDD_HHMMSS>.log) so runs never "
            "overwrite or mix with each other. Pass an explicit path if "
            "you want a fixed name instead."
        ),
    )
    args = parser.parse_args()

    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be greater than 0 and at most 1")

    run_start = datetime.now()
    log_file = args.log_file or f"eval_summary_{run_start.strftime('%Y%m%d_%H%M%S')}.log"

    validate_runtime_paths()
    lora_request = LoRARequest(ADAPTER_NAME, ADAPTER_ID, str(ADAPTER_DIR))
    print(
        f"Loading vLLM engine for {MODEL_ID} with adapter {ADAPTER_DIR} "
    )
    llm = LLM(
        model=MODEL_ID,
        dtype="bfloat16",
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=MAX_MODEL_LEN,
        enable_lora=True,
        max_loras=1,
        max_lora_rank=MAX_LORA_RANK,
    )
    tokenizer = llm.get_tokenizer()

    df = pd.read_csv(args.csv)
    train_df, test_df = train_test_split(
        df, test_size=args.test_size, random_state=args.random_state, shuffle=True
    )
    print(f"Total rows: {len(df)} -> train: {len(train_df)}, test (evaluated): {len(test_df)}")

    # --- Build every prompt up front ---
    row_indices = []
    prompt_texts = []
    requests_by_row = {}
    error_count = 0

    for i, row in test_df.iterrows():
        try:
            req = build_request(row)
        except Exception as e:
            print(f"Row {i}: SKIPPED (bad row data: {e})")
            error_count += 1
            continue

        prompt = build_prompt(req)
        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=ENABLE_THINKING,
        )

        row_indices.append(i)
        prompt_texts.append(text)
        requests_by_row[i] = req

    print(f"Prepared {len(prompt_texts)} prompts. Running batch generation...")

    # --- One batched call for ALL rows — this is the vLLM speed win ---
    sampling_params = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)
    if args.batch_size:
        outputs = []
        for start in range(0, len(prompt_texts), args.batch_size):
            chunk = prompt_texts[start:start + args.batch_size]
            print(f"Generating batch {start}-{start+len(chunk)} of {len(prompt_texts)}...")
            outputs.extend(
                llm.generate(
                    chunk,
                    sampling_params,
                    lora_request=lora_request,
                )
            )
    else:
        outputs = llm.generate(
            prompt_texts,
            sampling_params,
            lora_request=lora_request,
        )

    print("Generation complete. Scoring...")

    report_rows = []
    skipped_fields_seen = set()

    for i, output in zip(row_indices, outputs):
        row = test_df.loc[i]
        req = requests_by_row[i]
        raw_text = output.outputs[0].text
        answer_text = _strip_think_block(raw_text)

        try:
            model_output = _extract_json(answer_text)
        except Exception as e:
            print(f"Row {i}: ERROR parsing JSON: {e}")
            error_count += 1
            with open("evaluator_errors.log", "a", encoding="utf-8") as f:
                f.write(f"row={i} versionId={row.get('versionId')} error={e}\n\n")
            continue

        # Coerce empty/missing boolean fields to explicit False before scoring,
        # matching TASK_INSTRUCTIONS' own contract ("If a boolean variable is
        # not present, assign it the value false."). Without this, a model
        # that leaves a rarely-true boolean field empty gets scored as wrong
        # against a `false` ground truth even though the two mean the same
        # thing -- see normalize_output.py's module docstring for the full
        # explanation. This only touches empty boolean fields; every other
        # value (including a model's already-explicit answers) passes through
        # unchanged.
        model_output = normalize_model_output(model_output, req.postcall_data)

        schema_names = {v.name for v in req.postcall_data}
        spec_type_by_name = {v.name: v.type for v in req.postcall_data}
        ground_truth = (
            json.loads(row["post_call_detail"]) if pd.notna(row["post_call_detail"]) else {}
        )

        for field_name in schema_names:
            truth_val = evaluation.extract_value(ground_truth, field_name)
            if truth_val == "<missing>":
                continue
            spec_type = spec_type_by_name.get(field_name, "text")
            model_val = evaluation.extract_value(model_output, field_name)
            field_score = evaluation.score_field(spec_type, model_val, truth_val)
            report_rows.append({
                "row_index": i,
                "versionId": row.get("versionId"),
                "field_name": field_name,
                **field_score,
            })

        for key in ground_truth.keys():
            if key not in schema_names:
                skipped_fields_seen.add(key)

    if not report_rows:
        print("\nNo rows produced scorable output.")
        return

    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(args.out, index=False)

    categorical_df = report_df[report_df["category"] == "categorical"]
    text_df = report_df[report_df["category"] == "text"]

    summary_lines = []
    summary_lines.append(f"Run timestamp:          {run_start.isoformat()}")
    summary_lines.append(f"CSV:                    {args.csv}")
    summary_lines.append(f"Model:                  {MODEL_ID}")
    summary_lines.append(f"LoRA adapter:           {ADAPTER_DIR}")
    summary_lines.append(f"Test size / seed:       {args.test_size} / {args.random_state}")
    summary_lines.append("=" * 60)
    summary_lines.append("EVALUATION SUMMARY")
    summary_lines.append("=" * 60)
    summary_lines.append(f"Rows evaluated:        {len(row_indices) - error_count} / {len(test_df)} ({error_count} errored/skipped)")
    summary_lines.append(f"Categorical/boolean:   {categorical_df['score'].mean():.3f}  (n={len(categorical_df)})")
    summary_lines.append(f"Free text (fuzzy):     {text_df['score'].mean():.3f}  (n={len(text_df)})")
    summary_lines.append(f"Mixture (overall):     {report_df['score'].mean():.3f}  (n={len(report_df)})")
    summary_lines.append("")
    summary_lines.append("Per-field average score:")
    summary_lines.append(report_df.groupby(["field_name", "category"])["score"].mean().sort_values().to_string())
    if skipped_fields_seen:
        summary_lines.append(f"\nGround-truth fields NOT in schema (not scored): {sorted(skipped_fields_seen)}")
    summary_lines.append(f"\nFull report written to: {args.out}")
    summary_lines.append(f"Summary log written to: {log_file}")

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    with open(log_file, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")


if __name__ == "__main__":
    main()
