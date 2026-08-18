from runtime_config import (
    ENABLE_THINKING,
    GPU_MEMORY_UTILIZATION,
    MAX_MODEL_LEN,
    MAX_NEW_TOKENS,
    MODEL_ID,
)

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from vllm import LLM, SamplingParams

import evaluation
from normalize_output import normalize_model_output
from prompt_builder import SYSTEM_INSTRUCTION, build_prompt
from schemas import PromptBuildRequest


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


def _boolean_label(value):
    """Return a real boolean only for explicit true/false report values."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    return None


def _build_boolean_metrics(report_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    categorical = report_df[report_df["category"] == "categorical"]
    for field_name, group in categorical.groupby("field_name"):
        truths = group["truth_value"].map(_boolean_label)
        if truths.isna().any():
            continue

        predictions = group["model_value"].map(_boolean_label)
        positive = truths == True  # noqa: E712 - intentional pandas comparison
        negative = truths == False  # noqa: E712
        tp = int(((predictions == True) & positive).sum())  # noqa: E712
        fn = int(((predictions != True) & positive).sum())  # noqa: E712
        tn = int(((predictions == False) & negative).sum())  # noqa: E712
        fp = int(((predictions != False) & negative).sum())  # noqa: E712
        recall = tp / (tp + fn) if tp + fn else None
        specificity = tn / (tn + fp) if tn + fp else None
        balanced_accuracy = (
            (recall + specificity) / 2
            if recall is not None and specificity is not None
            else None
        )
        rows.append(
            {
                "field_name": field_name,
                "n": len(group),
                "positive_support": int(positive.sum()),
                "negative_support": int(negative.sum()),
                "tp": tp,
                "fn": fn,
                "tn": tn,
                "fp": fp,
                "positive_recall": recall,
                "specificity": specificity,
                "balanced_accuracy": balanced_accuracy,
                "accuracy": float(group["match"].mean()),
            }
        )
    return pd.DataFrame(rows)


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


def build_request(row: pd.Series) -> PromptBuildRequest:
    functions_called = (
        json.loads(row["functions_called"]) if pd.notna(row["functions_called"]) else []
    )
    call_metadata = (
        json.loads(row["call_metadata"]) if pd.notna(row["call_metadata"]) else {}
    )
    postcall_data = json.loads(row["postcall"]) if pd.notna(row["postcall"]) else []
    postcall_data = _add_evaluation_fields(postcall_data, row)

    return PromptBuildRequest(
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
            "Path for this run's evaluation log. By default it is written "
            "beside --out with the same stem and a .log extension."
        ),
    )
    args = parser.parse_args()

    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be greater than 0 and at most 1")

    run_start = datetime.now()
    report_path = Path(args.out)
    log_path = (
        Path(args.log_file)
        if args.log_file
        else report_path.with_suffix(".log")
    )

    print(f"Loading vLLM engine for {MODEL_ID} ...")
    llm = LLM(
        model=MODEL_ID,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=MAX_MODEL_LEN,
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
                )
            )
    else:
        outputs = llm.generate(
            prompt_texts,
            sampling_params,
        )

    print("Generation complete. Scoring...")

    report_rows = []
    skipped_fields_seen = set()
    invalid_ground_truth_counts = {}
    successfully_scored_rows = 0

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

        row_had_score = False
        for field_name in schema_names:
            truth_val = evaluation.extract_value(ground_truth, field_name)
            if truth_val == "<missing>":
                continue
            if evaluation.should_skip_ground_truth(field_name, truth_val):
                invalid_ground_truth_counts[field_name] = (
                    invalid_ground_truth_counts.get(field_name, 0) + 1
                )
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
            row_had_score = True

        if row_had_score:
            successfully_scored_rows += 1

        for key in ground_truth.keys():
            if key not in schema_names:
                skipped_fields_seen.add(key)

    if not report_rows:
        print("\nNo rows produced scorable output.")
        return

    report_df = pd.DataFrame(report_rows)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(report_path, index=False)

    categorical_df = report_df[report_df["category"] == "categorical"]
    text_df = report_df[report_df["category"] == "text"]
    nonempty_text_df = text_df[
        ~text_df["truth_value"].map(evaluation._is_empty)
    ]
    per_field = (
        report_df.groupby(["field_name", "category"])
        .agg(
            samples=("score", "size"),
            average_score=("score", "mean"),
            match_rate=("match", "mean"),
        )
        .sort_values(["match_rate", "average_score"])
    )
    boolean_metrics = _build_boolean_metrics(report_df)
    boolean_metrics_path = report_path.with_name(
        f"{report_path.stem}_boolean_metrics.csv"
    )
    boolean_metrics.to_csv(boolean_metrics_path, index=False)

    summary_lines = []
    summary_lines.append(f"Run timestamp:          {run_start.isoformat()}")
    summary_lines.append(f"CSV:                    {args.csv}")
    summary_lines.append(f"Model:                  {MODEL_ID}")
    summary_lines.append(f"Test size / seed:       {args.test_size} / {args.random_state}")
    summary_lines.append("=" * 60)
    summary_lines.append("EVALUATION SUMMARY")
    summary_lines.append("=" * 60)
    summary_lines.append(f"Rows evaluated:        {successfully_scored_rows} / {len(test_df)} ({error_count} errored/skipped)")
    summary_lines.append(f"Categorical/boolean:   {categorical_df['score'].mean():.3f}  (n={len(categorical_df)})")
    summary_lines.append(f"Free text (fuzzy):     {text_df['score'].mean():.3f}  (n={len(text_df)})")
    summary_lines.append(
        f"Non-empty free text:   {nonempty_text_df['score'].mean():.3f}  "
        f"(match={nonempty_text_df['match'].mean():.3f}, "
        f"n={len(nonempty_text_df)})"
    )
    summary_lines.append(f"Mixture (overall):     {report_df['score'].mean():.3f}  (n={len(report_df)})")
    summary_lines.append(
        f"Macro field match:     {per_field['match_rate'].mean():.3f}  "
        f"(fields={len(per_field)})"
    )
    summary_lines.append("")
    summary_lines.append("Per-field metrics:")
    summary_lines.append(per_field.to_string())
    if not boolean_metrics.empty:
        summary_lines.append("\nBoolean fields with both truth classes:")
        two_class = boolean_metrics[
            (boolean_metrics["positive_support"] > 0)
            & (boolean_metrics["negative_support"] > 0)
        ].sort_values("balanced_accuracy")
        summary_lines.append(
            two_class[
                [
                    "field_name",
                    "positive_support",
                    "negative_support",
                    "positive_recall",
                    "specificity",
                    "balanced_accuracy",
                    "accuracy",
                ]
            ].to_string(index=False)
            if not two_class.empty
            else "(none in this sample)"
        )
    if invalid_ground_truth_counts:
        summary_lines.append(
            "\nInvalid ground-truth labels skipped: "
            f"{invalid_ground_truth_counts}"
        )
    if skipped_fields_seen:
        summary_lines.append(f"\nGround-truth fields NOT in schema (not scored): {sorted(skipped_fields_seen)}")
    summary_lines.append(f"\nFull report written to: {report_path}")
    summary_lines.append(f"Boolean metrics written to: {boolean_metrics_path}")
    summary_lines.append(f"Evaluation log written to: {log_path}")

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    mismatch_df = report_df[~report_df["match"]]
    with log_path.open("w", encoding="utf-8") as f:
        f.write(summary_text + "\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write("MISMATCH DETAILS\n")
        f.write("=" * 60 + "\n")
        if mismatch_df.empty:
            f.write("No mismatches.\n")
        else:
            f.write(mismatch_df.to_string(index=False) + "\n")


if __name__ == "__main__":
    main()
