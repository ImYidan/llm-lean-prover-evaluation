"""Grouped DeepSeek generation with resumable raw checkpoints."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from technical.src.generation.adapters.deepseek import (
    DeepSeekPromptAdapter,
    assemble_deepseek_submission,
)
from technical.src.generation.candidates import Candidate
from technical.src.generation.checkpoints import (
    CheckpointWriter,
    load_checkpoint,
    truncate_checkpoint,
)
from technical.src.generation.generate import (
    _atomic_write_json,
    build_attempts,
    to_inference_record,
)
from technical.src.generation.proof_extraction import ProofAssemblyError


_REQUIRED_RAW_CHECKPOINT_FIELDS = (
    "origin_problem_id",
    "id_maps",
    "lean4_code",
    "model_input",
    "messages_history_for_this_attempt",
    "model_output",
    "raw_response",
    "full_code",
    "extraction_status",
    "finish_reason",
    "stop_reason",
)


def _validate_checkpoint_prefix(
    records: list[dict], expected: list[dict], samples: int
) -> None:
    if len(records) > len(expected):
        raise ValueError(
            f"checkpoint has {len(records)} rows but expected {len(expected)}"
        )

    for index, (record, attempt) in enumerate(zip(records, expected), start=1):
        for field in _REQUIRED_RAW_CHECKPOINT_FIELDS:
            if field not in record:
                raise ValueError(
                    f"checkpoint row {index} missing required field {field}"
                )
        for field in (
            "model_input",
            "model_output",
            "raw_response",
            "full_code",
            "extraction_status",
        ):
            if not isinstance(record[field], str):
                raise ValueError(f"checkpoint row {index} {field} must be a string")
        messages = record["messages_history_for_this_attempt"]
        if not isinstance(messages, list):
            raise ValueError(
                f"checkpoint row {index} messages_history_for_this_attempt "
                "must be a list"
            )
        if not all(isinstance(message, dict) for message in messages):
            raise ValueError(
                f"checkpoint row {index} messages_history_for_this_attempt "
                "must contain objects"
            )
        if record["finish_reason"] is not None and not isinstance(
            record["finish_reason"], str
        ):
            raise ValueError(
                f"checkpoint row {index} finish_reason must be a string or null"
            )
        stop_reason = record["stop_reason"]
        if stop_reason is not None and (
            isinstance(stop_reason, bool)
            or not isinstance(stop_reason, (str, int))
        ):
            raise ValueError(
                f"checkpoint row {index} stop_reason must be a string, integer, or null"
            )
        if record["raw_response"] != record["model_output"]:
            raise ValueError(
                f"checkpoint row {index} raw_response must equal model_output"
            )
        extraction_succeeded = record["extraction_status"] == "success"
        if extraction_succeeded and record["full_code"] == "None":
            raise ValueError(
                f"checkpoint row {index} successful extraction must contain full_code"
            )
        if not extraction_succeeded and record["full_code"] != "None":
            raise ValueError(
                f"checkpoint row {index} failed extraction must use full_code None"
            )
        expected_problem_id = attempt["problem_id"]
        if record.get("problem_id") != expected_problem_id:
            raise ValueError(
                f"checkpoint row {index} expected {expected_problem_id} "
                f"but found {record.get('problem_id')}"
            )
        expected_origin_id = attempt["origin_problem_id"]
        if record["origin_problem_id"] != expected_origin_id:
            raise ValueError(
                f"checkpoint row {index} expected origin {expected_origin_id}"
            )
        if record["lean4_code"] != attempt["lean4_code"]:
            raise ValueError(
                f"checkpoint row {index} does not match the source Lean code"
            )
        if record["id_maps"] != attempt["id_maps"]:
            raise ValueError(f"checkpoint row {index} has unexpected id_maps")


def _write_normalized_outputs(output_dir: Path, full_records: list[dict]) -> None:
    _atomic_write_json(output_dir / "full_records.json", full_records)
    _atomic_write_json(
        output_dir / "to_inference_codes.json",
        [to_inference_record(record) for record in full_records],
    )


def _record_from_candidate(
    *,
    attempt: dict,
    prompt: str,
    messages: list[dict],
    candidate: Candidate,
    assembler: Callable[[str, str], str | None],
) -> dict:
    model_output = candidate.text
    record = dict(attempt)
    record["model_input"] = prompt
    record["messages_history_for_this_attempt"] = messages
    record["model_output"] = model_output
    record["raw_response"] = model_output
    record["finish_reason"] = candidate.finish_reason
    record["stop_reason"] = candidate.stop_reason
    try:
        assembled = assembler(attempt["lean4_code"], model_output)
    except ProofAssemblyError as error:
        record["full_code"] = "None"
        record["extraction_status"] = error.condition
        if error.tactic is not None:
            record["extraction_tactic"] = error.tactic
    else:
        if assembled is None:
            record["full_code"] = "None"
            record["extraction_status"] = "missing fenced Lean code block"
        else:
            record["full_code"] = assembled
            record["extraction_status"] = "success"
    return record


def generate_grouped_records(
    rows: list[dict],
    *,
    tokenizer,
    backend,
    adapter,
    output_dir: Path,
    samples: int,
    generation_offset: int = 0,
    assembler: Callable[[str, str], str | None] = assemble_deepseek_submission,
) -> tuple[list[dict], list[dict]]:
    """Generate grouped candidates and resume from a validated raw prefix."""
    expected = build_attempts(rows, samples=samples, offset=generation_offset)
    output_dir = Path(output_dir)
    checkpoint_path = output_dir / "inference.jsonl"
    full_records = load_checkpoint(checkpoint_path)
    _validate_checkpoint_prefix(full_records, expected, samples)
    incomplete_candidates = len(full_records) % samples
    if incomplete_candidates:
        keep_records = len(full_records) - incomplete_candidates
        truncate_checkpoint(checkpoint_path, keep_records)
        full_records = full_records[:keep_records]
    checkpoint_writer = CheckpointWriter(checkpoint_path, records=full_records)

    next_source_index = len(full_records) // samples
    for source_index in range(next_source_index, len(expected) // samples):
        group = expected[source_index * samples : (source_index + 1) * samples]
        source_attempt = group[0]
        prompt, messages = adapter.build_prompt(source_attempt, tokenizer)
        outputs = backend.generate([prompt])
        if len(outputs) != 1:
            raise RuntimeError(
                "grouped generation backend returned the wrong number of request groups"
            )
        candidates = outputs[0]
        if len(candidates) != samples:
            raise RuntimeError(
                f"grouped generation expected {samples} candidates "
                f"but received {len(candidates)}"
            )

        for attempt, candidate in zip(group, candidates):
            record = _record_from_candidate(
                attempt=attempt,
                prompt=prompt,
                messages=messages,
                candidate=candidate,
                assembler=assembler,
            )
            checkpoint_writer.append(record)
            full_records.append(record)
        _write_normalized_outputs(output_dir, full_records)

    if not expected:
        _write_normalized_outputs(output_dir, [])
    elif len(full_records) == len(expected):
        _write_normalized_outputs(output_dir, full_records)
    return full_records, [to_inference_record(record) for record in full_records]


def generate_deepseek_records(
    rows: list[dict],
    *,
    tokenizer,
    backend,
    output_dir: Path,
    samples: int,
    generation_offset: int = 0,
    assembler: Callable[[str, str], str | None] = assemble_deepseek_submission,
) -> tuple[list[dict], list[dict]]:
    """Backward-compatible DeepSeek entry point."""
    return generate_grouped_records(
        rows,
        tokenizer=tokenizer,
        backend=backend,
        adapter=DeepSeekPromptAdapter(),
        output_dir=output_dir,
        samples=samples,
        generation_offset=generation_offset,
        assembler=assembler,
    )
