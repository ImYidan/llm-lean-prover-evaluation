"""Grouped DeepSeek generation with resumable raw checkpoints."""

from __future__ import annotations

from pathlib import Path

from technical.src.generation.adapters.deepseek import (
    DeepSeekPromptAdapter,
    assemble_deepseek_submission,
)
from technical.src.generation.candidates import Candidate
from technical.src.generation.checkpoints import append_checkpoint, load_checkpoint
from technical.src.generation.generate import (
    _atomic_write_json,
    build_attempts,
    to_inference_record,
)
from technical.src.generation.proof_extraction import ProofAssemblyError


def _validate_checkpoint_prefix(
    records: list[dict], expected: list[dict], samples: int
) -> None:
    if len(records) > len(expected):
        raise ValueError(
            f"checkpoint has {len(records)} rows but expected {len(expected)}"
        )

    for index, (record, attempt) in enumerate(zip(records, expected), start=1):
        expected_problem_id = attempt["problem_id"]
        if record.get("problem_id") != expected_problem_id:
            raise ValueError(
                f"checkpoint row {index} expected {expected_problem_id} "
                f"but found {record.get('problem_id')}"
            )
        expected_origin_id = attempt["origin_problem_id"]
        if (
            "origin_problem_id" in record
            and record["origin_problem_id"] != expected_origin_id
        ):
            raise ValueError(
                f"checkpoint row {index} expected origin {expected_origin_id}"
            )
        if "lean4_code" in record and record["lean4_code"] != attempt["lean4_code"]:
            raise ValueError(
                f"checkpoint row {index} does not match the source Lean code"
            )
        if "id_maps" in record and record["id_maps"] != attempt["id_maps"]:
            raise ValueError(f"checkpoint row {index} has unexpected id_maps")

    if len(records) % samples != 0:
        raise ValueError(
            "checkpoint must end after a complete source problem candidate group"
        )


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
        record["full_code"] = assemble_deepseek_submission(
            attempt["lean4_code"], model_output
        )
        record["extraction_status"] = "success"
    except ProofAssemblyError as error:
        record["full_code"] = "None"
        record["extraction_status"] = error.condition
        if error.tactic is not None:
            record["extraction_tactic"] = error.tactic
    return record


def generate_deepseek_records(
    rows: list[dict],
    *,
    tokenizer,
    backend,
    output_dir: Path,
    samples: int,
    generation_offset: int = 0,
) -> tuple[list[dict], list[dict]]:
    """Generate grouped DeepSeek candidates and resume from a valid raw prefix."""
    expected = build_attempts(rows, samples=samples, offset=generation_offset)
    output_dir = Path(output_dir)
    checkpoint_path = output_dir / "inference.jsonl"
    full_records = load_checkpoint(checkpoint_path)
    _validate_checkpoint_prefix(full_records, expected, samples)

    next_source_index = len(full_records) // samples
    adapter = DeepSeekPromptAdapter()

    for source_index in range(next_source_index, len(expected) // samples):
        group = expected[source_index * samples : (source_index + 1) * samples]
        source_attempt = group[0]
        prompt, messages = adapter.build_prompt(source_attempt, tokenizer)
        outputs = backend.generate([prompt])
        if len(outputs) != 1:
            raise RuntimeError(
                "DeepSeek generation backend returned the wrong number of request groups"
            )
        candidates = outputs[0]
        if len(candidates) != samples:
            raise RuntimeError(
                f"DeepSeek generation expected {samples} candidates "
                f"but received {len(candidates)}"
            )

        for attempt, candidate in zip(group, candidates):
            record = _record_from_candidate(
                attempt=attempt,
                prompt=prompt,
                messages=messages,
                candidate=candidate,
            )
            append_checkpoint(checkpoint_path, record)
            full_records.append(record)
        _write_normalized_outputs(output_dir, full_records)

    if not expected:
        _write_normalized_outputs(output_dir, [])
    elif len(full_records) == len(expected):
        _write_normalized_outputs(output_dir, full_records)
    return full_records, [to_inference_record(record) for record in full_records]
