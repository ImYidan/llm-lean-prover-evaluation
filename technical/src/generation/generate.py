"""Shared generation runner for portable Lean prover evaluation."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable

from technical.src.generation.adapters.goedel import GoedelPromptAdapter
from technical.src.generation.candidates import Candidate
from technical.src.generation.proof_extraction import (
    ProofAssemblyError,
    assemble_standard_submission,
)


def build_attempts(rows: list[dict], samples: int, offset: int) -> list[dict]:
    """Expand benchmark rows into stable ``_gN`` generation attempts."""
    if samples < 1:
        raise ValueError("samples must be positive")
    if offset < 0:
        raise ValueError("generation offset must be non-negative")

    attempts = []
    for row in rows:
        if not row.get("lean4_code"):
            continue
        origin_id = row.get("origin_problem_id") or row.get("problem_id") or row.get("name")
        if origin_id is None:
            raise ValueError("row is missing a problem identifier")
        for sample_index in range(samples):
            generation_id = f"{origin_id}_g{offset + sample_index}"
            attempt = dict(row)
            attempt.update(
                {
                    "origin_problem_id": origin_id,
                    "problem_id": generation_id,
                    "id_maps": [
                        {"origin_problem_id": origin_id},
                        {"generation_id": generation_id},
                    ],
                }
            )
            attempts.append(attempt)
    return attempts


def _atomic_write_json(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(records, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _chunks(items: list[dict], size: int) -> Iterable[list[dict]]:
    if size < 1:
        raise ValueError("chunk size must be positive")
    for start in range(0, len(items), size):
        yield items[start : start + size]


def to_inference_record(record: dict) -> dict:
    return {
        "problem_id": record["problem_id"],
        "origin_problem_id": record["origin_problem_id"],
        "id_maps": record["id_maps"],
        "lean4_code": record["lean4_code"],
        "model_input": record["model_input"],
        "messages_history_list": record["messages_history_for_this_attempt"],
        "model_output": record["model_output"],
        "full_code": record["full_code"],
    }


def generate_records(
    rows: list[dict],
    *,
    tokenizer,
    backend,
    adapter: GoedelPromptAdapter,
    assembler=assemble_standard_submission,
    output_dir: Path,
    samples: int,
    generation_offset: int = 0,
    chunk_size: int = 128,
) -> tuple[list[dict], list[dict]]:
    """Generate attempts and checkpoint the two historical JSON-array files."""
    attempts = build_attempts(rows, samples=samples, offset=generation_offset)
    full_records: list[dict] = []
    inference_records: list[dict] = []
    output_dir = Path(output_dir)

    for chunk in _chunks(attempts, chunk_size):
        prepared = []
        for attempt in chunk:
            prompt, messages = adapter.build_prompt(attempt["lean4_code"], tokenizer)
            prepared.append((attempt, prompt, messages))

        outputs = backend.generate([prompt for _, prompt, _ in prepared])
        if len(outputs) != len(prepared):
            raise RuntimeError("generation backend returned the wrong number of outputs")

        for (attempt, prompt, messages), candidates in zip(prepared, outputs):
            if len(candidates) != 1:
                raise RuntimeError(
                    "Goedel generation requires one candidate per request"
                )
            model_output = candidates[0].text
            record = dict(attempt)
            record["model_input"] = prompt
            record["messages_history_for_this_attempt"] = messages
            record["model_output"] = model_output
            try:
                assembled = assembler(record["lean4_code"], model_output)
            except ProofAssemblyError:
                assembled = None
            record["full_code"] = assembled if assembled is not None else "None"
            full_records.append(record)
            inference_records.append(to_inference_record(record))

        _atomic_write_json(output_dir / "full_records.json", full_records)
        _atomic_write_json(
            output_dir / "to_inference_codes.json", inference_records
        )

    if not attempts:
        _atomic_write_json(output_dir / "full_records.json", [])
        _atomic_write_json(output_dir / "to_inference_codes.json", [])
    return full_records, inference_records


def load_jsonl(path: Path, split: str | None = None) -> list[dict]:
    """Load benchmark JSONL, optionally retaining rows from one split."""
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row {line_number} must be an object")
            if split in (None, "none") or row.get("split") == split:
                rows.append(row)
    return rows


class _VllmBackend:
    def __init__(self, model, sampling_params) -> None:
        self.model = model
        self.sampling_params = sampling_params

    def generate(self, prompts: list[str]) -> list[list[Candidate]]:
        requests = self.model.generate(prompts, self.sampling_params)
        return [
            [
                Candidate(item.text, item.finish_reason, item.stop_reason)
                for item in request.outputs
            ]
            for request in requests
        ]


def create_vllm_backend(
    *,
    model_path: str,
    seed: int,
    tensor_parallel_size: int,
    max_model_len: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    samples_per_prompt: int = 1,
    trust_remote_code: bool = False,
    revision: str | None = None,
):
    """Create real tokenizer/backend objects while keeping imports lazy."""
    validate_generation_options(
        samples=samples_per_prompt,
        max_model_len=max_model_len,
        max_tokens=max_tokens,
    )
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=trust_remote_code, revision=revision
    )
    model = LLM(
        model=model_path,
        seed=seed,
        trust_remote_code=trust_remote_code,
        revision=revision,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
    )
    params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n=samples_per_prompt,
    )
    return tokenizer, _VllmBackend(model, params)


def validate_generation_options(
    *, samples: int, max_model_len: int, max_tokens: int
) -> None:
    """Reject invalid generation limits before loading optional model backends."""
    if samples < 1:
        raise ValueError("samples must be positive")
    if max_tokens >= max_model_len:
        raise ValueError("max_tokens must be less than max_model_len")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adapter", choices=("goedel", "deepseek"), default="goedel")
    parser.add_argument("--split", default="none")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--generation-offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-tokens", type=int, default=32767)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--revision")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validate_generation_options(
            samples=args.samples,
            max_model_len=args.max_model_len,
            max_tokens=args.max_tokens,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    rows = load_jsonl(args.input, args.split)
    tokenizer, backend = create_vllm_backend(
        model_path=args.model_path,
        seed=args.seed,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        samples_per_prompt=args.samples if args.adapter == "deepseek" else 1,
        trust_remote_code=args.trust_remote_code,
        revision=args.revision,
    )
    if args.adapter == "deepseek":
        from technical.src.generation.deepseek import generate_deepseek_records

        generate_deepseek_records(
            rows,
            tokenizer=tokenizer,
            backend=backend,
            output_dir=args.output_dir,
            samples=args.samples,
            generation_offset=args.generation_offset,
        )
        return 0
    generate_records(
        rows,
        tokenizer=tokenizer,
        backend=backend,
        adapter=GoedelPromptAdapter(),
        output_dir=args.output_dir,
        samples=args.samples,
        generation_offset=args.generation_offset,
        chunk_size=args.chunk_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
