"""Incremental ProofNet Pass@1/8/16/32 orchestration."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from technical.src.benchmarks.proofnet import (
    ProofNetPromptAdapter,
    assemble_proofnet_model_output,
    prepare_unique_rows,
)
from technical.src.evaluation.summarize_passk import summarize
from technical.src.generation.generate import (
    build_attempts,
    create_vllm_backend,
    generate_records,
    load_jsonl,
    to_inference_record,
)
from technical.src.verification.verify_full_header_file import verify_file_record


PASS_TARGETS = (1, 8, 16, 32)
ADD_SAMPLES = (1, 7, 8, 16)
OFFSETS = (0, 1, 8, 16)


def make_attempts(rows: list[dict], add_samples: int, offset: int) -> list[dict]:
    """Expose ProofNet stages through the shared generation-ID implementation."""
    return build_attempts(rows, samples=add_samples, offset=offset)


def run_incremental_stages(rows, generate_stage, verify_stage) -> dict[int, dict]:
    """Run incremental callbacks and return exact cumulative checkpoints."""
    cumulative_records = []
    cumulative_verification = []
    stages = {}
    for target, add_samples, offset in zip(PASS_TARGETS, ADD_SAMPLES, OFFSETS):
        generated = generate_stage(rows, add_samples, offset)
        verified = verify_stage(generated)
        if len(generated) != len(rows) * add_samples:
            raise RuntimeError(f"ProofNet Pass@{target} generation count mismatch")
        if len(verified) != len(generated):
            raise RuntimeError(f"ProofNet Pass@{target} verification count mismatch")
        cumulative_records.extend(generated)
        cumulative_verification.extend(verified)
        if len(cumulative_records) != len(rows) * target:
            raise RuntimeError(f"ProofNet Pass@{target} cumulative count mismatch")
        stages[target] = {
            "full_records": list(cumulative_records),
            "verification": list(cumulative_verification),
        }
    return stages


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--split", default="none")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--revision")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows, duplicate_report = prepare_unique_rows(load_jsonl(args.input, args.split))
    _write_json(args.output_dir / "proofnet_duplicate_problem_ids.json", duplicate_report)
    tokenizer, backend = create_vllm_backend(
        model_path=args.model_path,
        seed=args.seed,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        trust_remote_code=args.trust_remote_code,
        revision=args.revision,
    )

    def generate_stage(stage_rows, add_samples, offset):
        stage_dir = args.output_dir / f"stage_add{add_samples}_offset{offset}"
        records, _ = generate_records(
            stage_rows,
            tokenizer=tokenizer,
            backend=backend,
            adapter=ProofNetPromptAdapter(),
            assembler=assemble_proofnet_model_output,
            output_dir=stage_dir,
            samples=add_samples,
            generation_offset=offset,
            chunk_size=args.chunk_size,
        )
        return records

    def verify_stage(records):
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            return list(
                executor.map(
                    lambda record: verify_file_record(
                        record,
                        workspace=args.workspace,
                        timeout=args.timeout,
                        code_field="full_code",
                    ),
                    records,
                )
            )

    stages = run_incremental_stages(rows, generate_stage, verify_stage)
    for target, stage in stages.items():
        directory = args.output_dir / f"cumulative_pass@{target}"
        _write_json(directory / "full_records.json", stage["full_records"])
        _write_json(
            directory / "to_inference_codes.json",
            [to_inference_record(record) for record in stage["full_records"]],
        )
        _write_json(directory / "code_compilation_full_header.json", stage["verification"])
        maps = {record["problem_id"]: record["id_maps"] for record in stage["full_records"]}
        _write_json(directory / "summary.json", summarize(stage["verification"], maps))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
