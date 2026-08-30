"""Consolidated Putnam JSONL splitting, merging, and progress reporting."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path


GENERATION_SUFFIX = re.compile(r"_g\d+$")
GENERATION_ID = re.compile(r"^(?P<origin>.+)_g(?P<generation>\d+)$")


def chunk_bounds(total: int, chunks: int, index: int) -> tuple[int, int]:
    """Return balanced half-open bounds for one zero-based chunk."""
    if total < 0:
        raise ValueError("total must be non-negative")
    if chunks < 1:
        raise ValueError("chunks must be positive")
    if not 0 <= index < chunks:
        raise ValueError("chunk index must satisfy 0 <= index < chunks")
    base, extra = divmod(total, chunks)
    start = index * base + min(index, extra)
    size = base + int(index < extra)
    return start, start + size


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def split_jsonl(
    input_path: Path, output_path: Path, *, chunks: int, index: int
) -> list[dict]:
    rows = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("JSONL input must contain objects")
    start, end = chunk_bounds(len(rows), chunks, index)
    selected = rows[start:end]
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected)
    _atomic_write_text(output_path, text)
    return selected


def _load_json_list(path: Path) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"{path} must contain a JSON array of objects")
    return value


def merge_json_lists(
    inputs: list[Path],
    *,
    dedupe_key: str | None = None,
    on_duplicate: str = "error",
    output_path: Path | None = None,
) -> list[dict]:
    """Merge arrays stably with explicit duplicate handling."""
    if on_duplicate not in {"error", "first", "last"}:
        raise ValueError(f"unsupported duplicate policy: {on_duplicate}")
    merged = []
    positions = {}
    for path in inputs:
        for row in _load_json_list(path):
            if dedupe_key is None:
                merged.append(row)
                continue
            if dedupe_key not in row:
                raise ValueError(f"record is missing dedupe key: {dedupe_key}")
            value = row[dedupe_key]
            if value not in positions:
                positions[value] = len(merged)
                merged.append(row)
            elif on_duplicate == "error":
                raise ValueError(f"duplicate {dedupe_key}: {value}")
            elif on_duplicate == "last":
                merged[positions[value]] = row
    if output_path is not None:
        _atomic_write_text(
            output_path,
            json.dumps(merged, ensure_ascii=False, indent=2) + "\n",
        )
    return merged


def _record_id(record: dict, key: str) -> str:
    value = record.get(key)
    if value is None:
        raise ValueError(f"record is missing {key}")
    return str(value)


def _validate_generation_prefix(records: list[dict], key: str, target_pass: int) -> None:
    generations: dict[str, set[int]] = {}
    seen_ids: set[str] = set()
    for record in records:
        identifier = _record_id(record, key)
        if identifier in seen_ids:
            raise ValueError(f"duplicate generation ID: {identifier}")
        seen_ids.add(identifier)
        match = GENERATION_ID.fullmatch(identifier)
        if match is None:
            raise ValueError(f"invalid generation ID: {identifier}")
        generations.setdefault(match.group("origin"), set()).add(
            int(match.group("generation"))
        )
    expected = set(range(target_pass))
    for origin, actual in generations.items():
        if actual != expected:
            raise ValueError(
                f"{origin} expected generations 0..{target_pass - 1}, got {sorted(actual)}"
            )


def merge_putnam_stages(
    full_inputs: list[Path],
    inference_inputs: list[Path],
    verification_inputs: list[Path],
    *,
    target_pass: int,
    output_dir: Path,
) -> dict:
    """Atomically materialize one exact cumulative Putnam checkpoint."""
    if target_pass < 1:
        raise ValueError("target_pass must be positive")
    if not (len(full_inputs) == len(inference_inputs) == len(verification_inputs)):
        raise ValueError("stage input groups must have equal lengths")
    full = merge_json_lists(full_inputs)
    inference = merge_json_lists(inference_inputs)
    verification = merge_json_lists(verification_inputs)
    _validate_generation_prefix(full, "problem_id", target_pass)
    _validate_generation_prefix(inference, "problem_id", target_pass)
    _validate_generation_prefix(verification, "name", target_pass)
    full_ids = {_record_id(record, "problem_id") for record in full}
    inference_ids = {_record_id(record, "problem_id") for record in inference}
    verification_ids = {_record_id(record, "name") for record in verification}
    if full_ids != inference_ids or full_ids != verification_ids:
        raise ValueError("full, inference, and verification generation IDs differ")
    _atomic_write_text(
        output_dir / "full_records.json",
        json.dumps(full, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_write_text(
        output_dir / "to_inference_codes.json",
        json.dumps(inference, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_write_text(
        output_dir / "code_compilation_full_header.json",
        json.dumps(verification, ensure_ascii=False, indent=2) + "\n",
    )
    origins = {GENERATION_ID.fullmatch(identifier).group("origin") for identifier in full_ids}
    return {"problem_num": len(origins), "attempts": len(full), "target_pass": target_pass}


def summarize_chunk_progress(
    outputs: dict[int, Path], *, total_chunks: int, target_pass: int
) -> dict:
    """Summarize available terminal compilation arrays by original problem."""
    if total_chunks < 1:
        raise ValueError("total_chunks must be positive")
    invalid = sorted(index for index in outputs if not 0 <= index < total_chunks)
    if invalid:
        raise ValueError(f"chunk indices outside configured range: {invalid}")
    complete_chunks = sorted(index for index, path in outputs.items() if path.is_file())
    records = [
        record
        for index in complete_chunks
        for record in _load_json_list(outputs[index])
    ]
    solved: dict[str, bool] = {}
    for record in records:
        name = record.get("name") or record.get("problem_id")
        if name is None:
            raise ValueError("compilation record is missing a problem identifier")
        origin = GENERATION_SUFFIX.sub("", str(name))
        complete = bool(record.get("compilation_result", {}).get("complete"))
        solved[origin] = solved.get(origin, False) or complete
    return {
        "target_pass": target_pass,
        "complete_chunks": complete_chunks,
        "missing_chunks": [
            index for index in range(total_chunks) if index not in complete_chunks
        ],
        "terminal_attempts": len(records),
        "problem_num": len(solved),
        "solved_num": sum(solved.values()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    split = subparsers.add_parser("split")
    split.add_argument("--input", type=Path, required=True)
    split.add_argument("--output", type=Path, required=True)
    split.add_argument("--chunks", type=int, required=True)
    split.add_argument("--index", type=int, required=True)

    merge = subparsers.add_parser("merge")
    merge.add_argument("--input", type=Path, action="append", required=True)
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--dedupe-key")
    merge.add_argument("--on-duplicate", choices=("error", "first", "last"), default="error")

    progress = subparsers.add_parser("progress")
    progress.add_argument("--chunk", action="append", required=True, help="INDEX=PATH")
    progress.add_argument("--total-chunks", type=int, required=True)
    progress.add_argument("--target-pass", type=int, required=True)
    progress.add_argument("--output", type=Path, required=True)

    cumulative = subparsers.add_parser("cumulative")
    cumulative.add_argument("--full", type=Path, action="append", required=True)
    cumulative.add_argument("--inference", type=Path, action="append", required=True)
    cumulative.add_argument("--verification", type=Path, action="append", required=True)
    cumulative.add_argument("--target-pass", type=int, required=True)
    cumulative.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "split":
        split_jsonl(args.input, args.output, chunks=args.chunks, index=args.index)
    elif args.command == "merge":
        merge_json_lists(
            args.input,
            dedupe_key=args.dedupe_key,
            on_duplicate=args.on_duplicate,
            output_path=args.output,
        )
    elif args.command == "progress":
        outputs = {}
        for specification in args.chunk:
            index, separator, path = specification.partition("=")
            if not separator:
                raise ValueError("chunk arguments must use INDEX=PATH")
            outputs[int(index)] = Path(path)
        summary = summarize_chunk_progress(
            outputs, total_chunks=args.total_chunks, target_pass=args.target_pass
        )
        _atomic_write_text(
            args.output, json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
        )
    else:
        merge_putnam_stages(
            args.full,
            args.inference,
            args.verification,
            target_pass=args.target_pass,
            output_dir=args.output_dir,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
