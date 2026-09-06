#!/usr/bin/env python

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge DeepSeek generation or verification records."
    )
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--input-format", choices=("jsonl", "json"), required=True)
    parser.add_argument("--output-format", choices=("jsonl", "json"), default=None)
    parser.add_argument("--dedupe-key", default="problem_id")
    parser.add_argument("--expected-problems", type=int, required=True)
    parser.add_argument("--expected-samples", type=int, required=True)
    parser.add_argument("input_paths", type=Path, nargs="+")
    return parser.parse_args()


def load_records(path: Path, input_format: str) -> list[dict[str, Any]]:
    if input_format == "jsonl":
        records = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
        return records

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} is not a JSON list")
    return data


def generation_id(record: dict[str, Any]) -> int:
    if "generation_id" in record:
        return int(record["generation_id"])
    problem_id = str(record.get("problem_id") or record.get("name"))
    if "_g" not in problem_id:
        raise ValueError(f"Cannot infer generation_id for {problem_id}")
    return int(problem_id.rsplit("_g", 1)[1])


def origin_id(record: dict[str, Any]) -> str:
    if record.get("origin_problem_id"):
        return str(record["origin_problem_id"])
    problem_id = str(record.get("problem_id") or record.get("name"))
    return problem_id.rsplit("_g", 1)[0]


def sort_key(record: dict[str, Any]) -> tuple[int, str, int]:
    source_index = record.get("source_index")
    return (
        int(source_index) if source_index is not None else 10**9,
        origin_id(record),
        generation_id(record),
    )


def main() -> None:
    args = parse_args()
    output_format = args.output_format or args.input_format
    if args.expected_problems < 1:
        raise ValueError("--expected-problems must be positive")
    if args.expected_samples < 1:
        raise ValueError("--expected-samples must be positive")

    records_by_key: dict[str, dict[str, Any]] = {}
    grouped: dict[str, set[int]] = defaultdict(set)

    for path in args.input_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        for record in load_records(path, args.input_format):
            key_value = record.get(args.dedupe_key)
            if key_value is None:
                raise KeyError(f"{path} record is missing {args.dedupe_key}")
            key = str(key_value)
            if key in records_by_key:
                raise ValueError(f"Duplicate {args.dedupe_key}: {key}")
            records_by_key[key] = record
            grouped[origin_id(record)].add(generation_id(record))

    if len(grouped) != args.expected_problems:
        raise RuntimeError(
            f"Expected {args.expected_problems} problems, found {len(grouped)}"
        )
    expected_generations = set(range(args.expected_samples))
    for problem, generations in grouped.items():
        if generations != expected_generations:
            raise RuntimeError(
                f"{problem} has generations {sorted(generations)}, "
                f"expected {sorted(expected_generations)}"
            )

    ordered = sorted(records_by_key.values(), key=sort_key)
    expected_records = args.expected_problems * args.expected_samples
    if len(ordered) != expected_records:
        raise RuntimeError(f"Expected {expected_records} records, found {len(ordered)}")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = args.output_path.with_suffix(
        args.output_path.suffix + f".{os.getpid()}.tmp"
    )
    if output_format == "jsonl":
        with temporary_path.open("w", encoding="utf-8") as handle:
            for record in ordered:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    else:
        temporary_path.write_text(
            json.dumps(ordered, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    temporary_path.replace(args.output_path)
    print(
        f"Merged {len(args.input_paths)} files, {len(grouped)} problems, "
        f"{len(ordered)} records into {args.output_path}"
    )


if __name__ == "__main__":
    main()
