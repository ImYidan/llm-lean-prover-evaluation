#!/usr/bin/env python

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--expected-problems", type=int, required=True)
    parser.add_argument("--expected-samples", type=int, required=True)
    return parser.parse_args()


def normalized_record(
    record: dict[str, Any], duplicate_origins: set[str]
) -> dict[str, Any]:
    origin_id = record["origin_problem_id"]
    if origin_id not in duplicate_origins:
        return record

    source_index = record.get("source_index")
    if source_index is None:
        raise ValueError(
            f"Cannot disambiguate duplicate origin_problem_id {origin_id} "
            "without source_index"
        )

    generation_id = int(record["generation_id"])
    disambiguated_origin = f"{origin_id}__src{source_index}"
    updated = dict(record)
    updated["origin_problem_id"] = disambiguated_origin
    updated["problem_id"] = f"{disambiguated_origin}_g{generation_id}"
    return updated


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    input_paths = sorted(input_root.glob("shard_*/inference.jsonl"))
    if not input_paths:
        raise FileNotFoundError(f"No shard inference files found under {input_root}")

    raw_records: list[dict[str, Any]] = []
    origin_sources: dict[str, set[int | None]] = defaultdict(set)
    for input_path in input_paths:
        with input_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                record["_merge_input_path"] = str(input_path)
                record["_merge_line_number"] = line_number
                raw_records.append(record)
                origin_sources[record["origin_problem_id"]].add(record.get("source_index"))

    duplicate_origins = {
        origin_id
        for origin_id, source_indices in origin_sources.items()
        if len(source_indices) > 1
    }

    records: dict[str, dict[str, Any]] = {}
    grouped: dict[str, set[int]] = defaultdict(set)
    for raw_record in raw_records:
        record = normalized_record(raw_record, duplicate_origins)
        problem_id = record["problem_id"]
        if problem_id in records:
            raise ValueError(
                f"Duplicate problem_id {problem_id} in "
                f"{raw_record['_merge_input_path']}:{raw_record['_merge_line_number']}"
            )
        record.pop("_merge_input_path", None)
        record.pop("_merge_line_number", None)
        records[problem_id] = record
        grouped[record["origin_problem_id"]].add(int(record["generation_id"]))

    if len(grouped) != args.expected_problems:
        raise RuntimeError(
            f"Expected {args.expected_problems} problems, found {len(grouped)}"
        )
    expected_generation_ids = set(range(args.expected_samples))
    for origin_id, generation_ids in grouped.items():
        if generation_ids != expected_generation_ids:
            raise RuntimeError(
                f"{origin_id} has generations {sorted(generation_ids)}, "
                f"expected {sorted(expected_generation_ids)}"
            )

    ordered = sorted(
        records.values(),
        key=lambda record: (
            int(record.get("source_index", 10**9)),
            record["origin_problem_id"],
            int(record["generation_id"]),
        ),
    )
    expected_records = args.expected_problems * args.expected_samples
    if len(ordered) != expected_records:
        raise RuntimeError(f"Expected {expected_records} records, found {len(ordered)}")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        for record in ordered:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary_path.replace(output_path)
    print(
        f"Merged {len(input_paths)} shards, {len(grouped)} problems, "
        f"{len(ordered)} generations into {output_path}"
    )
    if duplicate_origins:
        print(
            "Disambiguated duplicate origin ids by source_index: "
            + ", ".join(sorted(duplicate_origins))
        )


if __name__ == "__main__":
    main()
