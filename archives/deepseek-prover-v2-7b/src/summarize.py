#!/usr/bin/env python

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verification-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ks", default="1,8,16,32")
    return parser.parse_args()


def parse_ks(raw: str) -> list[int]:
    values = sorted({int(value.strip()) for value in raw.split(",") if value.strip()})
    if not values or values[0] < 1:
        raise ValueError("--ks must contain positive integers")
    return values


def pass_at_k(total: int, correct: int, k: int) -> float:
    if not 1 <= k <= total:
        raise ValueError(f"k={k} must be in [1, {total}]")
    if correct <= 0:
        return 0.0
    if total - correct < k:
        return 1.0
    return 1.0 - math.comb(total - correct, k) / math.comb(total, k)


def is_correct(record: dict[str, Any]) -> bool:
    code = record.get("code", "").lower()
    blocked = ("apply?", "exact?", "sorry", "admit")
    return bool(record["compilation_result"].get("complete")) and not any(
        term in code for term in blocked
    )


def generation_id(record: dict[str, Any]) -> int:
    if "generation_id" in record:
        return int(record["generation_id"])
    match = re.search(r"_g(\d+)$", record["problem_id"])
    if match is None:
        raise ValueError(f"Cannot infer generation_id for {record['problem_id']}")
    return int(match.group(1))


def main() -> None:
    args = parse_args()
    records = json.loads(Path(args.verification_path).read_text(encoding="utf-8"))
    if not records:
        raise ValueError("Verification file is empty")

    grouped: dict[str, dict[int, bool]] = defaultdict(dict)
    source_indices: dict[str, int | None] = {}
    for record in records:
        origin_id = record["origin_problem_id"]
        sample_id = generation_id(record)
        if sample_id in grouped[origin_id]:
            raise ValueError(f"Duplicate generation {sample_id} for {origin_id}")
        grouped[origin_id][sample_id] = is_correct(record)
        source_indices[origin_id] = record.get("source_index")

    sample_counts = {len(values) for values in grouped.values()}
    if len(sample_counts) != 1:
        raise RuntimeError(f"Inconsistent samples per problem: {sorted(sample_counts)}")
    samples = sample_counts.pop()
    expected_ids = set(range(samples))
    for origin_id, values in grouped.items():
        if set(values) != expected_ids:
            raise RuntimeError(
                f"{origin_id} has generation ids {sorted(values)}, "
                f"expected {sorted(expected_ids)}"
            )

    requested_ks = parse_ks(args.ks)
    ks = [k for k in requested_ks if k <= samples]
    if not ks:
        raise ValueError(
            f"None of the requested k values {requested_ks} fit {samples} samples"
        )

    problems = len(grouped)
    summary: dict[str, Any] = {
        "problem_count": problems,
        "samples_per_problem": samples,
        "requested_k_values": requested_ks,
        "reported_k_values": ks,
        "metrics": {},
    }
    for k in ks:
        empirical_solved = sum(
            any(values[sample_id] for sample_id in range(k))
            for values in grouped.values()
        )
        estimated_percent = (
            100.0
            * sum(pass_at_k(samples, sum(values.values()), k) for values in grouped.values())
            / problems
        )
        summary["metrics"][str(k)] = {
            "empirical_solved": empirical_solved,
            "empirical_percent": 100.0 * empirical_solved / problems,
            "unbiased_estimate_percent": estimated_percent,
        }
        summary[f"pass_at_{k}_solved"] = empirical_solved
        summary[f"pass_at_{k}_percent"] = 100.0 * empirical_solved / problems
        summary[f"estimated_pass_at_{k}_percent"] = estimated_percent

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "per_problem.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            (
                "problem_id",
                "source_index",
                "correct_samples",
                "samples",
                "first_success_generation",
                *(f"pass_at_{k}" for k in ks),
            )
        )
        for problem_id in sorted(
            grouped,
            key=lambda item: (
                source_indices[item] if source_indices[item] is not None else 10**9,
                item,
            ),
        ):
            values = grouped[problem_id]
            successful = [sample_id for sample_id, correct in values.items() if correct]
            writer.writerow(
                (
                    problem_id,
                    source_indices[problem_id],
                    len(successful),
                    samples,
                    min(successful) if successful else "",
                    *(int(any(values[sample_id] for sample_id in range(k))) for k in ks),
                )
            )

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
