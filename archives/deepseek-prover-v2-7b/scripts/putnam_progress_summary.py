#!/usr/bin/env python

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any


BLOCKED_PROOF_TERMS = ("apply?", "exact?", "sorry", "admit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize completed DeepSeek PutnamBench chunks."
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--pass-target", type=int, required=True)
    parser.add_argument("--chunks", type=int, required=True)
    return parser.parse_args()


def is_correct(record: dict[str, Any]) -> bool:
    code = record.get("code", "").lower()
    result = record.get("compilation_result") or {}
    return bool(result.get("complete")) and not any(
        term in code for term in BLOCKED_PROOF_TERMS
    )


def generation_id(record: dict[str, Any]) -> int:
    if "generation_id" in record:
        return int(record["generation_id"])
    problem_id = str(record.get("problem_id") or record.get("name"))
    return int(problem_id.rsplit("_g", 1)[1])


def summarize_verification(path: Path, expected_samples: int) -> dict[str, Any]:
    records = json.loads(path.read_text(encoding="utf-8"))
    attempts_by_origin: dict[str, dict[int, bool]] = defaultdict(dict)
    for record in records:
        origin = str(record["origin_problem_id"])
        sample_id = generation_id(record)
        attempts_by_origin[origin][sample_id] = is_correct(record)

    expected_generations = set(range(expected_samples))
    complete = all(
        set(samples) == expected_generations for samples in attempts_by_origin.values()
    )
    solved = sum(1 for samples in attempts_by_origin.values() if any(samples.values()))
    problem_count = len(attempts_by_origin)
    return {
        "complete": complete,
        "problem_count": problem_count,
        "attempt_count": sum(len(samples) for samples in attempts_by_origin.values()),
        "solved_count": solved,
        "solved_percent": 100.0 * solved / problem_count if problem_count else 0.0,
        "attempts_by_origin": attempts_by_origin,
    }


def main() -> None:
    args = parse_args()
    if args.pass_target < 1:
        raise ValueError("--pass-target must be positive")
    if args.chunks < 1:
        raise ValueError("--chunks must be positive")

    args.run_root.mkdir(parents=True, exist_ok=True)
    width = max(2, len(str(args.chunks - 1)))
    pass_name = f"pass@{args.pass_target}"
    chunk_rows = []
    overall: dict[str, dict[int, bool]] = defaultdict(dict)

    for chunk_index in range(args.chunks):
        chunk_name = f"chunk_{chunk_index:0{width}d}"
        cumulative_dir = args.run_root / chunk_name / f"cumulative_{pass_name}"
        verification_path = cumulative_dir / "verification.json"
        row = {
            "chunk": chunk_name,
            "chunk_index": chunk_index,
            "status": "missing",
            "problem_count": 0,
            "attempt_count": 0,
            "solved_count": 0,
            "solved_percent": 0.0,
            "summary_dir": str(cumulative_dir / "summary"),
        }
        if verification_path.is_file():
            summary = summarize_verification(verification_path, args.pass_target)
            row.update(
                {
                    "status": "complete" if summary["complete"] else "partial",
                    "problem_count": summary["problem_count"],
                    "attempt_count": summary["attempt_count"],
                    "solved_count": summary["solved_count"],
                    "solved_percent": summary["solved_percent"],
                }
            )
            for origin, samples in summary["attempts_by_origin"].items():
                overall[origin].update(samples)
        chunk_rows.append(row)

    overall_problem_count = len(overall)
    overall_solved_count = sum(1 for samples in overall.values() if any(samples.values()))
    output = {
        "pass_target": args.pass_target,
        "completed_chunks": sum(1 for row in chunk_rows if row["status"] == "complete"),
        "partial_chunks": sum(1 for row in chunk_rows if row["status"] == "partial"),
        "total_chunks": args.chunks,
        "overall": {
            "problem_count": overall_problem_count,
            "attempt_count": sum(len(samples) for samples in overall.values()),
            "solved_count": overall_solved_count,
            "solved_percent": (
                100.0 * overall_solved_count / overall_problem_count
                if overall_problem_count
                else 0.0
            ),
        },
        "chunks": chunk_rows,
    }

    json_path = args.run_root / f"progress_{pass_name}.json"
    temporary_json = json_path.with_suffix(json_path.suffix + f".{os.getpid()}.tmp")
    temporary_json.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary_json.replace(json_path)

    tsv_path = args.run_root / f"progress_{pass_name}.tsv"
    temporary_tsv = tsv_path.with_suffix(tsv_path.suffix + f".{os.getpid()}.tmp")
    with temporary_tsv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "chunk",
            "status",
            "problem_count",
            "attempt_count",
            "solved_count",
            "solved_percent",
            "summary_dir",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in chunk_rows:
            writer.writerow({key: row[key] for key in fieldnames})
    temporary_tsv.replace(tsv_path)

    print(
        f"{pass_name}: chunks {output['completed_chunks']}/{args.chunks}, "
        f"overall solved {overall_solved_count}/{overall_problem_count} "
        f"({output['overall']['solved_percent']:.2f}%)."
    )


if __name__ == "__main__":
    main()
