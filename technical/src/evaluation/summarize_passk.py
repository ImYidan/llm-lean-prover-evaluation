"""Summarize historical Lean compilation records at every ID-map level."""

from __future__ import annotations

import argparse
import csv
import io
import json
from collections import Counter
from pathlib import Path

from technical.src.run_manifest import atomic_write_text


def _mapping_values(id_map: list[dict]) -> dict[str, object]:
    values = {}
    for mapping in id_map:
        if not isinstance(mapping, dict) or len(mapping) != 1:
            raise ValueError("each id_maps entry must contain exactly one mapping")
        values.update(mapping)
    return values


def summarize(
    records: list[dict],
    id_maps: dict[str, list[dict]],
    field: str = "complete",
    generation_records: list[dict] | None = None,
) -> dict:
    """Calculate solved counts while enforcing forbidden-tactic policy."""
    if field not in {"complete", "pass"}:
        raise ValueError(f"unsupported correctness field: {field}")
    if not records:
        raise ValueError("cannot summarize an empty record set")

    normalized_records = []
    grouped: dict[str, dict[object, dict[str, int]]] = {}
    for record in records:
        name = record.get("name") or record.get("problem_id")
        if name is None:
            raise ValueError("compilation record is missing a problem identifier")
        if name not in id_maps:
            raise ValueError(f"missing id_maps for generation: {name}")
        code = str(record.get("code", ""))
        compilation = record.get("compilation_result", {})
        correct = (
            bool(compilation.get(field))
            and "apply?" not in code
            and "exact?" not in code
        )
        mapping_values = _mapping_values(id_maps[name])
        normalized_records.append(
            {"name": name, "correct": correct, "id_maps": mapping_values}
        )
        for level, value in mapping_values.items():
            bucket = grouped.setdefault(level, {}).setdefault(
                value, {"correct": 0, "count": 0}
            )
            bucket["correct"] += int(correct)
            bucket["count"] += 1

    levels = {}
    for level, details in grouped.items():
        problem_num = len(details)
        solved_num = sum(detail["correct"] > 0 for detail in details.values())
        levels[level] = {
            "problem_num": problem_num,
            "solved_num": solved_num,
            "solved_ratio": f"{solved_num / problem_num * 100:.2f}",
            "details": details,
        }
    result = {"field": field, "levels": levels, "records": normalized_records}
    if generation_records is not None:
        result["generation_outcomes"] = {
            "total": len(generation_records),
            "finish_reason": dict(
                sorted(
                    Counter(
                        "null"
                        if record.get("finish_reason") is None
                        else str(record.get("finish_reason"))
                        for record in generation_records
                    ).items()
                )
            ),
            "extraction_status": dict(
                sorted(
                    Counter(
                        "null"
                        if record.get("extraction_status") is None
                        else str(record.get("extraction_status"))
                        for record in generation_records
                    ).items()
                )
            ),
        }
    return result


def write_summary_outputs(output_dir: Path, result: dict) -> None:
    """Atomically write the historical summary files and outcome metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = []
    for level, summary in result["levels"].items():
        handle = io.StringIO(newline="")
        writer = csv.writer(handle, delimiter="\t", quoting=csv.QUOTE_ALL)
        writer.writerow([level, "sum", "count"])
        for value, detail in summary["details"].items():
            writer.writerow([value, detail["correct"], detail["count"]])
        atomic_write_text(output_dir / f"{level}_summarize.csv", handle.getvalue())
        meta.append(
            {
                "level": level,
                "value": {
                    key: summary[key]
                    for key in ("problem_num", "solved_num", "solved_ratio")
                },
            }
        )
    if "generation_outcomes" in result:
        meta.append({"generation_outcomes": result["generation_outcomes"]})
    atomic_write_text(
        output_dir / "meta_summarize.json",
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--full-records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--field", choices=("complete", "pass"), default="complete")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = json.loads(args.input.read_text(encoding="utf-8"))
    full_records = json.loads(args.full_records.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not isinstance(full_records, list):
        raise ValueError("summary inputs must be JSON arrays")
    maps = {record["problem_id"]: record["id_maps"] for record in full_records}
    write_summary_outputs(
        args.output_dir,
        summarize(
            records,
            maps,
            args.field,
            generation_records=full_records,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
