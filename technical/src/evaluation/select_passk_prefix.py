"""Select empirical generation prefixes from larger sampling runs."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path


GENERATION_SUFFIX = re.compile(r"_g(\d+)$")


class GenerationIdError(ValueError):
    def __init__(self, invalid_count: int, total_count: int) -> None:
        self.invalid_count = invalid_count
        self.total_count = total_count
        super().__init__(
            f"{invalid_count} of {total_count} records have no generation suffix"
        )


def select_generation_prefix(records: list[dict], k: int) -> list[dict]:
    """Return records with generation indices ``0 <= g < k``."""
    if k < 1:
        raise ValueError("k must be positive")
    selected = []
    invalid = 0
    for record in records:
        name = record.get("name") or record.get("problem_id") or ""
        match = GENERATION_SUFFIX.search(str(name))
        if match is None:
            invalid += 1
        elif int(match.group(1)) < k:
            selected.append(record)
    if invalid:
        raise GenerationIdError(invalid, len(records))
    return selected


def _atomic_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary_name, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("prefix input must be a JSON array")
    _atomic_write(args.output, select_generation_prefix(records, args.k))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
