"""Durable append-only JSONL checkpoints for grouped model generation."""

from __future__ import annotations

import json
import os
from pathlib import Path


class CheckpointError(ValueError):
    """Raised when a checkpoint cannot safely be resumed."""


def _has_line_ending(line: bytes) -> bool:
    return line.endswith((b"\n", b"\r"))


def _checkpoint_lines(path: Path) -> list[bytes]:
    try:
        return path.read_bytes().splitlines(keepends=True)
    except FileNotFoundError:
        return []


def load_checkpoint(path: Path) -> list[dict]:
    """Load complete UTF-8 JSONL records, discarding one torn final line."""
    lines = _checkpoint_lines(Path(path))
    records = []
    for line_number, line in enumerate(lines, start=1):
        final_unterminated_line = (
            line_number == len(lines) and not _has_line_ending(line)
        )
        if not line.strip():
            if final_unterminated_line:
                break
            raise CheckpointError(
                f"malformed checkpoint JSON at line {line_number}: blank line"
            )
        try:
            text = line.decode("utf-8")
            record = json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            if final_unterminated_line:
                break
            raise CheckpointError(
                f"malformed checkpoint JSON at line {line_number}: {error}"
            ) from error
        if not isinstance(record, dict):
            raise CheckpointError(
                f"checkpoint record at line {line_number} must be an object"
            )
        records.append(record)
    index_checkpoint(records)
    return records


def index_checkpoint(records: list[dict]) -> dict[str, dict]:
    """Return records keyed by their unique string generation identifier."""
    indexed = {}
    for record_number, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise CheckpointError(
                f"checkpoint record {record_number} must be an object"
            )
        problem_id = record.get("problem_id")
        if not isinstance(problem_id, str):
            raise CheckpointError(
                f"checkpoint record {record_number} has no string problem_id"
            )
        if problem_id in indexed:
            if indexed[problem_id] == record:
                raise CheckpointError(f"duplicate problem_id: {problem_id}")
            raise CheckpointError(f"conflicting problem_id: {problem_id}")
        indexed[problem_id] = record
    return indexed


def _recover_final_line_before_append(path: Path) -> bool:
    """Remove a recoverable torn tail and report whether a separator is needed."""
    data = path.read_bytes()
    if not data or _has_line_ending(data):
        return False
    final_line_start = max(data.rfind(b"\n"), data.rfind(b"\r")) + 1
    final_line = data[final_line_start:]
    try:
        json.loads(final_line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        with path.open("r+b") as handle:
            handle.truncate(final_line_start)
        return False
    return True


def truncate_checkpoint(path: Path, keep_records: int) -> None:
    """Durably truncate a checkpoint after ``keep_records`` complete rows."""
    path = Path(path)
    if keep_records < 0:
        raise ValueError("keep_records must be non-negative")
    lines = _checkpoint_lines(path)
    if keep_records > len(lines):
        raise ValueError("cannot retain more checkpoint records than exist")
    boundary = sum(len(line) for line in lines[:keep_records])
    with path.open("r+b") as handle:
        handle.truncate(boundary)
        handle.flush()
        os.fsync(handle.fileno())


class CheckpointWriter:
    """Stateful durable writer that validates the existing checkpoint once."""

    def __init__(self, path: Path, records: list[dict] | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        loaded = load_checkpoint(self.path) if records is None else records
        self._indexed = index_checkpoint(loaded)
        self._needs_separator = (
            _recover_final_line_before_append(self.path)
            if self.path.exists()
            else False
        )

    def append(self, record: dict) -> None:
        """Validate and fsync one new unique checkpoint record."""
        candidate = index_checkpoint([record])
        problem_id = next(iter(candidate))
        if problem_id in self._indexed:
            if self._indexed[problem_id] == record:
                raise CheckpointError(f"duplicate problem_id: {problem_id}")
            raise CheckpointError(f"conflicting problem_id: {problem_id}")
        serialized = json.dumps(
            record, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        )
        with self.path.open("a", encoding="utf-8") as handle:
            if self._needs_separator:
                handle.write("\n")
            handle.write(serialized)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._needs_separator = False
        self._indexed[problem_id] = record


def append_checkpoint(path: Path, record: dict) -> None:
    """Durably append one unique JSONL checkpoint record."""
    CheckpointWriter(path).append(record)
