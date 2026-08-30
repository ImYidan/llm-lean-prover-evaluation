"""Verify complete Lean sources by compiling temporary files."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable


MESSAGE_RE = re.compile(
    r"^(?P<file>.*?):(?P<line>\d+):(?P<column>\d+): "
    r"(?P<severity>error|warning|info): (?P<data>.*)$"
)
CODE_FIELDS = ("full_code", "lean4_code", "code")


def select_code(record: dict, code_field: str = "auto") -> str:
    """Select a complete Lean source from a historical generation record."""
    if code_field != "auto":
        if code_field not in CODE_FIELDS:
            raise ValueError(f"unknown code field: {code_field}")
        value = record.get(code_field)
        if value is None:
            raise ValueError(f"record is missing code field: {code_field}")
        return str(value)
    for field in CODE_FIELDS:
        value = record.get(field)
        if value not in (None, ""):
            return str(value)
    raise ValueError("record is missing a Lean code field")


def parse_lean_output(text: str) -> list[dict]:
    """Parse Lean's location-prefixed textual diagnostics."""
    messages = []
    current = None
    for line in text.splitlines():
        match = MESSAGE_RE.match(line)
        if match:
            if current is not None:
                messages.append(current)
            current = {
                "severity": match.group("severity"),
                "pos": {
                    "line": int(match.group("line")),
                    "column": int(match.group("column")),
                },
                "endPos": None,
                "data": match.group("data"),
            }
        elif current is not None:
            current["data"] += "\n" + line
    if current is not None:
        messages.append(current)
    return messages


def _empty_result(system_errors: str | None) -> dict:
    return {
        "sorries": [],
        "tactics": [],
        "errors": [],
        "warnings": [],
        "infos": [],
        "ast": {},
        "system_errors": system_errors,
        "pass": False,
        "complete": False,
        "stdout": "",
        "stderr": "",
        "exit_code": None,
        "timed_out": system_errors is not None
        and system_errors.startswith("TIMEOUT ERROR:"),
    }


def verify_file_record(
    record: dict,
    *,
    workspace: Path,
    timeout: int,
    tmp_dir: Path | None = None,
    code_field: str = "auto",
    runner: Callable = subprocess.run,
) -> dict:
    """Compile one complete source without altering its bytes."""
    started = time.monotonic()
    name = record.get("problem_id") or record.get("name")
    if name is None:
        raise ValueError("record is missing a problem identifier")
    code = select_code(record, code_field)
    if not code.strip() or code == "None":
        compilation_result = _empty_result(None)
        return {
            "name": name,
            "problem_id": name,
            "code": code,
            "compilation_result": compilation_result,
            "verify_time": 0.0,
        }

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".lean",
            dir=tmp_dir,
            encoding="utf-8",
            newline="",
            delete=False,
        ) as temporary:
            temporary.write(code)
            temporary_path = Path(temporary.name)

        completed = runner(
            ["lake", "env", "lean", str(temporary_path)],
            cwd=Path(workspace),
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        output = "\n".join(part for part in (stdout, stderr) if part)
        messages = parse_lean_output(output)
        errors = [message for message in messages if message["severity"] == "error"]
        warnings = [
            message for message in messages if message["severity"] == "warning"
        ]
        infos = [message for message in messages if message["severity"] == "info"]
        sorries = [
            warning
            for warning in warnings
            if "declaration uses 'sorry'" in warning.get("data", "")
        ]
        passed = completed.returncode == 0 and not errors
        compilation_result = {
            "sorries": sorries,
            "tactics": [],
            "errors": errors,
            "warnings": warnings,
            "infos": infos,
            "ast": {},
            "system_errors": None,
            "pass": passed,
            "complete": passed and not sorries,
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": completed.returncode,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as error:
        compilation_result = _empty_result(f"TIMEOUT ERROR: {error}")
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass

    return {
        "name": name,
        "problem_id": name,
        "code": code,
        "compilation_result": compilation_result,
        "verify_time": round(time.monotonic() - started, 2),
    }


def load_records(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError("verification input must contain record objects")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--tmp-dir", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--code-field", choices=("auto", *CODE_FIELDS), default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = load_records(args.input)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(
            executor.map(
                lambda record: verify_file_record(
                    record,
                    workspace=args.workspace,
                    timeout=args.timeout,
                    tmp_dir=args.tmp_dir,
                    code_field=args.code_field,
                ),
                records,
            )
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{args.output.name}.", suffix=".tmp", dir=args.output.parent
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary_name, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
