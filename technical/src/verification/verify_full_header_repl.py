"""Verify complete Lean sources with robust full-header REPL framing."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from technical.src.verification.repl_scheduler import (
    ReplConfig,
    encode_repl_command,
    parse_repl_response,
    system_failure_result,
)
from technical.src.verification.verify_full_header_file import load_records, select_code


REPL_RESPONSE_KEYS = {"messages", "sorries", "tactics", "env"}


def iter_json_objects(text: str):
    """Yield complete JSON objects embedded in echoed or multiline output."""
    decoder = json.JSONDecoder()
    position = 0
    while True:
        start = text.find("{", position)
        if start < 0:
            return
        try:
            value, length = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            position = start + 1
            continue
        yield value
        position = start + length


def _is_response(value: object) -> bool:
    if not isinstance(value, dict) or "cmd" in value:
        return False
    return bool(REPL_RESPONSE_KEYS.intersection(value) or "error" in value or "message" in value)


def parse_full_header_response(block: str) -> dict:
    """Select the last actual REPL response, ignoring echoed JSON commands."""
    candidates = [value for value in iter_json_objects(block) if _is_response(value)]
    if not candidates:
        return system_failure_result(
            "PROTOCOL", "no Lean REPL response JSON object was found"
        )
    response = dict(candidates[-1])
    if not response.get("messages"):
        if "error" in response:
            response["messages"] = [
                {"severity": "error", "data": str(response["error"])}
            ]
        elif "message" in response:
            response["messages"] = [
                {"severity": "error", "data": str(response["message"])}
            ]
    return parse_repl_response(json.dumps(response, ensure_ascii=False))


class FullHeaderReplSession:
    """Persistent REPL process where every command carries its complete header."""

    def __init__(self, config: ReplConfig) -> None:
        import pexpect

        self._pexpect = pexpect
        self.config = config
        arguments = [
            "--noprofile",
            "--norc",
            "-c",
            'stty -echo -icanon; exec "$@"',
            "lean-repl",
            *config.repl_command,
        ]
        self.child = pexpect.spawn(
            "/bin/bash",
            arguments,
            cwd=str(config.workspace),
            encoding="utf-8",
            maxread=1,
            echo=False,
        )

    def verify(self, code: str) -> dict:
        if not code:
            return system_failure_result("INPUT", "Lean source is empty")
        self.child.sendline(encode_repl_command(code))
        self.child.sendline("")
        chunks = []
        deadline = time.monotonic() + self.config.proof_timeout
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return system_failure_result(
                        "TIMEOUT", "no valid Lean REPL response was read"
                    )
                self.child.expect(["\r\n\r\n", "\n\n"], timeout=remaining)
                block = self.child.before.strip()
                if block:
                    chunks.append(block)
                parsed = parse_full_header_response("\n".join(chunks))
                if not str(parsed.get("system_errors", "")).startswith("PROTOCOL ERROR:"):
                    return parsed
        except self._pexpect.TIMEOUT as error:
            return system_failure_result("TIMEOUT", error)
        except self._pexpect.EOF as error:
            return system_failure_result("EOF", error)
        except Exception as error:
            return system_failure_result("UNEXPECTED", error)

    def close(self) -> None:
        if self.child.isalive():
            self.child.close(force=True)


def _verify_partition(indexed_records, config, code_field, session_factory):
    results = []
    session = session_factory(config)
    try:
        for index, record in indexed_records:
            name = record.get("problem_id") or record.get("name")
            if name is None:
                raise ValueError("record is missing a problem identifier")
            code = select_code(record, code_field)
            started = time.monotonic()
            compilation_result = session.verify(code)
            results.append(
                (
                    index,
                    {
                        "name": name,
                        "problem_id": name,
                        "code": code,
                        "compilation_result": compilation_result,
                        "verify_time": round(time.monotonic() - started, 2),
                    },
                )
            )
            if compilation_result.get("system_errors") is not None:
                session.close()
                session = session_factory(config)
    finally:
        session.close()
    return results


def verify_full_header_records(
    records: list[dict],
    config: ReplConfig,
    workers: int,
    *,
    code_field: str = "auto",
    session_factory=FullHeaderReplSession,
) -> list[dict]:
    if workers < 1:
        raise ValueError("workers must be positive")
    partitions = [[] for _ in range(min(workers, max(1, len(records))))]
    for indexed_record in enumerate(records):
        partitions[indexed_record[0] % len(partitions)].append(indexed_record)
    with ThreadPoolExecutor(max_workers=len(partitions)) as executor:
        futures = [
            executor.submit(
                _verify_partition,
                partition,
                config,
                code_field,
                session_factory,
            )
            for partition in partitions
            if partition
        ]
        indexed_results = [item for future in futures for item in future.result()]
    return [result for _, result in sorted(indexed_results)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--repl-command", nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--code-field", choices=("auto", "full_code", "lean4_code", "code"), default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = load_records(args.input)
    config = ReplConfig(
        workspace=args.workspace,
        repl_command=tuple(args.repl_command),
        imports="",
        proof_timeout=args.timeout,
    )
    results = verify_full_header_records(
        records, config, args.workers, code_field=args.code_field
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{args.output.name}.", suffix=".tmp", dir=args.output.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(results, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_name, args.output)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
