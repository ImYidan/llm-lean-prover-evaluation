"""Persistent Lean REPL sessions and portable verification scheduling."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class ReplConfig:
    workspace: Path
    repl_command: tuple[str, ...]
    imports: str
    import_timeout: int = 100
    proof_timeout: int = 300
    memory_limit_gb: int = 0

    def __post_init__(self) -> None:
        if not self.repl_command:
            raise ValueError("REPL command must not be empty")
        if self.import_timeout < 1 or self.proof_timeout < 1:
            raise ValueError("REPL timeouts must be positive")
        if self.memory_limit_gb < 0:
            raise ValueError("REPL memory limit must be non-negative")


def encode_repl_command(command: str, env: int | None = None) -> str:
    """Encode a command as raw UTF-8 JSON accepted by the Lean REPL."""
    payload: dict[str, object] = {"cmd": command}
    if env is not None:
        payload["env"] = env
    return json.dumps(payload, ensure_ascii=False)


def system_failure_result(kind: str, detail: object) -> dict:
    """Return the stable compilation-result shape for transport failures."""
    return {
        "pass": False,
        "complete": False,
        "errors": [],
        "warnings": [],
        "infos": [],
        "sorries": [],
        "tactics": [],
        "system_errors": f"{kind} ERROR: {detail}",
    }


def parse_repl_response(block: str) -> dict:
    """Normalize one REPL JSON response into public compilation semantics."""
    try:
        response = json.loads(block)
    except json.JSONDecodeError as error:
        return system_failure_result("JSONDECODE", error)
    if not isinstance(response, dict):
        return system_failure_result("PROTOCOL", "response must be a JSON object")

    messages = response.get("messages", [])
    if not isinstance(messages, list):
        return system_failure_result("PROTOCOL", "messages must be a list")
    errors = [message for message in messages if message.get("severity") == "error"]
    warnings = [
        message for message in messages if message.get("severity") == "warning"
    ]
    infos = [message for message in messages if message.get("severity") == "info"]
    sorries = response.get("sorries", [])
    warning_blocks_completion = any(
        "declaration uses 'sorry'" in str(warning.get("data", ""))
        for warning in warnings
    )
    passed = not errors
    return {
        "pass": passed,
        "complete": passed and not sorries and not warning_blocks_completion,
        "errors": errors,
        "warnings": warnings,
        "infos": infos,
        "sorries": sorries,
        "tactics": response.get("tactics", []),
        "system_errors": None,
    }


class ReplSession:
    """One persistent Lean REPL process initialized with a shared import env."""

    def __init__(self, config: ReplConfig) -> None:
        import pexpect

        self._pexpect = pexpect
        self.config = config
        launch_arguments = [
            "--noprofile",
            "--norc",
            "-c",
            'stty -icanon; exec "$@"',
            "lean-repl",
            *config.repl_command,
        ]
        self.child = pexpect.spawn(
            "/bin/bash",
            launch_arguments,
            cwd=str(config.workspace),
            encoding="utf-8",
            maxread=1,
            echo=False,
        )
        initialization = self._exchange(
            encode_repl_command(config.imports), config.import_timeout
        )
        parsed = parse_repl_response(initialization)
        if parsed["system_errors"] is not None or not parsed["pass"]:
            self.close()
            raise RuntimeError("Lean REPL import initialization failed")

    def _exchange(self, payload: str, timeout: int) -> str:
        self.child.sendline(payload)
        self.child.sendline("")
        self.child.expect(["\r\n\r\n", "\n\n"], timeout=timeout)
        return self.child.before.strip()

    def verify(self, code: str) -> dict:
        try:
            block = self._exchange(
                encode_repl_command(code, env=0), self.config.proof_timeout
            )
            return parse_repl_response(block)
        except self._pexpect.TIMEOUT as error:
            return system_failure_result("TIMEOUT", error)
        except self._pexpect.EOF as error:
            return system_failure_result("EOF", error)
        except Exception as error:
            return system_failure_result("UNEXPECTED", error)

    def close(self) -> None:
        if self.child.isalive():
            self.child.close(force=True)


def _verify_partition(
    indexed_records: list[tuple[int, dict]],
    config: ReplConfig,
    session_factory: Callable[[ReplConfig], object],
) -> list[tuple[int, dict]]:
    results = []
    session = None
    try:
        session = session_factory(config)
        for index, record in indexed_records:
            started = time.monotonic()
            code = str(record.get("code", ""))
            compilation_result = session.verify(code)
            result = {
                "name": record.get("name", record.get("problem_id")),
                "code": code,
                "compilation_result": compilation_result,
                "verify_time": round(time.monotonic() - started, 2),
            }
            if "problem_id" in record:
                result["problem_id"] = record["problem_id"]
            results.append((index, result))

            if compilation_result.get("system_errors") is not None:
                session.close()
                session = session_factory(config)
    finally:
        if session is not None:
            session.close()
    return results


def verify_records(
    records: list[dict],
    config: ReplConfig,
    workers: int,
    *,
    session_factory: Callable[[ReplConfig], object] = ReplSession,
) -> list[dict]:
    """Verify records with one persistent REPL session per worker."""
    if workers < 1:
        raise ValueError("workers must be positive")
    partitions = [[] for _ in range(min(workers, max(1, len(records))))]
    for indexed_record in enumerate(records):
        partitions[indexed_record[0] % len(partitions)].append(indexed_record)
    with ThreadPoolExecutor(max_workers=len(partitions)) as executor:
        futures = [
            executor.submit(_verify_partition, partition, config, session_factory)
            for partition in partitions
            if partition
        ]
        indexed_results = [item for future in futures for item in future.result()]
    return [result for _, result in sorted(indexed_results)]
