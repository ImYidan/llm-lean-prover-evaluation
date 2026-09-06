#!/usr/bin/env python

import argparse
import concurrent.futures
import json
import os
import random
import signal
import subprocess
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--seed", type=int, default=30)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def system_failure(
    message: str,
    *,
    stdout: str = "",
    stderr: str = "",
    returncode: int | None = None,
) -> dict[str, Any]:
    return {
        "pass": False,
        "complete": False,
        "sorries": [],
        "tactics": [],
        "errors": [],
        "warnings": [],
        "infos": [],
        "system_errors": message,
        "system_messages": stderr,
        "raw_output": stdout,
        "returncode": returncode,
    }


def as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def terminate_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def verify_lean4_file(
    code: str,
    *,
    workspace: Path,
    lake_path: Path,
    repl_path: Path,
    timeout: int = 300,
) -> dict[str, Any]:
    command = [str(lake_path), "env", str(repl_path)]
    payload = json.dumps({"cmd": code}, ensure_ascii=False) + "\n\n"
    started = time.monotonic()
    process: subprocess.Popen[str] | None = None

    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            cwd=workspace,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(input=payload, timeout=timeout)
        except subprocess.TimeoutExpired as error:
            terminate_process_group(process)
            stdout, stderr = process.communicate()
            partial_stdout = as_text(stdout or error.stdout)
            partial_stderr = as_text(stderr or error.stderr)
            result = system_failure(
                f"TIMEOUT after {timeout} seconds",
                stdout=partial_stdout,
                stderr=partial_stderr,
                returncode=process.returncode,
            )
        else:
            raw_output = stdout.strip()
            if process.returncode != 0:
                result = system_failure(
                    f"Lean REPL exited with return code {process.returncode}",
                    stdout=stdout,
                    stderr=stderr,
                    returncode=process.returncode,
                )
            elif not raw_output:
                result = system_failure(
                    "Lean REPL returned empty stdout",
                    stdout=stdout,
                    stderr=stderr,
                    returncode=process.returncode,
                )
            else:
                try:
                    response = json.loads(raw_output)
                except json.JSONDecodeError as error:
                    result = system_failure(
                        f"JSONDECODE ERROR: {error}",
                        stdout=stdout,
                        stderr=stderr,
                        returncode=process.returncode,
                    )
                else:
                    messages = response.get("messages", [])
                    warnings = [
                        message
                        for message in messages
                        if message.get("severity") == "warning"
                    ]
                    errors = [
                        message
                        for message in messages
                        if message.get("severity") == "error"
                    ]
                    result = {
                        "pass": not errors,
                        "complete": False,
                        "sorries": response.get("sorries", []),
                        "tactics": response.get("tactics", []),
                        "errors": errors,
                        "warnings": warnings,
                        "infos": [
                            message
                            for message in messages
                            if message.get("severity") == "info"
                        ],
                        "system_errors": None,
                        "system_messages": stderr,
                        "raw_output": "",
                        "returncode": process.returncode,
                    }
                    result["complete"] = (
                        result["pass"]
                        and not result["sorries"]
                        and not any(
                            "declaration uses 'sorry'"
                            in str(warning.get("data", ""))
                            or "failed" in str(warning.get("data", ""))
                            for warning in warnings
                        )
                    )
    except Exception as error:
        if process is not None:
            terminate_process_group(process)
        result = system_failure(f"{type(error).__name__}: {error}")

    result["verify_time"] = round(time.monotonic() - started, 2)
    return result


def verify_record(
    record: dict[str, Any],
    *,
    workspace: Path,
    lake_path: Path,
    repl_path: Path,
    timeout: int,
) -> dict[str, Any]:
    compilation_result = verify_lean4_file(
        record["full_code"],
        workspace=workspace,
        lake_path=lake_path,
        repl_path=repl_path,
        timeout=timeout,
    )
    verify_time = compilation_result.pop("verify_time")
    return {
        "name": record["problem_id"],
        "problem_id": record["problem_id"],
        "origin_problem_id": record["origin_problem_id"],
        "generation_id": record["generation_id"],
        "source_index": record.get("source_index"),
        "code": record["full_code"],
        "compilation_result": compilation_result,
        "verify_time": verify_time,
    }


def failed_without_lean(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": record["problem_id"],
        "problem_id": record["problem_id"],
        "origin_problem_id": record["origin_problem_id"],
        "generation_id": record["generation_id"],
        "source_index": record.get("source_index"),
        "code": record.get("full_code", ""),
        "compilation_result": {
            "pass": False,
            "complete": False,
            "sorries": [],
            "tactics": [],
            "errors": [],
            "warnings": [],
            "infos": [],
            "system_errors": record.get("extraction_error") or "Empty generated proof",
        },
        "verify_time": 0,
    }


def main() -> None:
    args = parse_args()
    workspace = Path(os.environ["DEEPSEEK_LEAN_WORKSPACE"]).resolve()
    lake_path = Path(os.environ["DEEPSEEK_LAKE_PATH"]).resolve()
    repl = Path(os.environ["DEEPSEEK_REPL_PATH"]).resolve()
    if not lake_path.is_file():
        raise FileNotFoundError(f"Lake executable is missing: {lake_path}")
    if not repl.is_file():
        raise FileNotFoundError(f"Lean REPL is not built: {repl}")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.timeout < 1:
        raise ValueError("--timeout must be positive")

    records = load_jsonl(Path(args.input_path))
    problem_ids = [record["problem_id"] for record in records]
    if len(problem_ids) != len(set(problem_ids)):
        raise ValueError("Duplicate problem_id values found in inference input")
    direct_failures = [
        failed_without_lean(record)
        for record in records
        if not record.get("full_code")
    ]
    pending = [record for record in records if record.get("full_code")]
    random.Random(args.seed).shuffle(pending)

    normalized = []
    if pending:
        workers = min(args.workers, len(pending))
        completed = 0
        system_errors = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    verify_record,
                    record,
                    workspace=workspace,
                    lake_path=lake_path,
                    repl_path=repl,
                    timeout=args.timeout,
                )
                for record in pending
            ]
            for future in concurrent.futures.as_completed(futures):
                output = future.result()
                normalized.append(output)
                completed += 1
                if output["compilation_result"].get("system_errors"):
                    system_errors += 1
                if completed == len(pending) or completed % 10 == 0:
                    print(
                        f"Progress: {completed}/{len(pending)} proofs processed. "
                        f"System errors: {system_errors}",
                        flush=True,
                    )

    normalized.extend(direct_failures)
    normalized.sort(
        key=lambda item: (
            item["source_index"] if item["source_index"] is not None else 10**9,
            item["origin_problem_id"],
            item["generation_id"],
        )
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(normalized, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    complete = sum(
        bool(item["compilation_result"].get("complete")) for item in normalized
    )
    system_errors = sum(
        bool(item["compilation_result"].get("system_errors")) for item in normalized
    )
    print(
        f"verified={len(normalized)} complete={complete} "
        f"system_errors={system_errors}"
    )


if __name__ == "__main__":
    main()
