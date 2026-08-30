"""Verify standard assembled submissions with the persistent Lean REPL."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from technical.src.verification.repl_scheduler import ReplConfig, verify_records


DEFAULT_IMPORTS = (
    "import Mathlib\nimport Aesop\n\n"
    "set_option maxHeartbeats 0\n\n"
    "open BigOperators Real Nat Topology Rat\n"
)


def apply_header_policy(code: str, policy: str) -> str:
    """Remove REPL-invalid imports and optionally preserve option/open lines."""
    if policy not in {"strip", "preserve-options"}:
        raise ValueError(f"unknown header policy: {policy}")
    filtered = []
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("import"):
            continue
        if policy == "strip" and (
            stripped.startswith("set_option") or stripped.startswith("open")
        ):
            continue
        filtered.append(line)
    return "\n".join(filtered)


def prepare_records(records: list[dict], policy: str) -> list[dict]:
    prepared = []
    for record in records:
        problem_id = record.get("problem_id") or record.get("name")
        if problem_id is None:
            raise ValueError("verification record is missing a problem identifier")
        code = record.get("full_code", record.get("code", ""))
        if code in (None, "None"):
            code = ""
        prepared.append(
            {
                "name": problem_id,
                "problem_id": problem_id,
                "code": apply_header_policy(str(code), policy),
            }
        )
    return prepared


def _atomic_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--repl-command", nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--header-policy", choices=("strip", "preserve-options"))
    parser.add_argument("--import-timeout", type=int, default=100)
    parser.add_argument("--proof-timeout", type=int, default=300)
    parser.add_argument("--imports", default=DEFAULT_IMPORTS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    policy = args.header_policy
    if policy is None:
        policy = "preserve-options" if os.environ.get("KEEP_HEADER_LINES") == "1" else "strip"
    records = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("verification input must be a JSON array")
    config = ReplConfig(
        workspace=args.workspace,
        repl_command=tuple(args.repl_command),
        imports=args.imports,
        import_timeout=args.import_timeout,
        proof_timeout=args.proof_timeout,
    )
    results = verify_records(prepare_records(records, policy), config, args.workers)
    _atomic_write(args.output, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
