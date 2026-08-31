"""Validate caller-supplied Lean workspaces against public profile pins."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


MATHLIB_V49_REVISION = "2f65ba7f1a9144b20c8e7358513548e317d26de1"
MATHLIB_V49_TOOLCHAIN = "leanprover/lean4:v4.9.0-rc1"
FATE_V428_TOOLCHAIN = "v4.28.0"
FATE_V428_MATHLIB_REVISION = "8f9d9cff6bd728b17a24e163c9402775d9e6a365"
FATE_V428_REPL_REVISION = "527590ce2b9f3b5c4a9a1031e5b8fcfb909b9a4a"


def _git_revision(workspace: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _read_toolchain(workspace: Path) -> str:
    path = workspace / "lean-toolchain"
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as error:
        raise ValueError("Lean workspace is missing lean-toolchain") from error


def _validate_repl(workspace: Path, repl_command: tuple[str, ...]) -> None:
    if len(repl_command) != 1:
        raise ValueError("REPL command must name one executable path")
    command_path = Path(repl_command[0])
    if not command_path.is_absolute():
        command_path = workspace / command_path
    resolved_workspace = workspace.resolve()
    resolved_command = command_path.resolve()
    if not resolved_command.is_relative_to(resolved_workspace):
        raise ValueError("REPL executable must be inside the Lean workspace")
    if not resolved_command.is_file() or not os.access(resolved_command, os.X_OK):
        raise ValueError("REPL command must name an executable file")


def _fate_package_revisions(workspace: Path) -> dict[str, str]:
    manifest_path = workspace / "lake-manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError("fate-v428 workspace is missing lake-manifest.json") from error
    except json.JSONDecodeError as error:
        raise ValueError("fate-v428 lake-manifest.json is malformed") from error
    packages = manifest.get("packages") if isinstance(manifest, dict) else None
    if not isinstance(packages, list):
        raise ValueError("fate-v428 lake manifest must contain packages")
    return {
        str(package.get("name", "")).lower(): str(package.get("rev", ""))
        for package in packages
        if isinstance(package, dict)
    }


def validate_lean_profile(
    profile: str, workspace: Path, repl_command: tuple[str, ...]
) -> dict[str, str]:
    """Reject a caller workspace or REPL that does not match ``profile``."""
    workspace = Path(workspace)
    if not workspace.is_dir():
        raise ValueError("Lean workspace must be an existing directory")
    toolchain = _read_toolchain(workspace)
    if profile == "mathlib-v49":
        if toolchain != MATHLIB_V49_TOOLCHAIN:
            raise ValueError(
                f"lean profile mathlib-v49 requires toolchain {MATHLIB_V49_TOOLCHAIN}"
            )
        revision = _git_revision(workspace)
        if revision != MATHLIB_V49_REVISION:
            raise ValueError("mathlib-v49 workspace revision mismatch")
        result = {
            "profile": profile,
            "toolchain": toolchain,
            "mathlib_revision": revision,
        }
    elif profile == "fate-v428":
        if toolchain != FATE_V428_TOOLCHAIN:
            raise ValueError(
                f"lean profile fate-v428 requires toolchain {FATE_V428_TOOLCHAIN}"
            )
        revisions = _fate_package_revisions(workspace)
        if revisions.get("mathlib") != FATE_V428_MATHLIB_REVISION:
            raise ValueError("fate-v428 mathlib revision mismatch")
        if revisions.get("repl") != FATE_V428_REPL_REVISION:
            raise ValueError("fate-v428 REPL revision mismatch")
        result = {
            "profile": profile,
            "toolchain": toolchain,
            "mathlib_revision": revisions["mathlib"],
            "repl_revision": revisions["repl"],
        }
    else:
        raise ValueError(f"unsupported Lean profile: {profile}")
    _validate_repl(workspace, repl_command)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--repl-command", nargs="+", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_lean_profile(args.profile, args.workspace, tuple(args.repl_command))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
