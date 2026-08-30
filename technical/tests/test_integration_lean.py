"""Optional smoke tests against a caller-supplied Lean workspace."""

import os
import shlex
from pathlib import Path

import pytest

from technical.src.verification.repl_scheduler import ReplConfig, ReplSession
from technical.src.verification.verify_standard_repl import DEFAULT_IMPORTS
from technical.src.verification.verify_full_header_file import verify_file_record
from technical.src.verification.verify_full_header_repl import FullHeaderReplSession


@pytest.mark.integration
def test_basic_theorem_in_configured_lean_workspace():
    workspace = os.environ.get("LEAN_TEST_WORKSPACE")
    repl_command = os.environ.get("LEAN_TEST_REPL_COMMAND")
    if not workspace or not repl_command:
        pytest.skip(
            "set LEAN_TEST_WORKSPACE and LEAN_TEST_REPL_COMMAND to run Lean smoke"
        )

    session = ReplSession(
        ReplConfig(
            workspace=Path(workspace),
            repl_command=tuple(shlex.split(repl_command)),
            imports=DEFAULT_IMPORTS,
        )
    )
    try:
        result = session.verify("theorem test : 1 + 1 = 2 := by norm_num")
    finally:
        session.close()

    assert result["complete"] is True


@pytest.mark.integration
def test_full_header_verifiers_in_configured_lean_workspace():
    workspace = os.environ.get("LEAN_TEST_WORKSPACE")
    repl_command = os.environ.get("LEAN_TEST_REPL_COMMAND")
    if not workspace or not repl_command:
        pytest.skip(
            "set LEAN_TEST_WORKSPACE and LEAN_TEST_REPL_COMMAND to run Lean smoke"
        )

    source = "import Mathlib\n\ntheorem test_full : 1 + 1 = 2 := by norm_num\n"
    file_result = verify_file_record(
        {"problem_id": "test_full", "full_code": source},
        workspace=Path(workspace),
        timeout=300,
    )
    session = FullHeaderReplSession(
        ReplConfig(
            workspace=Path(workspace),
            repl_command=tuple(shlex.split(repl_command)),
            imports="",
        )
    )
    try:
        repl_result = session.verify(source)
    finally:
        session.close()

    assert file_result["compilation_result"]["complete"] is True
    assert repl_result["complete"] is True
