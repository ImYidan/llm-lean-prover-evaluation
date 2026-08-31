"""Optional smoke tests against a caller-supplied Lean workspace."""

import json
import os
import shlex
import tomllib
from pathlib import Path

import pytest

from technical.src.verification.repl_scheduler import ReplConfig, ReplSession
from technical.src.verification.verify_standard_repl import DEFAULT_IMPORTS
from technical.src.verification.verify_full_header_file import verify_file_record
from technical.src.verification.verify_full_header_repl import FullHeaderReplSession


ROOT = Path(__file__).parents[2]
FATE_PROFILE = ROOT / "technical/lean/fate-v428"
LEAN_428_VERSION = "v4.28.0"
MATHLIB_428_REV = "8f9d9cff6bd728b17a24e163c9402775d9e6a365"
REPL_428_REV = "527590ce2b9f3b5c4a9a1031e5b8fcfb909b9a4a"


def test_fate_v428_profile_pins_toolchain_lake_declarations_and_manifest():
    """Catch drift between the public FATE profile and the required v4.28 pins."""
    assert (
        (FATE_PROFILE / "lean-toolchain").read_text(encoding="utf-8").strip()
        == LEAN_428_VERSION
    )

    lakefile = tomllib.loads(
        (FATE_PROFILE / "lakefile.toml").read_text(encoding="utf-8")
    )
    requires = {package["name"]: package for package in lakefile["require"]}
    assert requires["mathlib"]["rev"] == MATHLIB_428_REV
    assert requires["repl"]["rev"] == REPL_428_REV

    manifest = json.loads(
        (FATE_PROFILE / "lake-manifest.json").read_text(encoding="utf-8")
    )
    packages = {package["name"].lower(): package for package in manifest["packages"]}
    assert packages["mathlib"]["rev"] == MATHLIB_428_REV
    assert packages["repl"]["rev"] == REPL_428_REV


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


@pytest.mark.integration
def test_fate_v428_repl_profile_accepts_trivial_full_header_proof():
    if os.environ.get("LEAN_FATE_V428_REPL_SMOKE") != "1":
        pytest.skip(
            "set LEAN_FATE_V428_REPL_SMOKE=1 to run the FATE v4.28 REPL smoke"
        )
    if not (FATE_PROFILE / "lake-manifest.json").is_file():
        pytest.skip("FATE v4.28 Lake manifest is not present")

    source = "import Mathlib\n\ntheorem fate_v428_smoke : 1 + 1 = 2 := by norm_num\n"
    session = FullHeaderReplSession(
        ReplConfig(
            workspace=FATE_PROFILE,
            repl_command=("lake", "exe", "repl"),
            imports="",
            proof_timeout=300,
        )
    )
    try:
        repl_result = session.verify(source)
    finally:
        session.close()

    assert repl_result["complete"] is True
