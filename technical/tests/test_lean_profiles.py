"""Tests for caller-supplied Lean workspace/profile validation."""

import json

import pytest

from technical.src import lean_profiles


def _executable(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def test_mathlib_v49_profile_keeps_goedel_rc1_contract(tmp_path, monkeypatch):
    """Catch DeepSeek's rc2 profile replacing the preserved Goedel rc1 branch."""
    workspace = tmp_path / "mathlib4"
    workspace.mkdir()
    (workspace / "lean-toolchain").write_text(
        "leanprover/lean4:v4.9.0-rc1\n", encoding="utf-8"
    )
    repl = _executable(workspace / ".lake/packages/REPL/.lake/build/bin/repl")
    monkeypatch.setattr(
        lean_profiles,
        "_git_revision",
        lambda path: "2f65ba7f1a9144b20c8e7358513548e317d26de1",
    )

    result = lean_profiles.validate_lean_profile(
        "mathlib-v49", workspace, (str(repl),)
    )

    assert result["toolchain"] == "leanprover/lean4:v4.9.0-rc1"


def test_deepseek_v49_rc2_profile_validates_toolchain_revision_and_repl(
    tmp_path, monkeypatch
):
    """Catch DeepSeek core verification drifting from its recorded rc2 toolchain."""
    workspace = tmp_path / "mathlib4"
    workspace.mkdir()
    (workspace / "lean-toolchain").write_text(
        "leanprover/lean4:v4.9.0-rc2\n", encoding="utf-8"
    )
    (workspace / "lake-manifest.json").write_text(
        json.dumps(
            {
                "packages": [
                    {
                        "name": "REPL",
                        "rev": "3334a97b268ecc67beb36a75787f7e831208a724",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    repl = _executable(workspace / ".lake/packages/REPL/.lake/build/bin/repl")
    monkeypatch.setattr(
        lean_profiles,
        "_git_revision",
        lambda path: "2f65ba7f1a9144b20c8e7358513548e317d26de1",
    )

    result = lean_profiles.validate_lean_profile(
        "deepseek-v49-rc2", workspace, (str(repl),)
    )

    assert result["profile"] == "deepseek-v49-rc2"
    assert result["toolchain"] == "leanprover/lean4:v4.9.0-rc2"
    assert result["repl_revision"] == "3334a97b268ecc67beb36a75787f7e831208a724"


def test_fate_profile_rejects_v49_workspace_before_verification(tmp_path):
    """Catch FATE verification running under v4.9 while labeled fate-v428."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "lean-toolchain").write_text(
        "leanprover/lean4:v4.9.0-rc1\n", encoding="utf-8"
    )
    repl = _executable(workspace / "repl")

    with pytest.raises(
        ValueError,
        match="lean profile fate-v428 requires toolchain v4.28.0",
    ):
        lean_profiles.validate_lean_profile(
            "fate-v428", workspace, (str(repl),)
        )


def test_fate_profile_validates_mathlib_and_repl_manifest_pins(tmp_path):
    """Catch a nominal v4.28 workspace with drifted Lake dependencies."""
    workspace = tmp_path / "fate"
    workspace.mkdir()
    (workspace / "lean-toolchain").write_text("v4.28.0\n", encoding="utf-8")
    (workspace / "lake-manifest.json").write_text(
        json.dumps(
            {
                "packages": [
                    {
                        "name": "mathlib",
                        "rev": "wrong",
                    },
                    {
                        "name": "repl",
                        "rev": "527590ce2b9f3b5c4a9a1031e5b8fcfb909b9a4a",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    repl = _executable(workspace / ".lake/packages/repl/.lake/build/bin/repl")

    with pytest.raises(ValueError, match="fate-v428 mathlib revision mismatch"):
        lean_profiles.validate_lean_profile(
            "fate-v428", workspace, (str(repl),)
        )


def test_profile_rejects_repl_executable_outside_workspace(tmp_path, monkeypatch):
    """Catch a pinned workspace paired with a REPL built elsewhere."""
    workspace = tmp_path / "mathlib4"
    workspace.mkdir()
    (workspace / "lean-toolchain").write_text(
        "leanprover/lean4:v4.9.0-rc2\n", encoding="utf-8"
    )
    (workspace / "lake-manifest.json").write_text(
        json.dumps(
            {
                "packages": [
                    {
                        "name": "REPL",
                        "rev": "3334a97b268ecc67beb36a75787f7e831208a724",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    repl = _executable(tmp_path / "other/repl")
    monkeypatch.setattr(
        lean_profiles,
        "_git_revision",
        lambda path: "2f65ba7f1a9144b20c8e7358513548e317d26de1",
    )

    with pytest.raises(ValueError, match="REPL executable must be inside"):
        lean_profiles.validate_lean_profile(
            "deepseek-v49-rc2", workspace, (str(repl),)
        )
