"""Tests for portable Lean verification result parsing and scheduling."""

import json
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

from technical.src.verification.repl_scheduler import (
    ReplConfig,
    ReplSession,
    encode_repl_command,
    parse_repl_response,
    system_failure_result,
    verify_records,
)
from technical.src.verification.verify_standard_repl import apply_header_policy
from technical.src.verification.verify_full_header_file import (
    parse_lean_output,
    select_code,
    verify_file_record,
)
from technical.src.verification.verify_full_header_repl import (
    parse_full_header_response,
    verify_full_header_records,
)


def test_complete_requires_no_errors_or_sorries():
    result = parse_repl_response('{"messages": [], "sorries": []}')

    assert result["pass"] is True
    assert result["complete"] is True


def test_sorry_is_not_complete():
    result = parse_repl_response('{"messages": [], "sorries": [{"pos": 1}]}')

    assert result["pass"] is True
    assert result["complete"] is False


def test_compiler_errors_fail_and_diagnostics_are_partitioned():
    block = json.dumps(
        {
            "messages": [
                {"severity": "error", "data": "unknown identifier"},
                {"severity": "warning", "data": "unused variable"},
                {"severity": "info", "data": "trace"},
            ],
            "sorries": [],
        }
    )

    result = parse_repl_response(block)

    assert result["pass"] is False
    assert result["complete"] is False
    assert [message["data"] for message in result["errors"]] == [
        "unknown identifier"
    ]
    assert len(result["warnings"]) == 1
    assert len(result["infos"]) == 1


def test_sorry_warning_is_not_complete():
    block = json.dumps(
        {
            "messages": [
                {"severity": "warning", "data": "declaration uses 'sorry'"}
            ],
            "sorries": [],
        }
    )

    result = parse_repl_response(block)

    assert result["pass"] is True
    assert result["complete"] is False


def test_unrelated_failed_warning_does_not_block_completion():
    block = json.dumps(
        {
            "messages": [
                {"severity": "warning", "data": "optional optimization failed"}
            ],
            "sorries": [],
        }
    )

    result = parse_repl_response(block)

    assert result["pass"] is True
    assert result["complete"] is True


def test_malformed_json_has_an_explicit_system_error():
    result = parse_repl_response("not json")

    assert result["pass"] is False
    assert result["complete"] is False
    assert result["system_errors"].startswith("JSONDECODE ERROR:")


def test_timeout_and_eof_have_stable_failure_shapes():
    timeout = system_failure_result("TIMEOUT", "proof exceeded 7 seconds")
    eof = system_failure_result("EOF", "process exited")

    assert timeout["pass"] is False
    assert timeout["complete"] is False
    assert timeout["system_errors"] == "TIMEOUT ERROR: proof exceeded 7 seconds"
    assert eof["system_errors"] == "EOF ERROR: process exited"


def test_outbound_json_preserves_unicode_instead_of_surrogate_escapes():
    encoded = encode_repl_command("#check 𝓝", env=0)

    assert "𝓝" in encoded
    assert "\\ud835" not in encoded
    assert json.loads(encoded) == {"cmd": "#check 𝓝", "env": 0}


def test_standard_header_policy_always_strips_imports():
    code = "import Mathlib\nset_option maxHeartbeats 0\nopen Nat\ntheorem t : True := by trivial"

    assert apply_header_policy(code, "strip") == "theorem t : True := by trivial"
    assert apply_header_policy(code, "preserve-options") == (
        "set_option maxHeartbeats 0\nopen Nat\ntheorem t : True := by trivial"
    )


class FakeSession:
    def __init__(self, config):
        self.config = config
        self.closed = False

    def verify(self, code):
        assert self.config.proof_timeout == 7
        return parse_repl_response('{"messages": [], "sorries": []}')

    def close(self):
        self.closed = True


def test_verify_records_preserves_historical_result_shape(tmp_path):
    config = ReplConfig(
        workspace=tmp_path,
        repl_command=("lake", "env", "repl"),
        imports="import Mathlib",
        import_timeout=3,
        proof_timeout=7,
    )
    records = [{"name": "p_g0", "code": "theorem p : True := by trivial"}]

    results = verify_records(records, config, workers=1, session_factory=FakeSession)

    assert len(results) == 1
    assert results[0]["name"] == "p_g0"
    assert results[0]["code"] == records[0]["code"]
    assert results[0]["compilation_result"]["complete"] is True
    assert isinstance(results[0]["verify_time"], float)


def test_repl_session_disables_pty_canonical_mode_before_exec(monkeypatch, tmp_path):
    calls = []

    class FakeChild:
        before = '{"messages": [], "sorries": [], "env": 0}'

        def sendline(self, value):
            calls.append(("sendline", value))

        def expect(self, patterns, timeout):
            calls.append(("expect", patterns, timeout))

        def isalive(self):
            return True

        def close(self, force):
            calls.append(("close", force))

    def spawn(command, arguments, **options):
        calls.append(("spawn", command, arguments, options))
        return FakeChild()

    fake_pexpect = SimpleNamespace(
        spawn=spawn,
        TIMEOUT=type("FakeTimeout", (Exception,), {}),
        EOF=type("FakeEof", (Exception,), {}),
    )
    monkeypatch.setitem(sys.modules, "pexpect", fake_pexpect)
    config = ReplConfig(
        workspace=tmp_path,
        repl_command=("lake", "env", "repl"),
        imports="import Mathlib",
    )

    session = ReplSession(config)
    session.close()

    spawn_call = calls[0]
    assert spawn_call[1] == "/bin/bash"
    assert any("stty -icanon" in argument for argument in spawn_call[2])
    assert spawn_call[2][-3:] == ["lake", "env", "repl"]


def test_full_file_code_selection_uses_explicit_or_auto_fields():
    record = {"full_code": "full", "lean4_code": "statement", "code": "fallback"}

    assert select_code(record, "auto") == "full"
    assert select_code(record, "lean4_code") == "statement"


def test_full_file_code_selection_rejects_missing_field():
    try:
        select_code({"problem_id": "p"}, "full_code")
    except ValueError as error:
        assert str(error) == "record is missing code field: full_code"
    else:
        raise AssertionError("missing full_code was accepted")


def test_full_file_parser_reads_multiline_lean_diagnostic():
    output = "file.lean:3:7: error: unknown identifier 'x'\nadditional context"

    messages = parse_lean_output(output)

    assert messages == [
        {
            "severity": "error",
            "pos": {"line": 3, "column": 7},
            "endPos": None,
            "data": "unknown identifier 'x'\nadditional context",
        }
    ]


def test_full_file_verifier_writes_header_byte_for_byte(tmp_path):
    code = "import Mathlib\nopen Nat\n\ntheorem t : True := by trivial\n"
    observed = {}

    def fake_runner(command, **options):
        observed["command"] = command
        observed["code"] = Path(command[-1]).read_text(encoding="utf-8")
        observed["options"] = options
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    result = verify_file_record(
        {"problem_id": "t", "full_code": code},
        workspace=tmp_path,
        timeout=11,
        tmp_dir=tmp_path,
        runner=fake_runner,
    )

    assert observed["command"][:3] == ["lake", "env", "lean"]
    assert observed["code"] == code
    assert observed["options"]["cwd"] == tmp_path
    assert result["compilation_result"]["complete"] is True


def test_full_file_verifier_reports_timeout(tmp_path):
    def timeout_runner(command, **options):
        raise subprocess.TimeoutExpired(command, options["timeout"])

    result = verify_file_record(
        {"problem_id": "t", "code": "theorem t : True := by trivial"},
        workspace=tmp_path,
        timeout=1,
        tmp_dir=tmp_path,
        runner=timeout_runner,
    )

    assert result["compilation_result"]["pass"] is False
    assert result["compilation_result"]["system_errors"].startswith("TIMEOUT ERROR:")


def test_full_header_repl_parser_ignores_echoed_command_and_reads_multiline_json():
    block = (
        '{"cmd": "theorem t : True := by trivial"}\n\nnoise\n'
        '{\n  "messages": [],\n  "sorries": [],\n  "env": 1\n}\n'
    )

    result = parse_full_header_response(block)

    assert result["pass"] is True
    assert result["complete"] is True


def test_full_header_repl_parser_reports_missing_response():
    result = parse_full_header_response('{"cmd": "echo only"}')

    assert result["pass"] is False
    assert result["complete"] is False
    assert result["system_errors"].startswith("PROTOCOL ERROR:")


def test_full_header_workers_reuse_one_persistent_session(tmp_path):
    counts = {"started": 0, "closed": 0}

    class Session:
        def __init__(self, config):
            counts["started"] += 1

        def verify(self, code):
            return parse_repl_response('{"messages": [], "sorries": []}')

        def close(self):
            counts["closed"] += 1

    config = ReplConfig(
        workspace=tmp_path,
        repl_command=("repl",),
        imports="",
    )
    records = [
        {"problem_id": f"p_g{index}", "full_code": f"theorem p{index} : True := by trivial"}
        for index in range(3)
    ]

    results = verify_full_header_records(
        records, config, workers=1, session_factory=Session
    )

    assert [result["name"] for result in results] == ["p_g0", "p_g1", "p_g2"]
    assert counts == {"started": 1, "closed": 1}


def test_full_header_repl_skips_non_complete_input_without_starting_lean(tmp_path):
    """Catch extraction failures being sent to a newly started Lean process."""
    counts = {"started": 0}

    class SessionMustNotStart:
        def __init__(self, config):
            counts["started"] += 1
            raise AssertionError("Lean session must not start for full_code=None")

    config = ReplConfig(
        workspace=tmp_path,
        repl_command=("repl",),
        imports="",
    )

    results = verify_full_header_records(
        [{"problem_id": "p_g0", "full_code": "None"}],
        config,
        workers=1,
        session_factory=SessionMustNotStart,
    )

    assert counts == {"started": 0}
    assert results[0]["code"] == "None"
    assert results[0]["verify_time"] == 0.0
    assert results[0]["compilation_result"]["complete"] is False
    assert results[0]["compilation_result"]["system_errors"] is None


def test_full_header_repl_restart_stays_lazy_before_non_complete_input(tmp_path):
    """A failed prior proof must not start Lean for a following extraction failure."""
    counts = {"started": 0, "closed": 0}

    class FailingSession:
        def __init__(self, config):
            counts["started"] += 1

        def verify(self, code):
            return system_failure_result("TIMEOUT", "test failure")

        def close(self):
            counts["closed"] += 1

    config = ReplConfig(
        workspace=tmp_path,
        repl_command=("repl",),
        imports="",
    )

    results = verify_full_header_records(
        [
            {"problem_id": "p_g0", "full_code": "theorem p : True := by trivial"},
            {"problem_id": "p_g1", "full_code": "None"},
        ],
        config,
        workers=1,
        session_factory=FailingSession,
    )

    assert counts == {"started": 1, "closed": 1}
    assert results[1]["compilation_result"]["system_errors"] is None
