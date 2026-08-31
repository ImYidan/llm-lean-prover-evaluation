"""Tests for the shared generation runner and Goedel prompt adapter."""

import json

import pytest

from technical.src.generation.adapters.goedel import GoedelPromptAdapter
from technical.src.generation.candidates import Candidate
from technical.src.generation.generate import build_attempts, build_parser, generate_records


class FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        return "CHAT::" + messages[0]["content"]

    def tokenize(self, text):
        return text.split()


class FakeBackend:
    def generate(self, prompts):
        assert prompts and all(prompt.startswith("CHAT::") for prompt in prompts)
        return [
            [Candidate("```lean4\ntheorem generated : True := by\n  trivial\n```")]
            for _ in prompts
        ]


def test_generation_offset_assigns_stable_ids():
    rows = [{"problem_id": "p", "lean4_code": "theorem p : True := by sorry"}]

    attempts = build_attempts(rows, samples=2, offset=8)

    assert [row["problem_id"] for row in attempts] == ["p_g8", "p_g9"]
    assert attempts[0]["id_maps"] == [
        {"origin_problem_id": "p"},
        {"generation_id": "p_g8"},
    ]


def test_prompt_adapter_preserves_statement_and_uses_chat_template():
    statement = "theorem target : True := by sorry"

    prompt, messages = GoedelPromptAdapter().build_prompt(statement, FakeTokenizer())

    assert prompt.startswith("CHAT::Complete the following Lean 4 code:")
    assert statement in messages[0]["content"]
    assert "detailed proof plan" in messages[0]["content"]


def test_singleton_candidate_writes_historical_json_arrays(tmp_path):
    rows = [{"name": "target", "lean4_code": "theorem target : True := by sorry"}]

    full_records, inference_records = generate_records(
        rows,
        tokenizer=FakeTokenizer(),
        backend=FakeBackend(),
        adapter=GoedelPromptAdapter(),
        output_dir=tmp_path,
        samples=1,
        generation_offset=4,
        chunk_size=1,
    )

    assert full_records == json.loads((tmp_path / "full_records.json").read_text())
    assert inference_records == json.loads(
        (tmp_path / "to_inference_codes.json").read_text()
    )
    record = full_records[0]
    assert record["origin_problem_id"] == "target"
    assert record["problem_id"] == "target_g4"
    assert record["model_input"].startswith("CHAT::")
    assert record["messages_history_for_this_attempt"][0]["role"] == "user"
    assert record["model_output"].startswith("```lean4")
    assert record["full_code"] == "theorem target : True := by\n  trivial"
    assert inference_records[0]["messages_history_list"] == record[
        "messages_history_for_this_attempt"
    ]


def test_goedel_rejects_grouped_candidates(tmp_path):
    backend = FakeBackend()
    backend.generate = lambda prompts: [[Candidate("a"), Candidate("b")]]

    with pytest.raises(RuntimeError, match="one candidate"):
        generate_records(
            [{"name": "target", "lean4_code": "theorem target : True := by sorry"}],
            tokenizer=FakeTokenizer(),
            backend=backend,
            adapter=GoedelPromptAdapter(),
            output_dir=tmp_path,
            samples=1,
        )


def test_attempt_builder_skips_rows_without_lean_code():
    rows = [{"problem_id": "missing"}, {"problem_id": "ok", "lean4_code": "x"}]

    attempts = build_attempts(rows, samples=1, offset=0)

    assert [row["problem_id"] for row in attempts] == ["ok_g0"]


def test_attempt_builder_requires_a_problem_identifier():
    rows = [{"lean4_code": "theorem target : True := by sorry"}]

    try:
        build_attempts(rows, samples=1, offset=0)
    except ValueError as error:
        assert str(error) == "row is missing a problem identifier"
    else:
        raise AssertionError("row without a problem identifier was accepted")


def test_remote_model_code_requires_explicit_opt_in():
    args = build_parser().parse_args(
        ["--input", "input.jsonl", "--model-path", "model", "--output-dir", "out"]
    )

    assert args.trust_remote_code is False
    assert args.revision is None
