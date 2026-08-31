"""Tests for resumable grouped DeepSeek generation."""

import json
import sys
import types

import pytest

from technical.src.generation.candidates import Candidate
from technical.src.generation.checkpoints import append_checkpoint, load_checkpoint
from technical.src.generation.deepseek import generate_deepseek_records
from technical.src.generation.generate import _VllmBackend, build_parser, create_vllm_backend


class FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        return "CHAT::" + messages[0]["content"]


class FakeGroupedBackend:
    def __init__(self, candidates_per_prompt):
        self.candidates_per_prompt = candidates_per_prompt
        self.prompts = []

    def generate(self, prompts):
        self.prompts.append(list(prompts))
        return [
            [
                Candidate(
                    "```lean4\ntheorem generated : True := by\n  trivial\n```",
                    finish_reason="stop",
                    stop_reason=128001,
                )
                for _ in range(self.candidates_per_prompt)
            ]
            for _ in prompts
        ]


class WrongGroupCountBackend:
    def generate(self, prompts):
        return []


class WrongCandidateCountBackend:
    def generate(self, prompts):
        return [[Candidate("```lean4\ntheorem p : True := by\n  trivial\n```")]]


def source_rows(count=2):
    return [
        {
            "problem_id": f"p{index}",
            "lean4_code": f"theorem p{index} : True := by sorry",
        }
        for index in range(count)
    ]


def test_grouped_generation_writes_raw_and_normalized_records(tmp_path):
    """Catch grouped candidates losing stable IDs, metadata, or normalization."""
    rows = [{"problem_id": "p", "lean4_code": "theorem p : True := by sorry"}]

    generate_deepseek_records(
        rows,
        tokenizer=FakeTokenizer(),
        backend=FakeGroupedBackend(2),
        output_dir=tmp_path,
        samples=2,
        generation_offset=8,
    )

    raw = load_checkpoint(tmp_path / "inference.jsonl")
    assert [row["problem_id"] for row in raw] == ["p_g8", "p_g9"]
    assert raw[0]["origin_problem_id"] == "p"
    assert raw[0]["finish_reason"] == "stop"
    assert raw[0]["stop_reason"] == 128001
    assert raw[0]["model_input"].startswith("CHAT::")
    assert raw[0]["model_output"].startswith("```lean4")
    assert raw[0]["full_code"] == "theorem p : True := by\n  trivial"
    assert raw[0]["extraction_status"] == "success"
    assert len(json.loads((tmp_path / "full_records.json").read_text())) == 2
    assert len(
        json.loads((tmp_path / "to_inference_codes.json").read_text())
    ) == 2


def test_resume_from_complete_prefix_requests_only_missing_source(tmp_path):
    """Catch resume logic that regens completed grouped source prompts."""
    rows = source_rows(2)
    generate_deepseek_records(
        rows[:1],
        tokenizer=FakeTokenizer(),
        backend=FakeGroupedBackend(2),
        output_dir=tmp_path,
        samples=2,
        generation_offset=4,
    )
    backend = FakeGroupedBackend(2)

    full_records, _ = generate_deepseek_records(
        rows,
        tokenizer=FakeTokenizer(),
        backend=backend,
        output_dir=tmp_path,
        samples=2,
        generation_offset=4,
    )

    assert len(backend.prompts) == 1
    assert len(backend.prompts[0]) == 1
    raw = load_checkpoint(tmp_path / "inference.jsonl")
    assert [record["problem_id"] for record in raw] == [
        "p0_g4",
        "p0_g5",
        "p1_g4",
        "p1_g5",
    ]
    assert [record["problem_id"] for record in full_records] == [
        "p0_g4",
        "p0_g5",
        "p1_g4",
        "p1_g5",
    ]


def test_resume_rejects_checkpoint_gap(tmp_path):
    """Catch checkpoints that skip an expected grouped candidate ID."""
    append_checkpoint(
        tmp_path / "inference.jsonl",
        {"problem_id": "p0_g4", "origin_problem_id": "p0"},
    )
    append_checkpoint(
        tmp_path / "inference.jsonl",
        {"problem_id": "p0_g6", "origin_problem_id": "p0"},
    )

    with pytest.raises(ValueError, match="checkpoint row 2 expected p0_g5"):
        generate_deepseek_records(
            source_rows(1),
            tokenizer=FakeTokenizer(),
            backend=FakeGroupedBackend(2),
            output_dir=tmp_path,
            samples=2,
            generation_offset=4,
        )


def test_resume_rejects_unexpected_offset(tmp_path):
    """Catch checkpoint prefixes from a different generation offset."""
    append_checkpoint(
        tmp_path / "inference.jsonl",
        {"problem_id": "p0_g3", "origin_problem_id": "p0"},
    )

    with pytest.raises(ValueError, match="checkpoint row 1 expected p0_g4"):
        generate_deepseek_records(
            source_rows(1),
            tokenizer=FakeTokenizer(),
            backend=FakeGroupedBackend(2),
            output_dir=tmp_path,
            samples=2,
            generation_offset=4,
        )


def test_resume_rejects_excess_candidates(tmp_path):
    """Catch checkpoints longer than the expected generation sequence."""
    for problem_id in ("p0_g4", "p0_g5", "p0_g6"):
        append_checkpoint(
            tmp_path / "inference.jsonl",
            {"problem_id": problem_id, "origin_problem_id": "p0"},
        )

    with pytest.raises(ValueError, match="checkpoint has 3 rows but expected 2"):
        generate_deepseek_records(
            source_rows(1),
            tokenizer=FakeTokenizer(),
            backend=FakeGroupedBackend(2),
            output_dir=tmp_path,
            samples=2,
            generation_offset=4,
        )


def test_grouped_generation_rejects_backend_group_count_mismatch(tmp_path):
    """Catch dropped grouped backend requests before partial normalization."""
    with pytest.raises(RuntimeError, match="wrong number of request groups"):
        generate_deepseek_records(
            source_rows(1),
            tokenizer=FakeTokenizer(),
            backend=WrongGroupCountBackend(),
            output_dir=tmp_path,
            samples=2,
            generation_offset=0,
        )


def test_grouped_generation_rejects_candidate_count_mismatch(tmp_path):
    """Catch grouped backends returning fewer candidates than requested."""
    with pytest.raises(RuntimeError, match="expected 2 candidates"):
        generate_deepseek_records(
            source_rows(1),
            tokenizer=FakeTokenizer(),
            backend=WrongCandidateCountBackend(),
            output_dir=tmp_path,
            samples=2,
            generation_offset=0,
        )


def test_vllm_backend_preserves_candidate_finish_and_stop_reasons():
    """Catch candidate conversion that drops backend termination metadata."""

    class Output:
        text = "proof"
        finish_reason = "length"
        stop_reason = "max_tokens"

    class Request:
        outputs = [Output()]

    class Model:
        def generate(self, prompts, sampling_params):
            assert prompts == ["prompt"]
            return [Request()]

    candidates = _VllmBackend(Model(), object()).generate(["prompt"])

    assert candidates == [[Candidate("proof", "length", "max_tokens")]]


def test_vllm_sampling_params_receives_samples_per_prompt(monkeypatch):
    """Catch real backend creation ignoring grouped sample count."""
    captured = {}

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return "tokenizer"

    class LLM:
        def __init__(self, **kwargs):
            captured["llm"] = kwargs

    class SamplingParams:
        def __init__(self, **kwargs):
            captured["sampling"] = kwargs

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoTokenizer=AutoTokenizer),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        types.SimpleNamespace(LLM=LLM, SamplingParams=SamplingParams),
    )

    tokenizer, backend = create_vllm_backend(
        model_path="model",
        seed=1,
        tensor_parallel_size=2,
        max_model_len=128,
        temperature=0.7,
        top_p=0.9,
        max_tokens=64,
        samples_per_prompt=3,
    )

    assert tokenizer == "tokenizer"
    assert isinstance(backend, _VllmBackend)
    assert captured["sampling"]["n"] == 3


def test_vllm_backend_rejects_max_tokens_at_model_limit_before_import(monkeypatch):
    """Catch validation that happens after importing optional heavy backends."""
    monkeypatch.delitem(sys.modules, "transformers", raising=False)
    monkeypatch.delitem(sys.modules, "vllm", raising=False)

    with pytest.raises(ValueError, match="max_tokens must be less than max_model_len"):
        create_vllm_backend(
            model_path="model",
            seed=1,
            tensor_parallel_size=1,
            max_model_len=128,
            temperature=1.0,
            top_p=0.95,
            max_tokens=128,
            samples_per_prompt=1,
        )


def test_vllm_backend_rejects_non_positive_samples_before_import(monkeypatch):
    """Catch grouped sample validation happening after backend imports."""
    monkeypatch.delitem(sys.modules, "transformers", raising=False)
    monkeypatch.delitem(sys.modules, "vllm", raising=False)

    with pytest.raises(ValueError, match="samples must be positive"):
        create_vllm_backend(
            model_path="model",
            seed=1,
            tensor_parallel_size=1,
            max_model_len=128,
            temperature=1.0,
            top_p=0.95,
            max_tokens=64,
            samples_per_prompt=0,
        )


def test_cli_defaults_to_goedel_adapter():
    """Keep the historical generation adapter as the CLI default."""
    args = build_parser().parse_args(
        ["--input", "input.jsonl", "--model-path", "model", "--output-dir", "out"]
    )

    assert args.adapter == "goedel"
    assert args.max_tokens < args.max_model_len


def test_cli_accepts_deepseek_adapter():
    """Expose DeepSeek grouped generation without changing the Goedel default."""
    args = build_parser().parse_args(
        [
            "--input",
            "input.jsonl",
            "--model-path",
            "model",
            "--output-dir",
            "out",
            "--adapter",
            "deepseek",
        ]
    )

    assert args.adapter == "deepseek"
