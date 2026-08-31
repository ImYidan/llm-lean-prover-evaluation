"""Tests for resumable grouped DeepSeek generation."""

import json
import sys
import types

import pytest

from technical.src.benchmarks.proofnet import assemble_proofnet_model_output
from technical.src.generation import generate as generate_cli
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


class StaticGroupedBackend:
    def __init__(self, text):
        self.text = text

    def generate(self, prompts):
        return [[Candidate(self.text, finish_reason="stop", stop_reason=None)]]


class BackendMustNotBeCalled:
    def __init__(self):
        self.calls = 0

    def generate(self, prompts):
        self.calls += 1
        raise AssertionError("backend.generate must not be called")


def source_rows(count=2):
    return [
        {
            "problem_id": f"p{index}",
            "lean4_code": f"theorem p{index} : True := by sorry",
        }
        for index in range(count)
    ]


def valid_checkpoint_record(problem_id="p0_g4", origin_problem_id="p0"):
    return {
        "problem_id": problem_id,
        "origin_problem_id": origin_problem_id,
        "id_maps": [
            {"origin_problem_id": origin_problem_id},
            {"generation_id": problem_id},
        ],
        "lean4_code": f"theorem {origin_problem_id} : True := by sorry",
        "model_input": "prompt",
        "messages_history_for_this_attempt": [{"role": "user", "content": "prompt"}],
        "model_output": "output",
        "raw_response": "output",
        "full_code": f"theorem {origin_problem_id} : True := by\n  trivial",
        "extraction_status": "success",
        "finish_reason": "stop",
        "stop_reason": 128001,
    }


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


@pytest.mark.parametrize(
    "missing_field",
    [
        "origin_problem_id",
        "id_maps",
        "lean4_code",
        "model_input",
        "messages_history_for_this_attempt",
        "model_output",
        "raw_response",
        "full_code",
        "extraction_status",
        "finish_reason",
        "stop_reason",
    ],
)
def test_resume_rejects_missing_raw_checkpoint_fields_before_backend(
    tmp_path, missing_field
):
    """Catch malformed complete-prefix checkpoints before model calls or append."""
    record = valid_checkpoint_record()
    record.pop(missing_field)
    append_checkpoint(tmp_path / "inference.jsonl", record)
    backend = BackendMustNotBeCalled()

    with pytest.raises(ValueError, match=f"missing required field {missing_field}"):
        generate_deepseek_records(
            source_rows(2),
            tokenizer=FakeTokenizer(),
            backend=backend,
            output_dir=tmp_path,
            samples=1,
            generation_offset=4,
        )

    assert backend.calls == 0
    assert len(load_checkpoint(tmp_path / "inference.jsonl")) == 1


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("origin_problem_id", "other", "expected origin p0"),
        ("lean4_code", "theorem other : True := by sorry", "source Lean code"),
        (
            "id_maps",
            [{"origin_problem_id": "p0"}, {"generation_id": "p0_g9"}],
            "unexpected id_maps",
        ),
    ],
)
def test_resume_rejects_conflicting_raw_checkpoint_values_before_backend(
    tmp_path, field, bad_value, message
):
    """Catch corrupt complete-prefix metadata before model calls or append."""
    record = valid_checkpoint_record()
    record[field] = bad_value
    append_checkpoint(tmp_path / "inference.jsonl", record)
    backend = BackendMustNotBeCalled()

    with pytest.raises(ValueError, match=message):
        generate_deepseek_records(
            source_rows(2),
            tokenizer=FakeTokenizer(),
            backend=backend,
            output_dir=tmp_path,
            samples=1,
            generation_offset=4,
        )

    assert backend.calls == 0
    assert load_checkpoint(tmp_path / "inference.jsonl") == [record]


def test_resume_rejects_checkpoint_gap(tmp_path):
    """Catch checkpoints that skip an expected grouped candidate ID."""
    append_checkpoint(tmp_path / "inference.jsonl", valid_checkpoint_record("p0_g4"))
    append_checkpoint(tmp_path / "inference.jsonl", valid_checkpoint_record("p0_g6"))

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
    append_checkpoint(tmp_path / "inference.jsonl", valid_checkpoint_record("p0_g3"))

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


def test_grouped_generation_uses_supplied_assembler(tmp_path):
    """Catch DeepSeek generation hardcoding the standard assembler."""
    calls = []

    def target_aware_assembler(statement, model_output):
        calls.append((statement, model_output))
        return statement.replace("sorry", "trivial") + "\n-- target-aware"

    full_records, _ = generate_deepseek_records(
        source_rows(1),
        tokenizer=FakeTokenizer(),
        backend=FakeGroupedBackend(1),
        output_dir=tmp_path,
        samples=1,
        generation_offset=4,
        assembler=target_aware_assembler,
    )

    assert calls == [
        (
            "theorem p0 : True := by sorry",
            "```lean4\ntheorem generated : True := by\n  trivial\n```",
        )
    ]
    assert full_records[0]["full_code"].endswith("-- target-aware")


def test_grouped_generation_marks_proofnet_unfenced_output_as_failed(tmp_path):
    """Catch ProofNet assembler returning None being recorded as success."""
    rows = [
        {
            "problem_id": "proofnet",
            "lean4_code": (
                "lemma helper : True := by trivial\n\n"
                "theorem target : True := by sorry"
            ),
        }
    ]

    full_records, _ = generate_deepseek_records(
        rows,
        tokenizer=FakeTokenizer(),
        backend=StaticGroupedBackend("proof plan only"),
        output_dir=tmp_path,
        samples=1,
        assembler=assemble_proofnet_model_output,
    )

    assert full_records[0]["full_code"] == "None"
    assert full_records[0]["extraction_status"] == "missing fenced Lean code block"


def test_grouped_generation_uses_real_proofnet_assembler_success_path(tmp_path):
    """Catch the assembler seam failing to support target-aware ProofNet output."""
    rows = [
        {
            "problem_id": "proofnet",
            "lean4_code": (
                "lemma helper : True := by trivial\n\n"
                "theorem target : True := by sorry"
            ),
        }
    ]
    output = (
        "plan\n```lean4\n"
        "lemma generated_helper : True := by trivial\n\n"
        "theorem target : True := by\n"
        "  trivial\n"
        "```"
    )

    full_records, _ = generate_deepseek_records(
        rows,
        tokenizer=FakeTokenizer(),
        backend=StaticGroupedBackend(output),
        output_dir=tmp_path,
        samples=1,
        assembler=assemble_proofnet_model_output,
    )

    assert full_records[0]["extraction_status"] == "success"
    assert full_records[0]["full_code"] == (
        "lemma helper : True := by trivial\n\n"
        "theorem target : True := by\n  trivial"
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
        dtype="bfloat16",
        gpu_memory_utilization=0.75,
    )

    assert tokenizer == "tokenizer"
    assert isinstance(backend, _VllmBackend)
    assert captured["sampling"]["n"] == 3
    assert captured["llm"]["dtype"] == "bfloat16"
    assert captured["llm"]["gpu_memory_utilization"] == 0.75


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


def test_vllm_backend_rejects_invalid_gpu_memory_utilization_before_import(
    monkeypatch,
):
    """Catch invalid model memory limits before importing optional heavy backends."""
    monkeypatch.delitem(sys.modules, "transformers", raising=False)
    monkeypatch.delitem(sys.modules, "vllm", raising=False)

    with pytest.raises(
        ValueError, match="gpu_memory_utilization must satisfy 0 < value <= 1"
    ):
        create_vllm_backend(
            model_path="model",
            seed=1,
            tensor_parallel_size=1,
            max_model_len=128,
            temperature=1.0,
            top_p=0.95,
            max_tokens=64,
            samples_per_prompt=1,
            gpu_memory_utilization=0,
        )


def test_vllm_backend_rejects_blank_dtype_before_import(monkeypatch):
    """Catch missing dtype values before importing optional heavy backends."""
    monkeypatch.delitem(sys.modules, "transformers", raising=False)
    monkeypatch.delitem(sys.modules, "vllm", raising=False)

    with pytest.raises(ValueError, match="dtype must be a non-empty string"):
        create_vllm_backend(
            model_path="model",
            seed=1,
            tensor_parallel_size=1,
            max_model_len=128,
            temperature=1.0,
            top_p=0.95,
            max_tokens=64,
            samples_per_prompt=1,
            dtype="",
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


def test_cli_accepts_backend_dtype_and_gpu_memory_utilization():
    """Expose reviewed DeepSeek model profile runtime settings to vLLM."""
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
            "--dtype",
            "bfloat16",
            "--gpu-memory-utilization",
            "0.90",
        ]
    )

    assert args.dtype == "bfloat16"
    assert args.gpu_memory_utilization == 0.90


def test_cli_accepts_deepseek_proofnet_assembly_mode():
    """Expose the target-aware ProofNet assembler for DeepSeek profile runs."""
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
            "--assembly-mode",
            "proofnet",
        ]
    )

    assert args.assembly_mode == "proofnet"


def test_deepseek_cli_proofnet_mode_prepares_rows_and_uses_target_assembler(
    tmp_path, monkeypatch
):
    """Catch a ProofNet run that falls back to the standard first-theorem assembler."""
    input_path = tmp_path / "proofnet.jsonl"
    source = (
        "lemma helper : True := by trivial\n\n"
        "theorem target : True := by sorry"
    )
    rows = [
        {"problem_id": "same id", "lean4_code": source},
        {"problem_id": "same id", "lean4_code": source.replace("target", "target_two")},
    ]
    input_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    output_dir = tmp_path / "out"
    captured = {}

    def fake_create_vllm_backend(**kwargs):
        captured["backend_kwargs"] = kwargs
        return FakeTokenizer(), object()

    def fake_generate_deepseek_records(rows, *, assembler, **kwargs):
        captured["rows"] = rows
        captured["kwargs"] = kwargs
        model_output = (
            "plan\n```lean4\n"
            "lemma generated_helper : True := by trivial\n\n"
            "theorem target : True := by\n"
            "  trivial\n"
            "```"
        )
        captured["assembled"] = assembler(rows[0]["lean4_code"], model_output)
        return [], []

    monkeypatch.setattr(generate_cli, "create_vllm_backend", fake_create_vllm_backend)
    import technical.src.generation.deepseek as deepseek_module

    monkeypatch.setattr(
        deepseek_module, "generate_deepseek_records", fake_generate_deepseek_records
    )

    result = generate_cli.main(
        [
            "--input",
            str(input_path),
            "--model-path",
            "model",
            "--output-dir",
            str(output_dir),
            "--adapter",
            "deepseek",
            "--assembly-mode",
            "proofnet",
            "--samples",
            "1",
            "--max-model-len",
            "128",
            "--max-tokens",
            "64",
            "--dtype",
            "bfloat16",
            "--gpu-memory-utilization",
            "0.80",
        ]
    )

    assert result == 0
    assert captured["backend_kwargs"]["dtype"] == "bfloat16"
    assert captured["backend_kwargs"]["gpu_memory_utilization"] == 0.80
    assert captured["assembled"] == (
        "lemma helper : True := by trivial\n\n"
        "theorem target : True := by\n  trivial"
    )
    assert captured["rows"][0]["problem_id"].startswith("same_id__row000__")
    assert captured["rows"][1]["problem_id"].startswith("same_id__row001__")
    assert (output_dir / "proofnet_duplicate_problem_ids.json").is_file()
