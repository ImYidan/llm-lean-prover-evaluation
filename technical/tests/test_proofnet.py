"""Tests for ProofNet-specific IDs, assembly, and staged sampling."""

from technical.src.benchmarks.proofnet import (
    ProofNetPromptAdapter,
    assemble_proofnet_model_output,
    assemble_proofnet_submission,
    prepare_unique_rows,
)
from technical.src.benchmarks.run_proofnet_incremental import (
    make_attempts,
    run_incremental_stages,
)
from technical.src.generation.adapters.goedel import GoedelPromptAdapter
from technical.src.generation.candidates import Candidate
from technical.src.generation.generate import generate_records


def test_prepare_unique_rows_disambiguates_duplicate_problem_ids():
    rows = [
        {"problem_id": "duplicate", "lean4_code": "theorem a : True := by sorry"},
        {"problem_id": "duplicate", "lean4_code": "theorem b : True := by sorry"},
    ]

    prepared, report = prepare_unique_rows(rows)

    assert len({row["problem_id"] for row in prepared}) == 2
    assert [row["source_problem_id"] for row in prepared] == ["duplicate", "duplicate"]
    assert report["input_rows"] == 2
    assert report["unique_output_problem_ids"] == 2
    assert "duplicate" in report["duplicate_source_problem_ids"]


def test_prepare_unique_rows_disambiguates_post_sanitization_collisions():
    rows = [
        {"problem_id": "a b", "lean4_code": "theorem a : True := by sorry"},
        {"problem_id": "a_b", "lean4_code": "theorem b : True := by sorry"},
    ]

    prepared, report = prepare_unique_rows(rows)

    assert len({row["problem_id"] for row in prepared}) == 2
    assert report["unique_output_problem_ids"] == 2


def test_assembly_targets_named_declaration_after_helper():
    statement = (
        "import Mathlib\n\n"
        "lemma helper : True := by trivial\n\n"
        "theorem target : True := by sorry"
    )
    generated = (
        "lemma generated_helper : True := by trivial\n\n"
        "theorem target : True := by\n  trivial"
    )

    assembled = assemble_proofnet_submission(statement, generated)

    assert assembled == (
        "import Mathlib\n\n"
        "lemma helper : True := by trivial\n\n"
        "theorem target : True := by\n  trivial"
    )
    assert "generated_helper" not in assembled


def test_proofnet_prompt_keeps_helper_and_real_target():
    statement = (
        "lemma helper : True := by trivial\n\n"
        "theorem target : True := by sorry"
    )

    class Tokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            return messages[0]["content"]

    prompt, messages = ProofNetPromptAdapter().build_prompt(statement, Tokenizer())

    assert "lemma helper : True := by trivial" in prompt
    assert "theorem target : True := by sorry" in prompt
    assert messages[0]["content"] == prompt


def test_assembly_supports_lemma_def_and_noncomputable_def_targets():
    declarations = (
        ("lemma", "lemma target : True := by sorry"),
        ("def", "def target : Nat := by sorry"),
        ("noncomputable def", "noncomputable def target : Nat := by sorry"),
    )
    for kind, statement in declarations:
        generated = f"{kind} target : {('True' if kind == 'lemma' else 'Nat')} := by\n  trivial"
        assembled = assemble_proofnet_submission(statement, generated)
        assert assembled.endswith(":= by\n  trivial")


def test_proofnet_model_output_assembly_extracts_final_lean_fence():
    statement = "theorem target : True := by sorry"
    output = (
        "proof plan\n```lean\ntheorem target : True := by\n  fail_if_success trivial\n```\n"
        "```lean4\ntheorem target : True := by\n  trivial\n```\ntrailing prose"
    )

    assembled = assemble_proofnet_model_output(statement, output)

    assert assembled == "theorem target : True := by\n  trivial"
    assert "```" not in assembled


def test_make_attempts_uses_shared_generation_offsets():
    rows = [{"problem_id": "p", "lean4_code": "theorem p : True := by sorry"}]

    attempts = make_attempts(rows, add_samples=2, offset=8)

    assert [record["problem_id"] for record in attempts] == ["p_g8", "p_g9"]


def test_incremental_stages_reach_exact_cumulative_checkpoints():
    rows = [{"problem_id": "p", "lean4_code": "theorem p : True := by sorry"}]
    calls = []

    def fake_generate(stage_rows, add_samples, offset):
        calls.append((add_samples, offset))
        return make_attempts(stage_rows, add_samples=add_samples, offset=offset)

    def fake_verify(records):
        return [
            {"name": record["problem_id"], "compilation_result": {"complete": False}}
            for record in records
        ]

    stages = run_incremental_stages(rows, fake_generate, fake_verify)

    assert calls == [(1, 0), (7, 1), (8, 8), (16, 16)]
    assert {target: len(stage["full_records"]) for target, stage in stages.items()} == {
        1: 1,
        8: 8,
        16: 16,
        32: 32,
    }
    assert {target: len(stage["verification"]) for target, stage in stages.items()} == {
        1: 1,
        8: 8,
        16: 16,
        32: 32,
    }


def test_shared_generator_accepts_proofnet_assembly_hook(tmp_path):
    class Tokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            return messages[0]["content"]

    class Backend:
        def generate(self, prompts):
            return [[Candidate("generated")] for _ in prompts]

    def assembler(statement, output):
        assert output == "generated"
        return statement.replace("sorry", "trivial")

    full_records, _ = generate_records(
        [{"problem_id": "p", "lean4_code": "theorem p : True := by sorry"}],
        tokenizer=Tokenizer(),
        backend=Backend(),
        adapter=GoedelPromptAdapter(),
        assembler=assembler,
        output_dir=tmp_path,
        samples=1,
    )

    assert full_records[0]["full_code"].endswith("by trivial")
