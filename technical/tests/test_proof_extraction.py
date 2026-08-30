"""Tests for portable Lean proof extraction and standard assembly."""

from technical.src.generation.proof_extraction import (
    ProofAssemblyError,
    assemble_standard_submission,
    extract_last_lean_block,
    replace_theorem_body,
)


def test_extracts_last_lean_fence():
    """Select the final Lean fence when a response contains multiple attempts."""
    text = (
        "```lean\nexample : True := by trivial\n```\n"
        "```lean4\ntheorem t : True := by trivial\n```"
    )

    assert extract_last_lean_block(text) == "theorem t : True := by trivial"


def test_returns_none_without_lean_fence():
    """Reject prose-only model output without inventing a submission."""
    assert extract_last_lean_block("analysis only") is None


def test_extracts_lean_fence_without_a_trailing_newline():
    """Accept a closing fence immediately after the generated Lean text."""
    assert extract_last_lean_block("```lean\ntheorem t : True := by trivial```") == (
        "theorem t : True := by trivial"
    )


def test_assembly_keeps_benchmark_statement_and_uses_generated_body():
    """Keep the locked declaration while replacing only its proof body."""
    statement = "theorem target : True := by sorry"
    output = "```lean4\ntheorem other : True := by\n  trivial\n```"

    assert assemble_standard_submission(statement, output) == (
        "theorem target : True := by\n  trivial"
    )


def test_replace_theorem_body_removes_comments_before_assembly():
    """Ignore comments while locating the locked placeholder and generated body."""
    statement = "-- benchmark\ntheorem target : True := by sorry\n/- trailing -/"
    generated = "theorem other : True := by\n  -- proof comment\n  trivial"

    assert replace_theorem_body(statement, generated) == "theorem target : True := by\n  \n  trivial"


def test_assembly_rejects_a_missing_benchmark_placeholder():
    """Fail explicitly when the benchmark has no replaceable theorem proof."""
    try:
        assemble_standard_submission("theorem target : True := by trivial", "```lean\ntheorem x : True := by trivial\n```")
    except ProofAssemblyError as error:
        assert error.condition == "missing benchmark theorem placeholder"
        assert "theorem target" not in str(error)
    else:
        raise AssertionError("missing benchmark placeholder was accepted")


def test_assembly_rejects_a_missing_generated_theorem_body():
    """Fail explicitly when the final Lean block has no theorem body."""
    try:
        assemble_standard_submission("theorem target : True := by sorry", "```lean\nexample : True := by trivial\n```")
    except ProofAssemblyError as error:
        assert error.condition == "missing generated theorem body"
        assert "example : True" not in str(error)
    else:
        raise AssertionError("missing generated theorem body was accepted")


def test_assembly_rejects_forbidden_search_tactics():
    """Forbid apply? and exact? before constructing a public submission."""
    statement = "theorem target : True := by sorry"
    output = "```lean\ntheorem other : True := by\n  exact?\n```"

    try:
        assemble_standard_submission(statement, output)
    except ProofAssemblyError as error:
        assert error.condition == "forbidden tactic"
        assert str(error) == "forbidden tactic: exact?"
    else:
        raise AssertionError("forbidden tactic was accepted")


def test_assembly_rejects_apply_search_tactic():
    """Reject apply? with the same typed public error as exact?."""
    try:
        replace_theorem_body(
            "theorem target : True := by sorry",
            "theorem other : True := by\n  apply?",
        )
    except ProofAssemblyError as error:
        assert error.condition == "forbidden tactic"
        assert error.tactic == "apply?"
    else:
        raise AssertionError("apply? was accepted")
