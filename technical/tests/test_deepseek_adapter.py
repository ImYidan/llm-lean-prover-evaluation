"""Tests for the DeepSeek prompt and proof-extraction adapter."""

import pytest

from technical.src.generation.adapters.deepseek import (
    DeepSeekPromptAdapter,
    assemble_deepseek_submission,
    extract_deepseek_proof,
)
from technical.src.generation.proof_extraction import ProofAssemblyError


class FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        return "CHAT::" + messages[0]["content"]


def test_prompt_preserves_supplied_theorem_header():
    """Keep the supplied declaration text unchanged in the official prompt."""
    statement = "theorem target (n : Nat) : n = n := by sorry"

    prompt, messages = DeepSeekPromptAdapter().build_prompt(statement, FakeTokenizer())

    assert prompt.startswith("CHAT::Complete the following Lean 4 code:")
    assert f"```lean4\n{statement}\n```" in messages[0]["content"]
    assert "detailed proof plan" in messages[0]["content"]


def test_prompt_reads_lean_code_from_a_benchmark_row():
    """Accept the benchmark-row boundary used by grouped generation."""
    row = {"lean4_code": "theorem target : True := by sorry"}

    _, messages = DeepSeekPromptAdapter().build_prompt(row, FakeTokenizer())

    assert "```lean4\ntheorem target : True := by sorry\n```" in messages[0]["content"]


def test_extracts_last_lean_fence():
    """Use the final Lean fence when earlier planning contains an attempt."""
    output = (
        "plan\n```lean4\nexample : False := by sorry\n```\n"
        "final\n```lean4\ntheorem p : True := by trivial\n```"
    )

    assert extract_deepseek_proof(output) == "theorem p : True := by trivial"


def test_rejects_missing_fenced_proof():
    """Report extraction failure instead of manufacturing Lean from prose."""
    with pytest.raises(ProofAssemblyError, match="missing fenced Lean"):
        extract_deepseek_proof("proof plan only")


@pytest.mark.parametrize("token", ["sorry", "admit", "apply?", "exact?"])
def test_rejects_blocked_tokens(token):
    """Reject standalone unsupported proof tokens before verification."""
    with pytest.raises(ProofAssemblyError, match="blocked"):
        extract_deepseek_proof(f"```lean4\ntheorem p : True := by {token}\n```")


def test_allows_blocked_token_substrings_inside_identifiers():
    """Do not reject ordinary identifiers that merely contain blocked text."""
    proof = "```lean4\ntheorem p : True := by\n  exactProof\n```"

    assert extract_deepseek_proof(proof) == "theorem p : True := by\n  exactProof"


def test_assembly_uses_the_locked_benchmark_header():
    """Delegate declaration substitution to the shared theorem-body helper."""
    statement = "theorem target : True := by sorry"
    output = "```lean4\ntheorem generated : True := by\n  trivial\n```"

    assert assemble_deepseek_submission(statement, output) == (
        "theorem target : True := by\n  trivial"
    )
