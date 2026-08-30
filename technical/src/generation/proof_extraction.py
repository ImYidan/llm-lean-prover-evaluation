"""Extract Lean proofs from model output and assemble locked submissions."""

from __future__ import annotations

import re


class ProofAssemblyError(ValueError):
    """Raised when a generated Lean proof cannot form a standard submission."""

    def __init__(self, condition: str, tactic: str | None = None) -> None:
        self.condition = condition
        self.tactic = tactic
        message = f"{condition}: {tactic}" if tactic is not None else condition
        super().__init__(message)


_LEAN_FENCE = re.compile(r"```(?:lean4|lean)[^\S\r\n]*\r?\n(.*?)```", re.DOTALL)
_BLOCK_COMMENT = re.compile(r"/-.*?-/", re.DOTALL)
_BENCHMARK_THEOREM = re.compile(r"\btheorem\b.*?:=\s*by\s+sorry\b", re.DOTALL)
_GENERATED_THEOREM = re.compile(r"\btheorem\b.*?:=\s*by\b", re.DOTALL)
_FORBIDDEN_TACTIC = re.compile(r"\b(apply\?|exact\?)")


def extract_last_lean_block(text: str) -> str | None:
    """Return the content of the final ``lean`` or ``lean4`` fenced block."""
    matches = _LEAN_FENCE.findall(text)
    return matches[-1].strip() if matches else None


def _remove_comments(text: str) -> str:
    """Remove Lean block and line comments for the local assembly parser."""
    without_blocks = _BLOCK_COMMENT.sub("", text)
    return "\n".join(line.split("--", 1)[0] for line in without_blocks.splitlines()).strip()


def _raise_if_forbidden(generated_code: str) -> None:
    match = _FORBIDDEN_TACTIC.search(generated_code)
    if match:
        raise ProofAssemblyError("forbidden tactic", match.group(1))


def replace_theorem_body(statement: str, generated_code: str) -> str:
    """Keep a benchmark theorem declaration and substitute its generated body.

    Comments are removed only for locating and assembling the declaration; this
    avoids treating prose comments as parts of either theorem.
    """
    _raise_if_forbidden(generated_code)
    benchmark = _remove_comments(statement)
    generated = _remove_comments(generated_code)

    benchmark_match = _BENCHMARK_THEOREM.search(benchmark)
    if benchmark_match is None:
        raise ProofAssemblyError("missing benchmark theorem placeholder")

    generated_match = _GENERATED_THEOREM.search(generated)
    if generated_match is None:
        raise ProofAssemblyError("missing generated theorem body")

    benchmark_prefix = benchmark[: benchmark_match.end()].removesuffix("sorry").rstrip()
    return benchmark_prefix + generated[generated_match.end() :]


def assemble_standard_submission(statement: str, model_output: str) -> str | None:
    """Extract a final Lean block and assemble it against a locked statement."""
    generated_code = extract_last_lean_block(model_output)
    if generated_code is None:
        return None
    return replace_theorem_body(statement, generated_code)
