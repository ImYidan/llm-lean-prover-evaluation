"""Prompt and proof-assembly policy for Pythagoras-Prover-4B."""

from __future__ import annotations

import re
from collections.abc import Mapping

from technical.src.generation.adapters.goedel import build_prompt_from_statement
from technical.src.generation.proof_extraction import ProofAssemblyError


_BLOCK_COMMENT = re.compile(r"/-.*?-/", re.DOTALL)
_FENCE_PATTERNS = (
    re.compile(r"```lean4\n(.*?)\n```", re.DOTALL),
    re.compile(r"```lean4\n(.*?)```", re.DOTALL),
    re.compile(r"```lean\n(.*?)```", re.DOTALL),
)
_PLACEHOLDER = re.compile(
    r"(?:(?:noncomputable\s+)?def|theorem|lemma)\s+.*?:=\s*(?:by\s*)?sorry",
    re.DOTALL,
)
_DECLARATION = re.compile(
    r"(?:^|\s)(?:(?:noncomputable\s+)?def|theorem|lemma)\s+.*?:=",
    re.DOTALL,
)


def _remove_comments(text: str) -> str:
    without_blocks = _BLOCK_COMMENT.sub("", text)
    return "\n".join(
        line.split("--", 1)[0] for line in without_blocks.splitlines()
    ).strip()


def _statement_for_prompt(row: Mapping[str, str]) -> str:
    source = row.get("lean4_code") or row["formal_statement"]
    if ":= by" in source:
        return source.split(":= by", 1)[0] + ":= by sorry"
    if ":= sorry" in source:
        return source.split(":= sorry", 1)[0] + ":= by sorry"
    return source


class PythagorasPromptAdapter:
    """Build the Goedel-style prompt used by the final autoregressive runs."""

    def build_prompt(
        self, row: Mapping[str, str], tokenizer
    ) -> tuple[str, list[dict]]:
        return build_prompt_from_statement(_statement_for_prompt(row), tokenizer)


def _declaration_prefix(text: str) -> str | None:
    match = _DECLARATION.search(_remove_comments(text))
    if match is None:
        return None
    prefix = re.sub(r"\s+", " ", match.group(0)).strip()
    return re.sub(r"\s*:=\s*$", " :=", prefix)


def _extract(model_output: str) -> str:
    for pattern in _FENCE_PATTERNS:
        matches = pattern.findall(model_output)
        if matches:
            return matches[-1]
    cleaned = _remove_comments(model_output)
    if _DECLARATION.search(cleaned):
        return model_output
    raise ProofAssemblyError("no_proof_extracted")


def _ensure_max_heartbeats_zero(code: str) -> str:
    if re.search(r"set_option\s+maxHeartbeats\s+\d+", code):
        return re.sub(
            r"set_option\s+maxHeartbeats\s+\d+",
            "set_option maxHeartbeats 0",
            code,
        )
    lines = code.splitlines()
    insert_at = 0
    for index, line in enumerate(lines):
        if line.strip().startswith("import "):
            insert_at = index + 1
    lines.insert(insert_at, "set_option maxHeartbeats 0")
    return "\n".join(lines)


def assemble_pythagoras_submission(statement: str, model_output: str) -> str:
    """Apply bare fallback, verbatim guard, statement grafting, and safety checks."""
    generated = _extract(model_output)
    if _declaration_prefix(statement) != _declaration_prefix(generated):
        raise ProofAssemblyError("statement_not_verbatim")
    if "apply?" in generated or "exact?" in generated:
        raise ProofAssemblyError("forbidden_search_tactic")

    cleaned_statement = _remove_comments(statement)
    placeholder = _PLACEHOLDER.search(cleaned_statement)
    if placeholder is None:
        raise ProofAssemblyError("missing benchmark theorem placeholder")
    cleaned_generated = _remove_comments(generated)
    declaration = _DECLARATION.search(cleaned_generated)
    if declaration is None:
        raise ProofAssemblyError("missing generated theorem body")

    prefix = cleaned_statement[: placeholder.end()].replace("sorry", "").rstrip()
    tail = cleaned_generated[declaration.end() :]
    if prefix.endswith("by"):
        tail = re.sub(r"^[ \t]*by\b", "", tail, count=1)
    assembled = _ensure_max_heartbeats_zero(prefix + tail)
    if "sorry" in assembled or "admit" in assembled:
        raise ProofAssemblyError("contains_sorry")
    return assembled
