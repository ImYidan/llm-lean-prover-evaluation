"""Prompt and proof-assembly policy for Kimina-Prover-Distill-8B."""

from __future__ import annotations

import re
from collections.abc import Mapping

from technical.src.generation.proof_extraction import ProofAssemblyError


SYSTEM_PROMPT = "You are an expert in mathematics and Lean 4."
_LEAN_FENCE = re.compile(r"```(?:lean4|lean)[^\S\r\n]*\r?\n(.*?)```", re.DOTALL)
_LEAN_FENCE_OPEN = re.compile(r"```(?:lean4|lean)[^\S\r\n]*\r?\n")
_GENERATED_DECLARATION = re.compile(r"\b(?:theorem|lemma)\b.*?:=\s*by", re.DOTALL)
_FORBIDDEN = ("apply?", "exact?", "sorry", "admit")


def _strip_doc_comment(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^/--\s*", "", text)
    text = re.sub(r"\s*-/\s*$", "", text)
    return text.strip()


def _formal_statement_for_prompt(row: Mapping[str, str]) -> str:
    formal = (row.get("formal_statement") or "").strip()
    if not formal:
        formal = row["lean4_code"].split(":= by", 1)[0] + ":= by"
    if re.search(r":=\s*by\b", formal):
        return re.split(r":=\s*by\b", formal, maxsplit=1)[0].rstrip() + " := by"
    if re.search(r":=\s*$", formal):
        return re.sub(r":=\s*$", ":= by", formal)
    return formal


class KiminaPromptAdapter:
    """Build the two-message prompt used by the recorded Kimina runs."""

    def build_prompt(
        self, row: Mapping[str, str], tokenizer
    ) -> tuple[str, list[dict]]:
        problem = _strip_doc_comment(row.get("informal_prefix", ""))
        formal = _formal_statement_for_prompt(row)
        user = "Think about and solve the following problem step by step in Lean 4."
        user += f"\n# Problem:{problem}"
        user += f"\n# Formal statement:\n```lean4\n{formal}\n```\n"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return prompt, messages


def _extract(model_output: str, *, lenient: bool) -> str:
    matches = _LEAN_FENCE.findall(model_output)
    if matches:
        return matches[-1].strip()
    if lenient:
        openings = list(_LEAN_FENCE_OPEN.finditer(model_output))
        if openings:
            return model_output[openings[-1].end() :].strip()
    raise ProofAssemblyError("no_extractable_final_lean")


def extract_kimina_proof(model_output: str) -> str:
    """Extract the final closed Lean fence using Kimina's strict policy."""
    return _extract(model_output, lenient=False)


def _assemble(
    statement: str,
    model_output: str,
    *,
    lenient: bool,
    reject_forbidden: bool,
) -> str:
    generated = _extract(model_output, lenient=lenient)
    if reject_forbidden:
        lowered = generated.lower()
        blocked = [token for token in _FORBIDDEN if token in lowered]
        if blocked:
            raise ProofAssemblyError("forbidden_term", ",".join(blocked))
    match = _GENERATED_DECLARATION.search(generated)
    if match is None:
        raise ProofAssemblyError("no_theorem_header_before_cutoff")
    if ":= by" not in statement:
        raise ProofAssemblyError("missing benchmark theorem placeholder")
    prefix = statement.split(":= by", 1)[0] + ":= by"
    return prefix + generated[match.end() :]


def assemble_kimina_submission(statement: str, model_output: str) -> str:
    """Strict generation-time extraction and canonical statement grafting."""
    return _assemble(
        statement, model_output, lenient=False, reject_forbidden=True
    )


def recover_kimina_submission(statement: str, model_output: str) -> str:
    """Recover an unclosed final fence for verification-time classification."""
    return _assemble(
        statement, model_output, lenient=True, reject_forbidden=False
    )
