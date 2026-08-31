"""Prompt and extraction policy for DeepSeek-Prover-V2."""

from __future__ import annotations

import re
from collections.abc import Mapping

from technical.src.generation.proof_extraction import (
    ProofAssemblyError,
    extract_last_lean_block,
    replace_theorem_body,
)


_DETAILED_PROOF_PLAN_INSTRUCTION = (
    "Before producing the Lean 4 code to formally prove the given theorem, "
    "provide a detailed proof plan outlining the main proof steps and strategies.\n"
    "The plan should highlight key ideas, intermediate lemmas, and proof "
    "structures that will guide the construction of the final formal proof."
)
_BLOCKED_PROOF_TOKEN = re.compile(
    r"(?<![\w'])(sorry|admit|apply\?|exact\?)(?![\w'])"
)


class DeepSeekPromptAdapter:
    """Build the official detailed-proof-plan prompt without altering Lean."""

    def build_prompt(
        self, benchmark_row: str | Mapping[str, str], tokenizer
    ) -> tuple[str, list[dict]]:
        lean4_code = (
            benchmark_row["lean4_code"]
            if isinstance(benchmark_row, Mapping)
            else benchmark_row
        )
        content = (
            "Complete the following Lean 4 code:\n\n"
            f"```lean4\n{lean4_code}\n```\n\n"
            f"{_DETAILED_PROOF_PLAN_INSTRUCTION}"
        )
        messages = [{"role": "user", "content": content}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt, messages


def extract_deepseek_proof(model_output: str) -> str:
    """Return the final approved Lean block or raise a typed extraction error."""
    generated_code = extract_last_lean_block(model_output)
    if generated_code is None:
        raise ProofAssemblyError("missing fenced Lean code block")

    blocked = _BLOCKED_PROOF_TOKEN.search(generated_code)
    if blocked is not None:
        raise ProofAssemblyError("blocked proof token", blocked.group(1))
    return generated_code


def assemble_deepseek_submission(statement: str, model_output: str) -> str:
    """Assemble extracted DeepSeek code with the locked benchmark theorem."""
    return replace_theorem_body(statement, extract_deepseek_proof(model_output))
