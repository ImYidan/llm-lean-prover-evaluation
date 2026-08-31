"""Prompt and extraction policy for DeepSeek-Prover-V2."""

from __future__ import annotations

import re
from collections.abc import Mapping

from technical.src.generation.proof_extraction import (
    ProofAssemblyError,
    extract_last_lean_block,
    replace_theorem_body,
)


_TASK_PREFIX = "Complete the following Lean 4 code:"
_DETAILED_PROOF_PLAN_INSTRUCTION = (
    "Before producing the Lean 4 code to formally prove the given theorem, "
    "provide a detailed proof plan outlining the main proof steps and strategies.\n"
    "The plan should highlight key ideas, intermediate lemmas, and proof "
    "structures that will guide the construction of the final formal proof."
)
_BLOCKED_PROOF_TOKEN = re.compile(
    r"(?<![\w'])(sorry|admit|apply\?|exact\?)(?![\w'])"
)


def deepseek_prompt_contract() -> dict:
    """Return the stable public semantics hashed into reusable run manifests."""
    return {
        "version": 1,
        "task_prefix": _TASK_PREFIX,
        "detailed_proof_plan_instruction": _DETAILED_PROOF_PLAN_INSTRUCTION,
        "chat_template": {
            "implementation": "tokenizer.apply_chat_template",
            "tokenize": False,
            "add_generation_prompt": True,
        },
    }


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
            f"{_TASK_PREFIX}\n\n"
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


def assemble_deepseek_proofnet_submission(
    statement: str, model_output: str
) -> str:
    """Apply DeepSeek extraction policy before target-aware ProofNet assembly."""
    generated_code = extract_deepseek_proof(model_output)
    from technical.src.benchmarks.proofnet import assemble_proofnet_submission

    return assemble_proofnet_submission(statement, generated_code)
