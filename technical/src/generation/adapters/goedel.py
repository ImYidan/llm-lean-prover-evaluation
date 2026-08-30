"""Prompt construction for Goedel-Prover-V2 models."""

from __future__ import annotations


def build_prompt_from_statement(
    formal_statement: str, tokenizer
) -> tuple[str, list[dict]]:
    """Render the shared Goedel proof-planning prompt for a locked statement."""
    content = (
            "Complete the following Lean 4 code:\n\n"
            f"```lean4\n{formal_statement}```\n\n"
            "Before producing the Lean 4 code to formally prove the given "
            "theorem, provide a detailed proof plan outlining the main proof "
            "steps and strategies.\n"
            "The plan should highlight key ideas, intermediate lemmas, and proof "
            "structures that will guide the construction of the final formal proof."
    )
    messages = [{"role": "user", "content": content}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt, messages


class GoedelPromptAdapter:
    """Build the standard proof-planning prompt used by Goedel-Prover-V2-32B."""

    def build_prompt(self, lean4_code: str, tokenizer) -> tuple[str, list[dict]]:
        formal_statement = lean4_code.split(":= by", 1)[0] + ":= by sorry"
        return build_prompt_from_statement(formal_statement, tokenizer)
