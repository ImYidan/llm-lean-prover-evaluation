"""Model-specific prompt adapters for the public generation runner."""

from technical.src.generation.adapters.deepseek import (
    DeepSeekPromptAdapter,
    assemble_deepseek_submission,
    extract_deepseek_proof,
)

__all__ = [
    "DeepSeekPromptAdapter",
    "assemble_deepseek_submission",
    "extract_deepseek_proof",
]
