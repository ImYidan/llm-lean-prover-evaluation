"""Model-specific prompt adapters for the public generation runner."""

from technical.src.generation.adapters.deepseek import (
    DeepSeekPromptAdapter,
    assemble_deepseek_submission,
    extract_deepseek_proof,
)
from technical.src.generation.adapters.kimina import KiminaPromptAdapter
from technical.src.generation.adapters.pythagoras import PythagorasPromptAdapter

__all__ = [
    "DeepSeekPromptAdapter",
    "assemble_deepseek_submission",
    "extract_deepseek_proof",
    "KiminaPromptAdapter",
    "PythagorasPromptAdapter",
]
