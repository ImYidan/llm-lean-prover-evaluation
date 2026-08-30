"""Validation and loading for portable pipeline configuration files."""

from pathlib import Path
from typing import Any

import yaml


MODEL_KEYS = ("id", "tensor_parallel_size", "trust_remote_code", "revision")
MODEL_GENERATION_KEYS = (
    "temperature",
    "top_p",
    "max_model_len",
    "samples_per_problem",
)
BENCHMARK_GENERATION_KEYS = ("strategy",)
VERIFICATION_KEYS = ("mode", "timeout")


def _require_mapping(config: dict[str, Any], section: str) -> dict[str, Any]:
    if section not in config:
        raise ValueError(f"missing required section: {section}")
    value = config[section]
    if not isinstance(value, dict):
        raise ValueError(f"configuration section '{section}' must be a mapping")
    return value


def _require_keys(section: dict[str, Any], name: str, keys: tuple[str, ...]) -> None:
    for key in keys:
        if key not in section:
            raise ValueError(f"missing required key: {name}.{key}")


def _validate_model_config(config: dict[str, Any]) -> None:
    model = _require_mapping(config, "model")
    generation = _require_mapping(config, "generation")
    _require_keys(model, "model", MODEL_KEYS)
    _require_keys(generation, "generation", MODEL_GENERATION_KEYS)


def _validate_benchmark_config(config: dict[str, Any]) -> None:
    generation = _require_mapping(config, "generation")
    verification = _require_mapping(config, "verification")
    _require_keys(generation, "generation", BENCHMARK_GENERATION_KEYS)
    _require_keys(verification, "verification", VERIFICATION_KEYS)


def load_config(path: Path) -> dict:
    """Load a model or benchmark YAML configuration with its required schema."""
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("configuration root must be a mapping")

    if "verification" in loaded:
        _validate_benchmark_config(loaded)
    else:
        _validate_model_config(loaded)
    return loaded
