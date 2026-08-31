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
DEEPSEEK_MODEL_ID = "deepseek-ai/DeepSeek-Prover-V2-7B"
DEEPSEEK_MODEL_KEYS = ("dtype",)
DEEPSEEK_GENERATION_KEYS = (
    "seed",
    "temperature",
    "top_p",
    "gpu_memory_utilization",
)
BENCHMARK_GENERATION_KEYS = ("strategy",)
VERIFICATION_KEYS = ("mode", "timeout")
RUN_GENERATION_KEYS = ("sample_schedule", "max_model_len", "max_tokens")
LEAN_PROFILE_KEYS = ("profile",)
RUN_VERIFICATION_MODES = ("full_header_repl",)
ASSEMBLY_KEYS = ("mode",)
RUN_ASSEMBLY_MODES = ("standard", "proofnet")


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
    if model["id"] == DEEPSEEK_MODEL_ID:
        _require_keys(model, "model", DEEPSEEK_MODEL_KEYS)
        _require_keys(generation, "generation", DEEPSEEK_GENERATION_KEYS)
        _require_non_empty_string(model["dtype"], "model.dtype")
        _require_unit_interval(
            generation["gpu_memory_utilization"],
            "generation.gpu_memory_utilization",
        )
    else:
        _require_keys(generation, "generation", MODEL_GENERATION_KEYS)


def _validate_benchmark_config(config: dict[str, Any]) -> None:
    generation = _require_mapping(config, "generation")
    verification = _require_mapping(config, "verification")
    _require_keys(generation, "generation", BENCHMARK_GENERATION_KEYS)
    _require_keys(verification, "verification", VERIFICATION_KEYS)


def _require_positive_int(value: Any, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _require_non_empty_string(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_unit_interval(value: Any, name: str) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not 0 < float(value) <= 1
    ):
        raise ValueError(f"{name} must satisfy 0 < value <= 1")


def _validate_sample_schedule(sample_schedule: Any) -> list[int]:
    if not isinstance(sample_schedule, list) or not sample_schedule:
        raise ValueError("generation.sample_schedule must be a non-empty list")
    for index, samples in enumerate(sample_schedule):
        _require_positive_int(samples, f"generation.sample_schedule[{index}]")
    return sample_schedule


def cumulative_generation_offsets(sample_schedule: list[int]) -> list[int]:
    """Return the generation offset at the start of every cumulative stage."""
    validated_schedule = _validate_sample_schedule(sample_schedule)
    offsets = []
    next_offset = 0
    for samples in validated_schedule:
        offsets.append(next_offset)
        next_offset += samples
    return offsets


def _validate_run_config(config: dict[str, Any]) -> None:
    generation = _require_mapping(config, "generation")
    lean = _require_mapping(config, "lean")
    verification = _require_mapping(config, "verification")
    assembly = _require_mapping(config, "assembly")
    _require_keys(generation, "generation", RUN_GENERATION_KEYS)
    _require_keys(lean, "lean", LEAN_PROFILE_KEYS)
    _require_keys(verification, "verification", VERIFICATION_KEYS)
    _require_keys(assembly, "assembly", ASSEMBLY_KEYS)

    _validate_sample_schedule(generation["sample_schedule"])
    _require_positive_int(generation["max_model_len"], "generation.max_model_len")
    _require_positive_int(generation["max_tokens"], "generation.max_tokens")
    _require_positive_int(verification["timeout"], "verification.timeout")
    if generation["max_tokens"] > generation["max_model_len"]:
        raise ValueError(
            "generation.max_tokens must not exceed generation.max_model_len"
        )
    if not isinstance(lean["profile"], str) or not lean["profile"]:
        raise ValueError("lean.profile must be a non-empty string")
    if not isinstance(verification["mode"], str):
        raise ValueError("verification.mode must be a string")
    if verification["mode"] not in RUN_VERIFICATION_MODES:
        raise ValueError(f"unsupported verification.mode: {verification['mode']}")
    if not isinstance(assembly["mode"], str):
        raise ValueError("assembly.mode must be a string")
    if assembly["mode"] not in RUN_ASSEMBLY_MODES:
        raise ValueError(f"unsupported assembly.mode: {assembly['mode']}")


def validate_model_run_config(model_config: dict, run_config: dict) -> None:
    """Validate cross-profile constraints that depend on the selected model."""
    model = _require_mapping(model_config, "model")
    generation = _require_mapping(run_config, "generation")
    if (
        model.get("id") == DEEPSEEK_MODEL_ID
        and generation["max_tokens"] >= generation["max_model_len"]
    ):
        raise ValueError(
            "generation.max_tokens must be less than generation.max_model_len"
        )


def load_config(path: Path) -> dict:
    """Load a model, benchmark, or run YAML configuration with its schema."""
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("configuration root must be a mapping")

    generation = loaded.get("generation")
    is_run_profile = "lean" in loaded or (
        isinstance(generation, dict) and "sample_schedule" in generation
    )
    if is_run_profile:
        _validate_run_config(loaded)
    elif "verification" in loaded:
        _validate_benchmark_config(loaded)
    else:
        _validate_model_config(loaded)
    return loaded
