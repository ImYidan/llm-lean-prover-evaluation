from pathlib import Path

import pytest

from technical.src.config import cumulative_generation_offsets, load_config


ROOT = Path(__file__).parents[2]


def test_load_config_rejects_missing_model_section(tmp_path: Path):
    """Catch a loader that accepts an incomplete model configuration."""
    path = tmp_path / "config.yaml"
    path.write_text("generation: {}\n", encoding="utf-8")

    try:
        load_config(path)
    except ValueError as error:
        assert "model" in str(error)
    else:
        raise AssertionError("missing model section was accepted")


def test_load_config_returns_public_model_settings():
    """Catch a loader that drops nested model or generation settings."""
    config = load_config(ROOT / "technical/configs/models/goedel-prover-v2-32b.yaml")

    assert config["model"]["id"] == "Goedel-LM/Goedel-Prover-V2-32B"
    assert config["model"]["tensor_parallel_size"] == 4
    assert config["model"]["trust_remote_code"] is False
    assert config["model"]["revision"] is None
    assert config["generation"]["samples_per_problem"] == 32


def test_load_config_keeps_goedel_model_schema_requirements(tmp_path: Path):
    """Catch a DeepSeek schema extension that silently weakens Goedel validation."""
    path = tmp_path / "goedel.yaml"
    path.write_text(
        """\
model:
  id: Goedel-LM/Goedel-Prover-V2-32B
  tensor_parallel_size: 4
  trust_remote_code: false
  revision: null
generation:
  temperature: 1.0
  top_p: 0.95
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="generation.max_model_len"):
        load_config(path)


def test_load_config_returns_benchmark_verification_settings():
    """Catch a loader that treats benchmark configuration as a model config."""
    config = load_config(ROOT / "technical/configs/benchmarks/proofnet.yaml")

    assert config["generation"]["strategy"] == "full_header"
    assert config["verification"]["mode"] == "full_header_file"
    assert config["verification"]["timeout"] == 300


def test_load_config_returns_pinned_deepseek_model_settings():
    """Catch a profile that loses the reviewed DeepSeek runtime settings."""
    config = load_config(ROOT / "technical/configs/models/deepseek-prover-v2-7b.yaml")

    assert config["model"]["id"] == "deepseek-ai/DeepSeek-Prover-V2-7B"
    assert config["model"]["revision"] == "a8d9e14432b2e8dd9df2a4d4e70f1ba9bc8d9b7b"
    assert config["model"]["dtype"] == "bfloat16"
    assert config["model"]["tensor_parallel_size"] == 1
    assert config["generation"] == {
        "seed": 30,
        "temperature": 1.0,
        "top_p": 0.95,
        "gpu_memory_utilization": 0.90,
    }


@pytest.mark.parametrize(
    ("name", "samples", "model_len", "new_tokens", "lean_profile", "timeout"),
    [
        ("minif2f", [32], 32768, 8192, "mathlib-v49", 300),
        ("proofnet", [32], 40960, 32768, "mathlib-v49", 300),
        ("putnam", [1, 7, 8, 16], 40960, 32768, "mathlib-v49", 300),
        ("fate-m", [32], 32768, 8192, "fate-v428", 4000),
        ("fate-h", [32], 32768, 8192, "fate-v428", 4000),
    ],
)
def test_deepseek_run_profiles(
    name: str,
    samples: list[int],
    model_len: int,
    new_tokens: int,
    lean_profile: str,
    timeout: int,
):
    """Catch benchmark profiles with mismatched generation or Lean limits."""
    config = load_config(ROOT / f"technical/configs/runs/deepseek/{name}.yaml")

    assert config["generation"]["sample_schedule"] == samples
    assert config["generation"]["max_model_len"] == model_len
    assert config["generation"]["max_tokens"] == new_tokens
    assert config["lean"]["profile"] == lean_profile
    assert config["verification"] == {"mode": "full_header_repl", "timeout": timeout}


def test_putnam_offsets_derive_from_its_staged_sample_schedule():
    """Catch a cumulative Putnam stage that starts at the wrong generation ID."""
    config = load_config(ROOT / "technical/configs/runs/deepseek/putnam.yaml")

    assert cumulative_generation_offsets(config["generation"]["sample_schedule"]) == [0, 1, 8, 16]


def test_load_config_rejects_run_profile_without_a_lean_profile(tmp_path: Path):
    """Catch a loader that accepts a run mapping without its Lean environment."""
    path = tmp_path / "run.yaml"
    path.write_text(
        """\
generation:
  sample_schedule: [32]
  max_model_len: 32768
  max_tokens: 8192
verification:
  mode: full_header_repl
  timeout: 300
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required section: lean"):
        load_config(path)


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("42", "verification.mode must be a string"),
        ("standard_repl", "unsupported verification.mode: standard_repl"),
    ],
    ids=("wrong-type", "unsupported-mode"),
)
def test_load_config_rejects_invalid_run_profile_verification_mode(
    tmp_path: Path, mode: str, message: str
):
    """Catch a run profile selecting an unchecked verifier implementation."""
    path = tmp_path / "run.yaml"
    path.write_text(
        f"""\
generation:
  sample_schedule: [32]
  max_model_len: 32768
  max_tokens: 8192
lean:
  profile: mathlib-v49
verification:
  mode: {mode}
  timeout: 300
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_config(path)
