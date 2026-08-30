from pathlib import Path

from technical.src.config import load_config


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


def test_load_config_returns_benchmark_verification_settings():
    """Catch a loader that treats benchmark configuration as a model config."""
    config = load_config(ROOT / "technical/configs/benchmarks/proofnet.yaml")

    assert config["generation"]["strategy"] == "full_header"
    assert config["verification"]["mode"] == "full_header_file"
    assert config["verification"]["timeout"] == 300
