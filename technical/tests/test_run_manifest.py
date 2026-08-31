"""Tests for hash-bound reusable run artifacts."""

import json
from pathlib import Path

import pytest
import yaml

from technical.src.run_manifest import (
    ManifestConflictError,
    build_run_manifest,
    canonical_sha256,
    ensure_run_manifest,
    main as manifest_main,
    validate_completion_marker,
    write_completion_marker,
)


ROOT = Path(__file__).parents[2]
MODEL_CONFIG = ROOT / "technical/configs/models/deepseek-prover-v2-7b.yaml"
RUN_CONFIG = ROOT / "technical/configs/runs/deepseek/putnam.yaml"
MINIF2F_RUN_CONFIG = ROOT / "technical/configs/runs/deepseek/minif2f.yaml"
VALID_INPUT = '{"problem_id":"p","lean4_code":"theorem p : True := by sorry"}\n'


def test_canonical_hash_is_stable_across_mapping_order():
    """Catch manifests hashing serialization order instead of config semantics."""
    assert canonical_sha256({"b": 2, "a": [1, 3]}) == canonical_sha256(
        {"a": [1, 3], "b": 2}
    )


def test_run_manifest_binds_public_semantics_without_private_paths(tmp_path):
    """Catch reusable outputs missing prompt, config, input, or chunk identity."""
    input_path = tmp_path / "private-location/putnam.jsonl"
    input_path.parent.mkdir()
    input_path.write_text(VALID_INPUT, encoding="utf-8")

    manifest = build_run_manifest(
        model_config_path=MODEL_CONFIG,
        run_config_path=RUN_CONFIG,
        input_path=input_path,
        pipeline="deepseek-putnam",
        chunk_count=8,
    )

    assert manifest["model"]["id"] == "deepseek-ai/DeepSeek-Prover-V2-7B"
    assert manifest["model"]["revision"] == (
        "a8d9e14432b2e8dd9df2a4d4e70f1ba9bc8d9b7b"
    )
    assert manifest["sampling"]["sample_schedule"] == [1, 7, 8, 16]
    assert manifest["assembly"] == {"mode": "standard"}
    assert manifest["chunk_count"] == 8
    assert len(manifest["input_sha256"]) == 64
    assert len(manifest["prompt"]["contract_sha256"]) == 64
    serialized = json.dumps(manifest, sort_keys=True)
    assert str(tmp_path) not in serialized
    assert "putnam.jsonl" not in serialized
    assert "/home/" not in serialized


def test_run_manifest_allowlists_semantics_instead_of_copying_extension_fields(
    tmp_path,
):
    """Catch local extension fields being serialized into a path-free manifest."""
    private_value = "/" + "home/private/checkpoint"
    model_config = yaml.safe_load(MODEL_CONFIG.read_text(encoding="utf-8"))
    run_config = yaml.safe_load(RUN_CONFIG.read_text(encoding="utf-8"))
    for section in ("model", "generation"):
        model_config[section]["local_extension"] = private_value
    for section in ("generation", "lean", "verification", "assembly"):
        run_config[section]["local_extension"] = private_value
    model_path = tmp_path / "model.yaml"
    run_path = tmp_path / "run.yaml"
    model_path.write_text(yaml.safe_dump(model_config), encoding="utf-8")
    run_path.write_text(yaml.safe_dump(run_config), encoding="utf-8")
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(VALID_INPUT, encoding="utf-8")

    manifest = build_run_manifest(
        model_config_path=model_path,
        run_config_path=run_path,
        input_path=input_path,
        pipeline="deepseek-putnam",
        chunk_count=2,
    )

    assert private_value not in json.dumps(manifest, sort_keys=True)
    assert set(manifest["model"]) == {
        "id",
        "revision",
        "dtype",
        "tensor_parallel_size",
        "trust_remote_code",
    }
    assert set(manifest["sampling"]) == {
        "seed",
        "temperature",
        "top_p",
        "gpu_memory_utilization",
        "sample_schedule",
        "max_model_len",
        "max_tokens",
    }


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("model", "revision"),
        ("generation", "temperature"),
    ],
)
def test_run_manifest_rejects_non_scalar_values_in_allowlisted_fields(
    tmp_path, section, field
):
    """Catch private mappings hidden inside a manifest's approved field names."""
    model_config = yaml.safe_load(MODEL_CONFIG.read_text(encoding="utf-8"))
    model_config[section][field] = {"path": "/" + "home/private/value"}
    model_path = tmp_path / "model.yaml"
    model_path.write_text(yaml.safe_dump(model_config), encoding="utf-8")
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(VALID_INPUT, encoding="utf-8")

    with pytest.raises(ValueError):
        build_run_manifest(
            model_config_path=model_path,
            run_config_path=RUN_CONFIG,
            input_path=input_path,
            pipeline="deepseek-putnam",
            chunk_count=2,
        )


@pytest.mark.parametrize(
    "input_text",
    [
        "not-json\n",
        "[]\n",
        "{}\n",
        '{"problem_id":"p","lean4_code":42}\n',
        "\n",
    ],
)
def test_manifest_cli_rejects_invalid_jsonl_without_writing_manifest(
    tmp_path, input_text
):
    """Catch malformed or empty benchmark input binding a fresh output root."""
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(input_text, encoding="utf-8")
    manifest_path = tmp_path / "output/run_manifest.json"

    with pytest.raises((ValueError, json.JSONDecodeError)):
        manifest_main(
            [
                "ensure",
                "--manifest",
                str(manifest_path),
                "--model-config",
                str(MODEL_CONFIG),
                "--run-config",
                str(RUN_CONFIG),
                "--input",
                str(input_path),
                "--pipeline",
                "deepseek-putnam",
                "--chunks",
                "2",
            ]
        )

    assert not manifest_path.exists()


def test_manifest_cli_applies_putnam_constraints_before_writing(tmp_path):
    """Catch direct manifest creation bypassing Putnam's staged run contract."""
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(VALID_INPUT, encoding="utf-8")
    run_config = yaml.safe_load(RUN_CONFIG.read_text(encoding="utf-8"))
    run_config["generation"]["sample_schedule"] = [32]
    run_path = tmp_path / "run.yaml"
    run_path.write_text(yaml.safe_dump(run_config), encoding="utf-8")
    manifest_path = tmp_path / "output/run_manifest.json"

    with pytest.raises(ValueError, match="Putnam.*sample schedule"):
        manifest_main(
            [
                "ensure",
                "--manifest",
                str(manifest_path),
                "--model-config",
                str(MODEL_CONFIG),
                "--run-config",
                str(run_path),
                "--input",
                str(input_path),
                "--pipeline",
                "deepseek-putnam",
                "--chunks",
                "2",
            ]
        )

    assert not manifest_path.exists()


def test_standard_manifest_rejects_goedel_rc1_profile(tmp_path):
    """Catch a DeepSeek standard run silently reusing Goedel's rc1 toolchain."""
    run_config = yaml.safe_load(MINIF2F_RUN_CONFIG.read_text(encoding="utf-8"))
    run_config["lean"]["profile"] = "mathlib-v49"
    run_path = tmp_path / "run.yaml"
    run_path.write_text(yaml.safe_dump(run_config), encoding="utf-8")
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(VALID_INPUT, encoding="utf-8")

    with pytest.raises(ValueError, match="miniF2F requires deepseek-v49-rc2"):
        build_run_manifest(
            model_config_path=MODEL_CONFIG,
            run_config_path=run_path,
            input_path=input_path,
            pipeline="deepseek-standard",
            chunk_count=1,
        )


def test_standard_manifest_rejects_proofnet_under_fate_profile(tmp_path):
    """Catch ProofNet being rebound from its recorded rc2 environment to FATE."""
    proofnet_path = ROOT / "technical/configs/runs/deepseek/proofnet.yaml"
    run_config = yaml.safe_load(proofnet_path.read_text(encoding="utf-8"))
    run_config["lean"]["profile"] = "fate-v428"
    run_path = tmp_path / "run.yaml"
    run_path.write_text(yaml.safe_dump(run_config), encoding="utf-8")
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(VALID_INPUT, encoding="utf-8")

    with pytest.raises(ValueError, match="ProofNet requires deepseek-v49-rc2"):
        build_run_manifest(
            model_config_path=MODEL_CONFIG,
            run_config_path=run_path,
            input_path=input_path,
            pipeline="deepseek-standard",
            chunk_count=1,
        )


def test_standard_manifest_rejects_wrong_benchmark_assembly(tmp_path):
    """Catch a benchmark manifest binding the wrong source assembly policy."""
    proofnet_path = ROOT / "technical/configs/runs/deepseek/proofnet.yaml"
    run_config = yaml.safe_load(proofnet_path.read_text(encoding="utf-8"))
    run_config["assembly"]["mode"] = "standard"
    run_path = tmp_path / "run.yaml"
    run_path.write_text(yaml.safe_dump(run_config), encoding="utf-8")
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(VALID_INPUT, encoding="utf-8")

    with pytest.raises(ValueError, match="ProofNet requires proofnet assembly"):
        build_run_manifest(
            model_config_path=MODEL_CONFIG,
            run_config_path=run_path,
            input_path=input_path,
            pipeline="deepseek-standard",
            chunk_count=1,
        )


def test_conflicting_manifest_rejects_checkpoint_and_marker_reuse(tmp_path):
    """Catch stale outputs being silently rebound after public input changes."""
    input_path = tmp_path / "putnam.jsonl"
    input_path.write_text(VALID_INPUT, encoding="utf-8")
    manifest_path = tmp_path / "output/run_manifest.json"
    first = build_run_manifest(
        model_config_path=MODEL_CONFIG,
        run_config_path=RUN_CONFIG,
        input_path=input_path,
        pipeline="deepseek-putnam",
        chunk_count=2,
    )
    ensure_run_manifest(manifest_path, first)
    checkpoint = manifest_path.parent / "inference.jsonl"
    checkpoint.write_text('{"problem_id":"p_g0"}\n', encoding="utf-8")
    marker = manifest_path.parent / "COMPLETE"
    binding = {"kind": "standard"}
    write_completion_marker(marker, first, binding)
    marker_bytes = marker.read_bytes()

    input_path.write_text(
        '{"problem_id":"changed","lean4_code":"theorem changed : True := by sorry"}\n',
        encoding="utf-8",
    )
    conflicting = build_run_manifest(
        model_config_path=MODEL_CONFIG,
        run_config_path=RUN_CONFIG,
        input_path=input_path,
        pipeline="deepseek-putnam",
        chunk_count=2,
    )

    with pytest.raises(ManifestConflictError, match="run manifest conflict"):
        ensure_run_manifest(manifest_path, conflicting)

    assert checkpoint.read_text(encoding="utf-8") == '{"problem_id":"p_g0"}\n'
    assert marker.read_bytes() == marker_bytes
    validate_completion_marker(marker, first, binding)


def test_missing_manifest_rejects_preexisting_checkpoint_reuse(tmp_path):
    """Catch legacy/unbound checkpoints being adopted by a new run manifest."""
    input_path = tmp_path / "putnam.jsonl"
    input_path.write_text(VALID_INPUT, encoding="utf-8")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    checkpoint = output_dir / "inference.jsonl"
    checkpoint.write_text('{"problem_id":"p_g0"}\n', encoding="utf-8")
    expected = build_run_manifest(
        model_config_path=MODEL_CONFIG,
        run_config_path=RUN_CONFIG,
        input_path=input_path,
        pipeline="deepseek-putnam",
        chunk_count=2,
    )

    with pytest.raises(
        ManifestConflictError, match="existing artifacts require a run manifest"
    ):
        ensure_run_manifest(output_dir / "run_manifest.json", expected)

    assert checkpoint.is_file()
    assert not (output_dir / "run_manifest.json").exists()


def test_completion_marker_rejects_conflicting_artifact_binding(tmp_path):
    """Catch a bare/stale marker being reused for another chunk or target pass."""
    input_path = tmp_path / "putnam.jsonl"
    input_path.write_text(VALID_INPUT, encoding="utf-8")
    manifest = build_run_manifest(
        model_config_path=MODEL_CONFIG,
        run_config_path=RUN_CONFIG,
        input_path=input_path,
        pipeline="deepseek-putnam",
        chunk_count=2,
    )
    marker = tmp_path / "COMPLETE"
    write_completion_marker(
        marker,
        manifest,
        {"kind": "putnam-cumulative", "chunk_index": 0, "target_pass": 1},
    )

    with pytest.raises(ManifestConflictError, match="completion marker conflict"):
        validate_completion_marker(
            marker,
            manifest,
            {"kind": "putnam-cumulative", "chunk_index": 1, "target_pass": 1},
        )

    assert not list(tmp_path.glob(".*.tmp"))
