"""Tests for hash-bound reusable run artifacts."""

import json
from pathlib import Path

import pytest

from technical.src.run_manifest import (
    ManifestConflictError,
    build_run_manifest,
    canonical_sha256,
    ensure_run_manifest,
    validate_completion_marker,
    write_completion_marker,
)


ROOT = Path(__file__).parents[2]
MODEL_CONFIG = ROOT / "technical/configs/models/deepseek-prover-v2-7b.yaml"
RUN_CONFIG = ROOT / "technical/configs/runs/deepseek/putnam.yaml"


def test_canonical_hash_is_stable_across_mapping_order():
    """Catch manifests hashing serialization order instead of config semantics."""
    assert canonical_sha256({"b": 2, "a": [1, 3]}) == canonical_sha256(
        {"a": [1, 3], "b": 2}
    )


def test_run_manifest_binds_public_semantics_without_private_paths(tmp_path):
    """Catch reusable outputs missing prompt, config, input, or chunk identity."""
    input_path = tmp_path / "private-location/putnam.jsonl"
    input_path.parent.mkdir()
    input_path.write_text('{"problem_id":"p"}\n', encoding="utf-8")

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


def test_conflicting_manifest_rejects_checkpoint_and_marker_reuse(tmp_path):
    """Catch stale outputs being silently rebound after public input changes."""
    input_path = tmp_path / "putnam.jsonl"
    input_path.write_text('{"problem_id":"p"}\n', encoding="utf-8")
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

    input_path.write_text('{"problem_id":"changed"}\n', encoding="utf-8")
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
    input_path.write_text("{}\n", encoding="utf-8")
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
    input_path.write_text("{}\n", encoding="utf-8")
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
