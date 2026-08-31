"""Tests for append-only generation checkpoints."""

import json

import pytest

from technical.src.generation.checkpoints import (
    CheckpointError,
    append_checkpoint,
    index_checkpoint,
    load_checkpoint,
)


def test_missing_checkpoint_loads_as_no_records(tmp_path):
    """Catch a loader that treats a fresh run as a corrupt checkpoint."""
    assert load_checkpoint(tmp_path / "inference.jsonl") == []


def test_append_writes_valid_checkpoint_rows(tmp_path):
    """Catch an append that omits or corrupts a completed record."""
    path = tmp_path / "inference.jsonl"

    append_checkpoint(path, {"problem_id": "p_g0", "model_output": "by trivial"})
    append_checkpoint(path, {"problem_id": "p_g1", "model_output": "by rfl"})

    assert load_checkpoint(path) == [
        {"problem_id": "p_g0", "model_output": "by trivial"},
        {"problem_id": "p_g1", "model_output": "by rfl"},
    ]
    assert [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ] == [
        {"problem_id": "p_g0", "model_output": "by trivial"},
        {"problem_id": "p_g1", "model_output": "by rfl"},
    ]


def test_torn_final_line_is_discarded(tmp_path):
    """Catch recovery that rejects the one partial write interruption can leave."""
    path = tmp_path / "inference.jsonl"
    path.write_text('{"problem_id":"p_g0"}\n{"problem', encoding="utf-8")

    assert load_checkpoint(path) == [{"problem_id": "p_g0"}]


def test_append_replaces_a_torn_final_line(tmp_path):
    """Catch an append that turns a recoverable partial write into corruption."""
    path = tmp_path / "inference.jsonl"
    path.write_text('{"problem_id":"p_g0"}\n{"problem', encoding="utf-8")

    append_checkpoint(path, {"problem_id": "p_g1"})

    assert load_checkpoint(path) == [
        {"problem_id": "p_g0"},
        {"problem_id": "p_g1"},
    ]


def test_malformed_interior_line_is_rejected(tmp_path):
    """Catch recovery that silently skips a completed corrupt checkpoint row."""
    path = tmp_path / "inference.jsonl"
    path.write_text(
        '{"problem_id":"p_g0"}\n{bad}\n{"problem_id":"p_g1"}\n',
        encoding="utf-8",
    )

    with pytest.raises(CheckpointError, match="line 2"):
        load_checkpoint(path)


def test_blank_interior_line_is_rejected(tmp_path):
    """Catch a loader that silently skips a complete blank checkpoint row."""
    path = tmp_path / "inference.jsonl"
    path.write_text(
        '{"problem_id":"p_g0"}\n\n{"problem_id":"p_g1"}\n',
        encoding="utf-8",
    )

    with pytest.raises(CheckpointError, match="line 2"):
        load_checkpoint(path)


def test_index_rejects_missing_or_non_string_problem_ids():
    """Catch indexing records without stable string generation identifiers."""
    with pytest.raises(CheckpointError, match="record 1"):
        index_checkpoint([{}])
    with pytest.raises(CheckpointError, match="record 1"):
        index_checkpoint([{"problem_id": 1}])


def test_index_rejects_duplicate_and_conflicting_problem_ids():
    """Catch resume ambiguity from repeated generation IDs."""
    record = {"problem_id": "p_g0", "model_output": "by trivial"}

    with pytest.raises(CheckpointError, match="duplicate problem_id: p_g0"):
        index_checkpoint([record, dict(record)])
    with pytest.raises(CheckpointError, match="conflicting problem_id: p_g0"):
        index_checkpoint(
            [record, {"problem_id": "p_g0", "model_output": "by rfl"}]
        )
