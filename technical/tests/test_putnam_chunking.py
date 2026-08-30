"""Tests for Putnam JSONL chunking, merging, and progress summaries."""

import json

from technical.src.benchmarks.putnam_chunking import (
    chunk_bounds,
    merge_putnam_stages,
    merge_json_lists,
    split_jsonl,
    summarize_chunk_progress,
)


def test_chunk_bounds_cover_even_and_uneven_inputs_without_overlap():
    assert [chunk_bounds(8, 4, index) for index in range(4)] == [
        (0, 2),
        (2, 4),
        (4, 6),
        (6, 8),
    ]
    bounds = [chunk_bounds(10, 3, index) for index in range(3)]
    assert bounds == [(0, 4), (4, 7), (7, 10)]
    assert [item for start, end in bounds for item in range(start, end)] == list(range(10))


def test_chunk_bounds_allow_more_chunks_than_rows_and_reject_bad_index():
    assert [chunk_bounds(2, 4, index) for index in range(4)] == [
        (0, 1),
        (1, 2),
        (2, 2),
        (2, 2),
    ]
    try:
        chunk_bounds(2, 4, 4)
    except ValueError as error:
        assert str(error) == "chunk index must satisfy 0 <= index < chunks"
    else:
        raise AssertionError("invalid chunk index was accepted")


def test_split_jsonl_writes_only_the_selected_chunk(tmp_path):
    source = tmp_path / "input.jsonl"
    source.write_text("".join(json.dumps({"i": i}) + "\n" for i in range(5)))
    output = tmp_path / "chunk.jsonl"

    selected = split_jsonl(source, output, chunks=2, index=1)

    assert selected == [{"i": 3}, {"i": 4}]
    assert [json.loads(line) for line in output.read_text().splitlines()] == selected


def test_merge_preserves_order_and_has_explicit_duplicate_policy(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps([{"problem_id": "a"}, {"problem_id": "b"}]))
    second.write_text(json.dumps([{"problem_id": "b"}, {"problem_id": "c"}]))

    assert [row["problem_id"] for row in merge_json_lists([first, second])] == [
        "a",
        "b",
        "b",
        "c",
    ]
    assert [
        row["problem_id"]
        for row in merge_json_lists(
            [first, second], dedupe_key="problem_id", on_duplicate="first"
        )
    ] == ["a", "b", "c"]
    try:
        merge_json_lists(
            [first, second], dedupe_key="problem_id", on_duplicate="error"
        )
    except ValueError as error:
        assert "duplicate problem_id: b" in str(error)
    else:
        raise AssertionError("duplicate merge key was accepted")


def test_progress_summary_reports_terminal_solved_and_missing_chunks(tmp_path):
    chunk0 = tmp_path / "chunk0.json"
    chunk2 = tmp_path / "chunk2.json"
    chunk0.write_text(
        json.dumps(
            [
                {"name": "a_g0", "compilation_result": {"complete": True}},
                {"name": "b_g0", "compilation_result": {"complete": False}},
            ]
        )
    )
    chunk2.write_text(
        json.dumps(
            [{"name": "c_g0", "compilation_result": {"complete": False}}]
        )
    )

    summary = summarize_chunk_progress(
        {0: chunk0, 2: chunk2}, total_chunks=3, target_pass=8
    )

    assert summary == {
        "target_pass": 8,
        "complete_chunks": [0, 2],
        "missing_chunks": [1],
        "terminal_attempts": 3,
        "problem_num": 3,
        "solved_num": 1,
    }


def test_cumulative_stage_merge_requires_exact_generation_prefix(tmp_path):
    stage0 = tmp_path / "stage0"
    stage1 = tmp_path / "stage1"
    stage0.mkdir()
    stage1.mkdir()
    for directory, generation in ((stage0, 0), (stage1, 1)):
        identifier = f"p_g{generation}"
        full = [{"problem_id": identifier, "origin_problem_id": "p"}]
        inference = [{"problem_id": identifier, "origin_problem_id": "p"}]
        verification = [
            {"name": identifier, "compilation_result": {"complete": False}}
        ]
        (directory / "full.json").write_text(json.dumps(full))
        (directory / "inference.json").write_text(json.dumps(inference))
        (directory / "verification.json").write_text(json.dumps(verification))

    output = tmp_path / "cumulative"
    result = merge_putnam_stages(
        [stage0 / "full.json", stage1 / "full.json"],
        [stage0 / "inference.json", stage1 / "inference.json"],
        [stage0 / "verification.json", stage1 / "verification.json"],
        target_pass=2,
        output_dir=output,
    )

    assert result == {"problem_num": 1, "attempts": 2, "target_pass": 2}
    assert [row["problem_id"] for row in json.loads((output / "full_records.json").read_text())] == [
        "p_g0",
        "p_g1",
    ]


def test_cumulative_stage_merge_rejects_generation_gaps(tmp_path):
    paths = []
    for kind, key in (("full", "problem_id"), ("inference", "problem_id"), ("verification", "name")):
        path = tmp_path / f"{kind}.json"
        path.write_text(json.dumps([{key: "p_g1", "origin_problem_id": "p"}]))
        paths.append(path)

    try:
        merge_putnam_stages(
            [paths[0]], [paths[1]], [paths[2]], target_pass=1, output_dir=tmp_path / "out"
        )
    except ValueError as error:
        assert "expected generations" in str(error)
    else:
        raise AssertionError("generation gap was accepted")
