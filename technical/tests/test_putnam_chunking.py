"""Tests for Putnam JSONL chunking, merging, and progress summaries."""

import json
import threading
from pathlib import Path

from technical.src.benchmarks.putnam_chunking import (
    chunk_bounds,
    finalization_lock,
    finalize_putnam_target,
    is_global_progress_complete,
    merge_putnam_stages,
    merge_json_lists,
    split_jsonl,
    summarize_chunk_progress,
)
from technical.src.run_manifest import (
    build_run_manifest,
    ensure_run_manifest,
    write_completion_marker,
)


ROOT = Path(__file__).parents[2]
DEEPSEEK_PUTNAM_PIPELINE = (
    ROOT / "technical/pipelines/run_deepseek_putnam_chunked.sbatch"
)
MODEL_CONFIG = ROOT / "technical/configs/models/deepseek-prover-v2-7b.yaml"
RUN_CONFIG = ROOT / "technical/configs/runs/deepseek/putnam.yaml"


def _run_manifest(tmp_path, chunks=2):
    input_path = tmp_path / "putnam.jsonl"
    input_path.write_text(
        "".join(
            json.dumps(
                {
                    "problem_id": f"p{index}",
                    "lean4_code": f"theorem p{index} : True := by sorry",
                }
            )
            + "\n"
            for index in range(chunks)
        ),
        encoding="utf-8",
    )
    manifest = build_run_manifest(
        model_config_path=MODEL_CONFIG,
        run_config_path=RUN_CONFIG,
        input_path=input_path,
        pipeline="deepseek-putnam",
        chunk_count=chunks,
    )
    path = tmp_path / "output/run_manifest.json"
    ensure_run_manifest(path, manifest)
    return path, manifest


def _write_cumulative_chunk(output_dir, manifest, index, target_pass=1):
    directory = output_dir / f"chunks/chunk_{index}/cumulative_pass{target_pass}"
    directory.mkdir(parents=True)
    identifier = f"p{index}_g0"
    full = [
        {
            "problem_id": identifier,
            "origin_problem_id": f"p{index}",
            "id_maps": [
                {"origin_problem_id": f"p{index}"},
                {"generation_id": identifier},
            ],
            "finish_reason": "stop",
            "extraction_status": "success",
        }
    ]
    inference = [{"problem_id": identifier, "origin_problem_id": f"p{index}"}]
    verification = [
        {
            "name": identifier,
            "code": f"theorem p{index} : True := by trivial",
            "compilation_result": {"complete": index == 0},
        }
    ]
    (directory / "full_records.json").write_text(json.dumps(full), encoding="utf-8")
    (directory / "to_inference_codes.json").write_text(
        json.dumps(inference), encoding="utf-8"
    )
    (directory / "code_compilation_full_header.json").write_text(
        json.dumps(verification), encoding="utf-8"
    )
    write_completion_marker(
        directory / "COMPLETE",
        manifest,
        {
            "kind": "putnam-cumulative",
            "chunk_index": index,
            "target_pass": target_pass,
        },
    )
    return directory


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


def test_global_progress_complete_requires_no_missing_chunks():
    """Catch a partial Putnam invocation being promoted to global completion."""
    assert is_global_progress_complete({"missing_chunks": []}) is True
    assert is_global_progress_complete({"missing_chunks": [1]}) is False

    try:
        is_global_progress_complete({"complete_chunks": [0]})
    except ValueError as error:
        assert str(error) == "progress summary is missing missing_chunks"
    else:
        raise AssertionError("progress without missing_chunks was accepted")


def test_putnam_finalization_lock_serializes_concurrent_callers(tmp_path):
    """Catch concurrent Slurm callers entering one target finalizer together."""
    lock_path = tmp_path / "finalize.lock"
    started = threading.Event()
    entered = threading.Event()

    def contender():
        started.set()
        with finalization_lock(lock_path):
            entered.set()

    with finalization_lock(lock_path):
        thread = threading.Thread(target=contender)
        thread.start()
        assert started.wait(1)
        assert not entered.wait(0.05)
    assert entered.wait(1)
    thread.join(timeout=1)
    assert not thread.is_alive()


def test_putnam_finalizer_gates_completion_and_stale_partial_cannot_overwrite(
    tmp_path,
):
    """Catch a late partial scan replacing already completed global data."""
    manifest_path, manifest = _run_manifest(tmp_path, chunks=2)
    output_dir = manifest_path.parent
    chunk0 = _write_cumulative_chunk(output_dir, manifest, 0)

    partial = finalize_putnam_target(
        output_dir=output_dir,
        total_chunks=2,
        target_pass=1,
        run_manifest_path=manifest_path,
    )

    global_dir = output_dir / "cumulative_pass1"
    assert partial["status"] == "partial"
    assert partial["missing_chunks"] == [1]
    assert not (global_dir / "COMPLETE").exists()
    assert not (output_dir / "COMPLETE").exists()

    chunk1 = _write_cumulative_chunk(output_dir, manifest, 1)
    complete = finalize_putnam_target(
        output_dir=output_dir,
        total_chunks=2,
        target_pass=1,
        run_manifest_path=manifest_path,
    )
    published = (global_dir / "full_records.json").read_bytes()

    assert complete["status"] == "complete"
    assert complete["missing_chunks"] == []
    assert (global_dir / "COMPLETE").is_file()
    assert (output_dir / "COMPLETE").is_file()
    assert (global_dir / "summary/meta_summarize.json").is_file()
    assert len(json.loads(published)) == 2

    (chunk1 / "COMPLETE").unlink()
    (chunk0 / "full_records.json").write_text("[]", encoding="utf-8")
    stale = finalize_putnam_target(
        output_dir=output_dir,
        total_chunks=2,
        target_pass=1,
        run_manifest_path=manifest_path,
    )

    assert stale["status"] == "already_complete"
    assert (global_dir / "full_records.json").read_bytes() == published


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


def test_deepseek_putnam_pipeline_declares_exact_cumulative_stages():
    """Catch drift from the reviewed 1, 1+7, 1+7+8, 1+7+8+16 schedule."""
    assert DEEPSEEK_PUTNAM_PIPELINE.is_file()
    text = DEEPSEEK_PUTNAM_PIPELINE.read_text(encoding="utf-8")

    assert 'case "${TARGET_PASS}:${GENERATION_OFFSET}:${SAMPLES}" in' in text
    assert '1:0:1) STAGE_SPECS=("0:1") ;;' in text
    assert '8:1:7) STAGE_SPECS=("0:1" "1:7") ;;' in text
    assert '16:8:8) STAGE_SPECS=("0:1" "1:7" "8:8") ;;' in text
    assert '32:16:16) STAGE_SPECS=("0:1" "1:7" "8:8" "16:16") ;;' in text
    assert "unsupported checkpoint" in text

    assert "--adapter" in text and "deepseek" in text
    assert "putnam_chunking split" in text
    assert "putnam_chunking cumulative" in text
    assert "putnam_chunking finalize" in text
    assert "technical.src.run_manifest" in text
    assert "check-marker" in text
    assert "write-marker" in text
    assert 'touch "${CUMULATIVE_DIR}/COMPLETE"' not in text
    assert 'touch "${OUTPUT_DIR}/COMPLETE"' not in text
