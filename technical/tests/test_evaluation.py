"""Tests for Pass@k prefix selection and historical summary semantics."""

import json

from technical.src.evaluation.select_passk_prefix import (
    GenerationIdError,
    select_generation_prefix,
)
from technical.src.evaluation.summarize_passk import summarize, write_summary_outputs


def test_prefix_selection_keeps_only_generations_below_k():
    records = [{"name": f"p_g{index}"} for index in range(10)]

    selected = select_generation_prefix(records, 8)

    assert [record["name"] for record in selected] == [f"p_g{i}" for i in range(8)]


def test_prefix_selection_counts_invalid_generation_ids():
    records = [{"name": "p_g0"}, {"name": "missing_suffix"}, {"code": "x"}]

    try:
        select_generation_prefix(records, 1)
    except GenerationIdError as error:
        assert error.invalid_count == 2
        assert error.total_count == 3
    else:
        raise AssertionError("invalid generation IDs disappeared silently")


def _id_maps():
    return {
        "a_g0": [{"origin_problem_id": "a"}, {"generation_id": "a_g0"}],
        "a_g1": [{"origin_problem_id": "a"}, {"generation_id": "a_g1"}],
        "b_g0": [{"origin_problem_id": "b"}, {"generation_id": "b_g0"}],
        "b_g1": [{"origin_problem_id": "b"}, {"generation_id": "b_g1"}],
    }


def test_summary_marks_an_origin_solved_when_one_generation_is_correct():
    records = [
        {"name": "a_g0", "code": "by fail_if_success trivial", "compilation_result": {"complete": False}},
        {"name": "a_g1", "code": "by trivial", "compilation_result": {"complete": True}},
        {"name": "b_g0", "code": "by trivial", "compilation_result": {"complete": False}},
        {"name": "b_g1", "code": "by trivial", "compilation_result": {"complete": False}},
    ]

    result = summarize(records, _id_maps(), field="complete")

    origin = result["levels"]["origin_problem_id"]
    assert origin["problem_num"] == 2
    assert origin["solved_num"] == 1
    assert origin["solved_ratio"] == "50.00"
    assert origin["details"]["a"] == {"correct": 1, "count": 2}


def test_summary_can_use_pass_field_and_forbids_search_tactics():
    records = [
        {"name": "a_g0", "code": "by exact?", "compilation_result": {"pass": True}},
        {"name": "a_g1", "code": "by trivial", "compilation_result": {"pass": False}},
    ]
    maps = {key: value for key, value in _id_maps().items() if key.startswith("a_")}

    result = summarize(records, maps, field="pass")

    assert result["levels"]["origin_problem_id"]["solved_num"] == 0
    assert result["records"][0]["correct"] is False


def test_summary_rejects_empty_denominator():
    try:
        summarize([], {}, field="complete")
    except ValueError as error:
        assert str(error) == "cannot summarize an empty record set"
    else:
        raise AssertionError("empty denominator was accepted")


def test_summary_reports_generation_outcomes_separately_from_lean_results():
    """Catch length/extraction outcomes being collapsed into Lean rejection."""
    records = [
        {
            "name": "a_g0",
            "code": "None",
            "compilation_result": {"complete": False},
        },
        {
            "name": "a_g1",
            "code": "theorem a : True := by trivial",
            "compilation_result": {"complete": True},
        },
    ]
    maps = {key: value for key, value in _id_maps().items() if key.startswith("a_")}
    generation_records = [
        {
            "problem_id": "a_g0",
            "finish_reason": "length",
            "extraction_status": "blocked proof token",
        },
        {
            "problem_id": "a_g1",
            "finish_reason": "stop",
            "extraction_status": "success",
        },
    ]

    result = summarize(
        records,
        maps,
        field="complete",
        generation_records=generation_records,
    )

    assert result["generation_outcomes"] == {
        "total": 2,
        "finish_reason": {"length": 1, "stop": 1},
        "extraction_status": {"blocked proof token": 1, "success": 1},
    }
    assert result["levels"]["origin_problem_id"]["solved_num"] == 1


def test_summary_writer_preserves_legacy_meta_and_separates_generation_outcomes(
    tmp_path,
):
    """Catch DeepSeek metadata changing the historical Goedel meta array schema."""
    result = {
        "field": "complete",
        "levels": {
            "origin_problem_id": {
                "problem_num": 1,
                "solved_num": 1,
                "solved_ratio": "100.00",
                "details": {"a": {"correct": 1, "count": 2}},
            }
        },
        "records": [],
        "generation_outcomes": {
            "total": 2,
            "finish_reason": {"length": 1, "stop": 1},
            "extraction_status": {"blocked proof token": 1, "success": 1},
        },
    }

    write_summary_outputs(tmp_path, result)

    assert json.loads((tmp_path / "meta_summarize.json").read_text()) == [
        {
            "level": "origin_problem_id",
            "value": {
                "problem_num": 1,
                "solved_num": 1,
                "solved_ratio": "100.00",
            },
        }
    ]
    assert json.loads((tmp_path / "generation_outcomes.json").read_text()) == {
        "total": 2,
        "finish_reason": {"length": 1, "stop": 1},
        "extraction_status": {"blocked proof token": 1, "success": 1},
    }
