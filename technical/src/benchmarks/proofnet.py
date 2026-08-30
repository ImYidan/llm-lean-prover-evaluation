"""ProofNet row normalization and target-aware full-header assembly."""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict

from technical.src.generation.proof_extraction import (
    ProofAssemblyError,
    extract_last_lean_block,
)
from technical.src.generation.adapters.goedel import build_prompt_from_statement


DECLARATION_HEADER = re.compile(
    r"(?m)^\s*(?:noncomputable\s+)?(?:theorem|lemma|def)\s+"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_'.]*)\b"
)
PLACEHOLDER = re.compile(r":=\s*by\s*(?P<placeholder>sorry)\b")
PROOF_START = re.compile(r":=\s*by\b")


class ProofNetPromptAdapter:
    """Keep the full ProofNet header and target in the Goedel prompt."""

    def build_prompt(self, lean4_code: str, tokenizer) -> tuple[str, list[dict]]:
        return build_prompt_from_statement(lean4_code, tokenizer)


def _mask_comments(source: str) -> str:
    """Replace nested Lean comments with spaces while preserving offsets."""
    characters = list(source)
    index = 0
    block_depth = 0
    line_comment = False
    while index < len(source):
        pair = source[index : index + 2]
        if line_comment:
            if source[index] == "\n":
                line_comment = False
            else:
                characters[index] = " "
            index += 1
            continue
        if block_depth:
            if pair == "/-":
                characters[index : index + 2] = [" ", " "]
                block_depth += 1
                index += 2
            elif pair == "-/":
                characters[index : index + 2] = [" ", " "]
                block_depth -= 1
                index += 2
            else:
                if source[index] != "\n":
                    characters[index] = " "
                index += 1
            continue
        if pair == "--":
            characters[index : index + 2] = [" ", " "]
            line_comment = True
            index += 2
        elif pair == "/-":
            characters[index : index + 2] = [" ", " "]
            block_depth = 1
            index += 2
        else:
            index += 1
    return "".join(characters)


def _target_name_and_placeholder(source: str) -> tuple[str, re.Match]:
    masked = _mask_comments(source)
    placeholders = list(PLACEHOLDER.finditer(masked))
    if not placeholders:
        raise ProofAssemblyError("missing ProofNet target placeholder")
    placeholder = placeholders[-1]
    declarations = [
        match for match in DECLARATION_HEADER.finditer(masked) if match.start() < placeholder.start()
    ]
    if not declarations:
        raise ProofAssemblyError("missing ProofNet target declaration")
    return declarations[-1].group("name"), placeholder


def _generated_target_proof(source: str, target_name: str) -> int:
    masked = _mask_comments(source)
    declarations = list(DECLARATION_HEADER.finditer(masked))
    for index, declaration in enumerate(declarations):
        if declaration.group("name") != target_name:
            continue
        end = declarations[index + 1].start() if index + 1 < len(declarations) else len(masked)
        proof_start = PROOF_START.search(masked, declaration.end(), end)
        if proof_start is not None:
            return proof_start.end()
    raise ProofAssemblyError("missing generated ProofNet target body")


def assemble_proofnet_submission(statement: str, generated_code: str) -> str:
    """Keep the benchmark header and replace only its named target proof body."""
    forbidden = re.search(r"\b(apply\?|exact\?)", generated_code)
    if forbidden:
        raise ProofAssemblyError("forbidden tactic", forbidden.group(1))
    target_name, placeholder = _target_name_and_placeholder(statement)
    generated_body_start = _generated_target_proof(generated_code, target_name)
    locked_prefix = statement[: placeholder.start("placeholder")].rstrip()
    return locked_prefix + generated_code[generated_body_start:]


def assemble_proofnet_model_output(statement: str, model_output: str) -> str | None:
    """Extract the final Lean fence before applying target-aware assembly."""
    generated_code = extract_last_lean_block(model_output)
    if generated_code is None:
        return None
    return assemble_proofnet_submission(statement, generated_code)


def _ensure_lean4_code(row: dict) -> str:
    code = row.get("lean4_code")
    if isinstance(code, str) and code.strip():
        return code
    header = str(row.get("header") or "")
    informal = str(row.get("informal_prefix") or "")
    statement = str(row.get("formal_statement") or "").rstrip()
    if not statement:
        return ""
    if statement.endswith(":= by"):
        statement += " sorry"
    elif statement.endswith(":="):
        statement += " by sorry"
    elif not PLACEHOLDER.search(_mask_comments(statement)):
        statement += " := by sorry"
    return header + informal + statement


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "problem"


def prepare_unique_rows(rows: list[dict]) -> tuple[list[dict], dict]:
    """Create stable unique IDs while retaining duplicate provenance."""
    source_ids = [
        str(row.get("problem_id") or row.get("name") or f"row_{index}")
        for index, row in enumerate(rows)
    ]
    counts = Counter(source_ids)
    safe_counts = Counter(_safe_id(source_id) for source_id in source_ids)
    seen: Counter[str] = Counter()
    duplicate_report = defaultdict(list)
    prepared = []
    for index, (row, source_id) in enumerate(zip(rows, source_ids)):
        seen[source_id] += 1
        code = _ensure_lean4_code(row)
        digest = hashlib.sha1(code.encode("utf-8")).hexdigest()[:10]
        safe_id = _safe_id(source_id)
        unique_id = (
            f"{safe_id}__row{index:03d}__{digest}"
            if safe_counts[safe_id] > 1
            else safe_id
        )
        prepared_row = dict(row)
        prepared_row.update(
            {
                "source_problem_id": source_id,
                "source_problem_occurrence": seen[source_id] - 1,
                "proofnet_row_index": index,
                "proofnet_code_sha1": digest,
                "lean4_code": code,
                "problem_id": unique_id,
                "origin_problem_id": unique_id,
            }
        )
        prepared.append(prepared_row)
        if safe_counts[safe_id] > 1:
            duplicate_report[source_id].append(
                {
                    "row_index": index,
                    "source_problem_id": source_id,
                    "unique_problem_id": unique_id,
                    "code_sha1": digest,
                }
            )
    report = {
        "input_rows": len(rows),
        "unique_source_problem_ids": len(counts),
        "duplicate_source_problem_ids": dict(duplicate_report),
        "output_rows": len(prepared),
        "unique_output_problem_ids": len({row["problem_id"] for row in prepared}),
    }
    return prepared, report
