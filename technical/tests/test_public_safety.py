import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).parents[2]
FORBIDDEN = (
    "Co-" + "authored-by:" + " " + "".join(("C", "o", "d", "e", "x")),
)
REQUIRED = (
    ROOT / "README.md",
    ROOT / ".gitmodules",
    ROOT / "LICENSE",
    ROOT / "NOTICE",
    ROOT / "docs/deepseek-7b-cot.md",
    ROOT / "technical/environment.yml",
    ROOT / "technical/lean-toolchain",
    ROOT / "technical/lean/fate-v428/lean-toolchain",
    ROOT / "technical/lean/fate-v428/lakefile.toml",
    ROOT / "technical/lean/fate-v428/lake-manifest.json",
    ROOT / "technical/configs/models/goedel-prover-v2-32b.yaml",
    ROOT / "technical/pipelines/run_deepseek.sbatch",
    ROOT / "technical/pipelines/run_deepseek_putnam_chunked.sbatch",
)
TEXT_SUFFIXES = {".py", ".sh", ".sbatch", ".yaml", ".yml", ".md", ".txt"}
GENERATED_OUTPUT_NAMES = {
    "COMPLETE",
    "code_compilation_full_header.json",
    "code_compilation_repl.json",
    "full_records.json",
    "inference.jsonl",
    "meta_summarize.json",
    "origin_problem_id_summarize.csv",
    "generation_id_summarize.csv",
    "progress.json",
    "proofnet_duplicate_problem_ids.json",
    "summary.csv",
    "summary.json",
    "to_inference_codes.json",
}
MODEL_ARTIFACT_SUFFIXES = (
    ".bin",
    ".gguf",
    ".pt",
    ".pth",
    ".safetensors",
)
MODEL_ARTIFACT_NAMES = {
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
}
AI_ATTRIBUTION_NAMES = (
    "".join(("C", "o", "d", "e", "x")),
    "".join(("C", "h", "a", "t", "G", "P", "T")),
    "".join(("C", "l", "a", "u", "d", "e")),
    "".join(("O", "p", "e", "n", "A", "I")),
)
REQUIRED_IGNORE_PATTERNS = {
    "**/COMPLETE",
    "**/code_compilation_full_header.json",
    "**/code_compilation_repl.json",
    "**/full_records.json",
    "**/generation_id_summarize.csv",
    "**/inference.jsonl",
    "**/meta_summarize.json",
    "**/origin_problem_id_summarize.csv",
    "**/progress.json",
    "**/progress_pass*.json",
    "**/proofnet_duplicate_problem_ids.json",
    "**/summary.csv",
    "**/summary.json",
    "**/to_inference_codes.json",
    "**/*.bin",
    "**/*.gguf",
    "**/*.pt",
    "**/*.pth",
    "**/*.safetensors",
    "**/special_tokens_map.json",
    "**/tokenizer.json",
    "**/tokenizer.model",
    "**/tokenizer_config.json",
    "**/.cache/",
    "/checkpoints/",
    "/data/",
    "/datasets/",
    "/logs/",
    "/models/",
    "/outputs/",
    "/results/",
    "/runs/",
    "/weights/",
    "*.err",
    "*.log",
    "slurm-*.out",
    "technical/vendor/**/.lake/",
}
BROAD_SOURCE_IGNORE_PATTERNS = {
    "*.json",
    "*.jsonl",
    "*.lean",
    "*.md",
    "*.py",
    "*.sbatch",
    "*.sh",
    "*.toml",
    "*.yaml",
    "*.yml",
    "docs/",
    "technical/",
    "technical/src/",
    "technical/tests/",
}
EXCLUDED_DIRECTORIES = {
    ".elan",
    ".git",
    ".lake",
    ".pytest_cache",
    ".superpowers",
    "__pycache__",
    "vendor",
}
ARTIFACT_EXCLUDED_DIRECTORIES = {
    ".git",
    ".pytest_cache",
    ".superpowers",
    "__pycache__",
    "vendor",
}
FORBIDDEN_ARTIFACT_DIRECTORIES = {".lake", ".elan"}
FORBIDDEN_TOP_LEVEL_ARTIFACT_DIRECTORIES = {
    ".superpowers",
    "checkpoints",
    "data",
    "datasets",
    "logs",
    "models",
    "outputs",
    "results",
    "runs",
    "weights",
}
FORBIDDEN_COMPILED_SUFFIXES = (
    ".olean",
    ".olean.trace",
    ".ilean",
    ".o",
    ".so",
    ".dylib",
    ".dll",
)
FATE_PROFILE = ROOT / "technical/lean/fate-v428"
MATHLIB_428_REV = "8f9d9cff6bd728b17a24e163c9402775d9e6a365"
MATHLIB_428_URL = "https://github.com/leanprover-community/mathlib4.git"
REPL_428_REV = "527590ce2b9f3b5c4a9a1031e5b8fcfb909b9a4a"
REPL_428_URL = "https://github.com/leanprover-community/repl.git"
FATE_VENDOR_DIRECTORIES = {
    "aesop",
    "batteries",
    "Cli",
    "importGraph",
    "mathlib",
    "mathlib4",
    "proofwidgets",
    "Qq",
    "quote4",
    "REPL",
    "repl",
}
GOEDEL_PIPELINES = (
    ROOT / "technical/pipelines/run_standard.sbatch",
    ROOT / "technical/pipelines/run_proofnet_incremental.sbatch",
    ROOT / "technical/pipelines/run_putnam_chunked.sbatch",
)
DEEPSEEK_PIPELINES = (
    ROOT / "technical/pipelines/run_deepseek.sbatch",
    ROOT / "technical/pipelines/run_deepseek_putnam_chunked.sbatch",
)
PIPELINES = GOEDEL_PIPELINES + DEEPSEEK_PIPELINES
TEXT_VIOLATION_PATTERNS = (
    ("private_path", re.compile(r"/(?:home|scratch)/[A-Za-z0-9._~/-]+")),
    (
        "private_key",
        re.compile(
            r"BEGIN (?:RSA|OPENSSH|EC|DSA) PRIVATE KEY",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "credential_assignment",
        re.compile(
            r"\b(?:api[_-]?key|access[_-]?token|secret[_-]?key|hf[_-]?token)"
            r"\s*[:=]\s*[\"']?[A-Za-z0-9_./+=-]{12,}",
            flags=re.IGNORECASE,
        ),
    ),
    ("secret_token", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    (
        "attribution_trailer",
        re.compile("Co-" + "authored-by:", flags=re.IGNORECASE),
    ),
    (
        "generated_attribution",
        re.compile(
            r"(?:generated|written|built)\s+by\s+"
            + rf"(?:{'|'.join(re.escape(name) for name in AI_ATTRIBUTION_NAMES)})",
            flags=re.IGNORECASE,
        ),
    ),
)


def _last_effective_line(text: str) -> str:
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            lines.append(stripped)
    return lines[-1]


def _line_number(text: str, needle: str) -> int:
    for index, line in enumerate(text.splitlines(), start=1):
        if needle in line:
            return index
    raise AssertionError(f"{needle!r} not found")


def _required_paths(root: Path) -> tuple[Path, ...]:
    return (
        root / ".gitmodules",
        root / "LICENSE",
        root / "NOTICE",
        root / "technical/environment.yml",
        root / "technical/lean-toolchain",
        root / "technical/configs/models/goedel-prover-v2-32b.yaml",
    )


def _is_excluded(path: Path, root: Path) -> bool:
    return bool(EXCLUDED_DIRECTORIES.intersection(path.relative_to(root).parts))


def _is_technical_text_file(path: Path) -> bool:
    return path.suffix in TEXT_SUFFIXES or path.suffix == ""


def _clean_gitignore_patterns(path: Path) -> set[str]:
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def public_text_paths(root: Path) -> set[Path]:
    """Return public text files while excluding repository metadata directories."""
    paths = {path for path in _required_paths(root) if path.is_file()}
    paths.update(path for path in root.glob("README*") if path.is_file())
    paths.update(
        path for path in (root / ".gitignore", root / "pytest.ini") if path.is_file()
    )

    technical_root = root / "technical"
    if technical_root.is_dir():
        paths.update(
            path
            for path in technical_root.rglob("*")
            if path.is_file()
            and not _is_excluded(path, root)
            and _is_technical_text_file(path)
        )
    docs_root = root / "docs"
    if docs_root.is_dir():
        paths.update(
            path
            for path in docs_root.rglob("*")
            if path.is_file()
            and not _is_excluded(path, root)
            and _is_technical_text_file(path)
        )
    return paths


def find_public_text_violations(root: Path) -> list[tuple[str, str]]:
    """Return forbidden marker occurrences in public text files below ``root``."""
    violations = []
    for path in public_text_paths(root):
        text = path.read_text(encoding="utf-8")
        for forbidden in FORBIDDEN:
            if forbidden in text:
                violations.append((path.relative_to(root).as_posix(), forbidden))
        for label, pattern in TEXT_VIOLATION_PATTERNS:
            if pattern.search(text):
                violations.append((path.relative_to(root).as_posix(), label))
    return violations


def _is_forbidden_generated_artifact(path: Path) -> bool:
    return (
        path.name in GENERATED_OUTPUT_NAMES
        or (path.name.startswith("progress_pass") and path.suffix == ".json")
        or path.name in MODEL_ARTIFACT_NAMES
        or path.name.endswith(MODEL_ARTIFACT_SUFFIXES)
    )


def find_forbidden_public_artifacts(root: Path) -> list[str]:
    """Return generated Lean artifacts or vendored dependencies in public paths."""
    violations = []
    tracked_files = _git_tracked_files(root)
    paths = (
        (root / tracked_file for tracked_file in tracked_files)
        if tracked_files is not None
        else root.rglob("*")
    )
    for path in paths:
        relative = path.relative_to(root)
        parts = relative.parts
        if ARTIFACT_EXCLUDED_DIRECTORIES.intersection(parts):
            continue
        if parts and parts[0] in FORBIDDEN_TOP_LEVEL_ARTIFACT_DIRECTORIES:
            violations.append(relative.as_posix())
            continue
        if FORBIDDEN_ARTIFACT_DIRECTORIES.intersection(parts):
            violations.append(relative.as_posix())
            continue
        if _is_forbidden_generated_artifact(path):
            violations.append(relative.as_posix())
            continue
        if path.is_file() and path.name.endswith(FORBIDDEN_COMPILED_SUFFIXES):
            violations.append(relative.as_posix())
            continue
        if (
            len(parts) > 3
            and parts[:3] == ("technical", "lean", "fate-v428")
            and parts[3] in FATE_VENDOR_DIRECTORIES
        ):
            violations.append(relative.as_posix())
    return sorted(violations)


def _git_tracked_files(root: Path) -> list[Path] | None:
    git_dir = root / ".git"
    if not git_dir.exists():
        return None
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return [Path(line) for line in result.stdout.splitlines() if line]


def _tracked_public_violations(root: Path) -> list[str]:
    tracked_files = _git_tracked_files(root)
    if tracked_files is None:
        return []

    violations = []
    for tracked_file in tracked_files:
        parts = tracked_file.parts
        path = root / tracked_file
        if parts and parts[0] in FORBIDDEN_TOP_LEVEL_ARTIFACT_DIRECTORIES:
            violations.append(tracked_file.as_posix())
        elif ".cache" in parts:
            violations.append(tracked_file.as_posix())
        elif _is_forbidden_generated_artifact(path):
            violations.append(tracked_file.as_posix())
    return violations


def test_release_gitignore_covers_generated_outputs_without_hiding_source_files():
    """Catch a release ignore file that permits outputs or masks source."""
    patterns = _clean_gitignore_patterns(ROOT / ".gitignore")

    assert REQUIRED_IGNORE_PATTERNS <= patterns
    assert BROAD_SOURCE_IGNORE_PATTERNS.isdisjoint(patterns)


def test_deepseek_operation_docs_cover_public_release_contract():
    """Catch a public guide that omits a required release contract."""
    text = (ROOT / "docs/deepseek-7b-cot.md").read_text(encoding="utf-8")

    required_terms = (
        "--model-path",
        "deepseek-ai/DeepSeek-Prover-V2-7B",
        "a8d9e14432b2e8dd9df2a4d4e70f1ba9bc8d9b7b",
        "miniF2F",
        "ProofNet",
        "PutnamBench",
        "FATE-M",
        "FATE-H",
        "inference.jsonl",
        "full_records.json",
        "to_inference_codes.json",
        "code_compilation_repl.json",
        "code_compilation_full_header.json",
        "meta_summarize.json",
        "mathlib-v49",
        "fate-v428",
        "Lean v4.28.0",
        "LEAN_TEST_WORKSPACE",
        "LEAN_TEST_REPL_COMMAND",
        "network",
        "credential",
        "COMPLETE",
        "TARGET_PASS",
        "Pass@32",
    )
    for term in required_terms:
        assert term in text


def test_fate_v428_profile_pins_direct_github_sources():
    """Catch a FATE profile that loses direct upstream source pins."""
    lakefile = (FATE_PROFILE / "lakefile.toml").read_text(encoding="utf-8")
    assert MATHLIB_428_URL in lakefile
    assert REPL_428_URL in lakefile
    assert MATHLIB_428_REV in lakefile
    assert REPL_428_REV in lakefile

    manifest = json.loads(
        (FATE_PROFILE / "lake-manifest.json").read_text(encoding="utf-8")
    )
    packages = {package["name"].lower(): package for package in manifest["packages"]}
    assert packages["mathlib"]["url"] == MATHLIB_428_URL
    assert packages["mathlib"]["rev"] == MATHLIB_428_REV
    assert packages["repl"]["url"] == REPL_428_URL
    assert packages["repl"]["rev"] == REPL_428_REV


def test_extensionless_technical_file_is_scanned_for_forbidden_markers(
    tmp_path: Path,
):
    """Catch a scanner that only considers suffix-based technical files."""
    lean_toolchain = tmp_path / "technical/lean-toolchain"
    lean_toolchain.parent.mkdir()
    marker = "/" + "home/private"
    lean_toolchain.write_text(marker, encoding="utf-8")

    violations = find_public_text_violations(tmp_path)

    assert ("technical/lean-toolchain", "private_path") in violations


def test_public_text_files_have_no_private_paths_or_ai_attribution():
    """Catch private machine details or prohibited AI attribution."""
    assert all(path.is_file() for path in REQUIRED)

    assert find_public_text_violations(ROOT) == []


def test_tracked_files_exclude_internal_outputs_weights_and_caches():
    """Catch private planning files and generated artifacts in Git."""
    assert _tracked_public_violations(ROOT) == []


def test_public_tree_omits_generated_lean_artifacts_and_fate_dependency_trees(
    tmp_path: Path,
):
    """Catch accidentally publishing Lake caches or vendored FATE dependencies."""
    (tmp_path / "technical/lean/fate-v428/.lake/packages/mathlib").mkdir(
        parents=True
    )
    (tmp_path / "technical/lean/fate-v428/.lake/packages/mathlib/Mathlib.olean").write_text(
        "",
        encoding="utf-8",
    )
    (tmp_path / "technical/lean/fate-v428/.elan").mkdir()
    (tmp_path / "technical/lean/fate-v428/repl").mkdir()
    (tmp_path / "technical/lean/fate-v428/build").mkdir()
    (tmp_path / "technical/lean/fate-v428/build/Foo.o").write_text(
        "",
        encoding="utf-8",
    )
    (tmp_path / "technical/vendor/mathlib4/.lake/build/Mathlib.olean").mkdir(
        parents=True
    )
    (tmp_path / "runs/deepseek").mkdir(parents=True)
    (tmp_path / "runs/deepseek/inference.jsonl").write_text(
        "{}\n",
        encoding="utf-8",
    )
    (tmp_path / "runs/deepseek/full_records.json").write_text(
        "[]",
        encoding="utf-8",
    )
    (tmp_path / "runs/deepseek/progress_pass8.json").write_text(
        "{}",
        encoding="utf-8",
    )
    (tmp_path / "models/deepseek").mkdir(parents=True)
    (tmp_path / "models/deepseek/model.safetensors").write_text(
        "",
        encoding="utf-8",
    )
    (tmp_path / "models/deepseek/tokenizer.json").write_text(
        "{}",
        encoding="utf-8",
    )
    (tmp_path / "data/private_benchmark.jsonl").parent.mkdir(parents=True)
    (tmp_path / "data/private_benchmark.jsonl").write_text(
        "{}\n",
        encoding="utf-8",
    )

    violations = find_forbidden_public_artifacts(tmp_path)

    assert "technical/lean/fate-v428/.lake" in violations
    assert "technical/lean/fate-v428/.elan" in violations
    assert "technical/lean/fate-v428/repl" in violations
    assert "technical/lean/fate-v428/build/Foo.o" in violations
    assert "runs/deepseek/inference.jsonl" in violations
    assert "runs/deepseek/full_records.json" in violations
    assert "runs/deepseek/progress_pass8.json" in violations
    assert "models/deepseek/model.safetensors" in violations
    assert "models/deepseek/tokenizer.json" in violations
    assert "data/private_benchmark.jsonl" in violations
    assert "technical/vendor/mathlib4/.lake" not in violations


def test_public_tree_has_no_generated_lean_artifacts_or_fate_dependency_trees():
    assert find_forbidden_public_artifacts(ROOT) == []


def test_slurm_pipelines_are_portable_and_fail_fast():
    assert all(path.is_file() for path in PIPELINES)
    expected_modules = {
        "run_standard.sbatch": (
            "technical.src.generation.generate",
            "technical.src.verification.verify_standard_repl",
            "technical.src.evaluation.summarize_passk",
        ),
        "run_proofnet_incremental.sbatch": (
            "technical.src.benchmarks.run_proofnet_incremental",
        ),
        "run_putnam_chunked.sbatch": (
            "technical.src.benchmarks.putnam_chunking",
            "technical.src.generation.generate",
            "technical.src.verification.verify_full_header_repl",
        ),
        "run_deepseek.sbatch": (
            "technical.src.generation.generate",
            "technical.src.verification.verify_full_header_repl",
            "technical.src.evaluation.summarize_passk",
        ),
        "run_deepseek_putnam_chunked.sbatch": (
            "technical.src.benchmarks.putnam_chunking",
            "technical.src.generation.generate",
            "technical.src.verification.verify_full_header_repl",
            "technical.src.evaluation.summarize_passk",
        ),
    }
    for path in PIPELINES:
        text = path.read_text(encoding="utf-8")
        assert "set -euo pipefail" in text
        assert "#SBATCH --" not in text
        assert "inference_putnam" not in text
        for module in expected_modules[path.name]:
            assert module in text
        for variable in ("INPUT_PATH", "OUTPUT_DIR", "MODEL_PATH", "WORKSPACE"):
            assert f'"${{{variable}}}"' in text
    putnam = (ROOT / "technical/pipelines/run_putnam_chunked.sbatch").read_text(
        encoding="utf-8"
    )
    assert "--generation-offset" in putnam
    assert "putnam_chunking merge" in putnam
    assert "cumulative_pass" in putnam
    assert '[[ -f "${candidate_marker}" ]]' in putnam


def test_deepseek_pipelines_use_profiles_adapter_and_shared_verifier():
    standard = DEEPSEEK_PIPELINES[0].read_text(encoding="utf-8")
    putnam = DEEPSEEK_PIPELINES[1].read_text(encoding="utf-8")

    for text in (standard, putnam):
        assert "--adapter" in text and "deepseek" in text
        assert "--model-path" in text and '"${MODEL_PATH}"' in text
        assert "--max-tokens" in text and '"${MAX_TOKENS}"' in text
        assert "--dtype" in text and '"${DTYPE}"' in text
        assert "--gpu-memory-utilization" in text
        assert '"${GPU_MEMORY_UTILIZATION}"' in text
        assert "technical.src.verification.verify_full_header_repl" in text
        assert "verify_standard_repl" not in text
        assert "verify_full_header_file" not in text
        assert "sample_schedule" in text
        assert "max_model_len" in text
        assert "max_tokens" in text
        assert "dtype" in text
        assert "gpu_memory_utilization" in text
        assert 'run["assembly"]["mode"]' in text
        assert "Path(sys.argv[2]).stem" not in text
        assert "--assembly-mode" in text
        assert "rm -f" in text and "COMPLETE" in text

    assert "RUN_CONFIG" in standard
    assert _last_effective_line(standard) == 'touch "${OUTPUT_DIR}/COMPLETE"'

    assert 'run["assembly"]["mode"] != "standard"' in putnam
    assert 'if [[ "${GLOBAL_COMPLETE}" == "true" ]]; then' in putnam
    assert _line_number(putnam, "putnam_chunking progress") < _line_number(
        putnam, "putnam_chunking is-complete"
    )
    assert _line_number(putnam, "putnam_chunking is-complete") < _line_number(
        putnam, "technical.src.evaluation.summarize_passk"
    )
    assert _line_number(putnam, "technical.src.evaluation.summarize_passk") < _line_number(
        putnam, 'touch "${OUTPUT_DIR}/COMPLETE"'
    )
    assert _last_effective_line(putnam) != 'touch "${OUTPUT_DIR}/COMPLETE"'


def test_slurm_scripts_contain_no_private_runtime_paths_or_weights():
    forbidden = (
        "/" + "home/",
        "/" + "scratch/",
        "~/",
        ".cache/" + "huggingface",
        "HF" + "_HOME",
        "HF" + "_HUB_CACHE",
        "HUGGINGFACE" + "_HUB_CACHE",
        "TRANSFORMERS" + "_CACHE",
        "HF" + "_TOKEN",
        "HUGGINGFACE" + "_TOKEN",
        "api" + "_key",
        "access" + "_token",
        "tokenizer" + ".json",
        ".safe" + "tensors",
        "pytorch_model" + ".bin",
    )
    violations = []
    for path in (ROOT / "technical/pipelines").glob("*.sbatch"):
        text = path.read_text(encoding="utf-8")
        for marker in forbidden:
            if marker in text:
                violations.append((path.name, marker))

    assert violations == []


def test_runtime_dependencies_and_pipeline_configs_are_effective():
    environment = (ROOT / "technical/environment.yml").read_text(encoding="utf-8")
    assert "pexpect==4.9.0" in environment

    standard, proofnet, putnam = (
        path.read_text(encoding="utf-8") for path in GOEDEL_PIPELINES
    )
    assert "standard_repl" in standard and "--proof-timeout" in standard
    assert "full_header_file" in proofnet and "BENCHMARK_CONFIG" in proofnet
    assert "full_header_repl" in putnam and "BENCHMARK_CONFIG" in putnam
    for pipeline in (standard, proofnet, putnam):
        assert "tensor_parallel_size" in pipeline
        assert "trust_remote_code" in pipeline


def test_technical_readme_documents_the_public_data_flow_and_inputs():
    readme = ROOT / "technical/README.md"
    assert readme.is_file()
    text = readme.read_text(encoding="utf-8")
    for stage in ("Generation", "Extraction", "Verification", "Summary"):
        assert stage in text
    for pipeline in GOEDEL_PIPELINES:
        assert f"pipelines/{pipeline.name}" in text
    for supplied_input in ("--model-path", "--input", "--workspace"):
        assert supplied_input in text
    for private_history in ("cleanup history", "recovery script", "server experiment"):
        assert private_history not in text.lower()
