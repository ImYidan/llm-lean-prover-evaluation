from pathlib import Path


ROOT = Path(__file__).parents[2]
FORBIDDEN = (
    "/" + "home/",
    "/" + "scratch/",
    "Co-" + "authored-by:" + " " + "".join(("C", "o", "d", "e", "x")),
)
REQUIRED = (
    ROOT / ".gitmodules",
    ROOT / "LICENSE",
    ROOT / "NOTICE",
    ROOT / "technical/environment.yml",
    ROOT / "technical/lean-toolchain",
    ROOT / "technical/configs/models/goedel-prover-v2-32b.yaml",
)
TEXT_SUFFIXES = {".py", ".sh", ".sbatch", ".yaml", ".yml", ".md", ".txt"}
EXCLUDED_DIRECTORIES = {".git", ".superpowers", "vendor"}
PIPELINES = (
    ROOT / "technical/pipelines/run_standard.sbatch",
    ROOT / "technical/pipelines/run_proofnet_incremental.sbatch",
    ROOT / "technical/pipelines/run_putnam_chunked.sbatch",
)


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
    return paths


def find_public_text_violations(root: Path) -> list[tuple[str, str]]:
    """Return forbidden marker occurrences in public text files below ``root``."""
    violations = []
    for path in public_text_paths(root):
        text = path.read_text(encoding="utf-8")
        for forbidden in FORBIDDEN:
            if forbidden in text:
                violations.append((path.relative_to(root).as_posix(), forbidden))
    return violations


def test_extensionless_technical_file_is_scanned_for_forbidden_markers(
    tmp_path: Path,
):
    """Catch a scanner that only considers suffix-based technical files."""
    lean_toolchain = tmp_path / "technical/lean-toolchain"
    lean_toolchain.parent.mkdir()
    marker = "/" + "home/"
    lean_toolchain.write_text(marker, encoding="utf-8")

    violations = find_public_text_violations(tmp_path)

    assert ("technical/lean-toolchain", marker) in violations


def test_public_text_files_have_no_private_paths_or_ai_attribution():
    """Catch private machine details or prohibited AI attribution."""
    assert all(path.is_file() for path in REQUIRED)

    assert find_public_text_violations(ROOT) == []


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
    }
    for path in PIPELINES:
        text = path.read_text(encoding="utf-8")
        assert "set -euo pipefail" in text
        assert "#SBATCH --account" not in text
        assert "#SBATCH --partition" not in text
        assert "inference_putnam" not in text
        for module in expected_modules[path.name]:
            assert module in text
        for variable in ("INPUT_PATH", "OUTPUT_DIR", "MODEL_PATH", "WORKSPACE"):
            assert f'"${{{variable}}}"' in text
    putnam = PIPELINES[2].read_text(encoding="utf-8")
    assert "--generation-offset" in putnam
    assert "putnam_chunking merge" in putnam
    assert "cumulative_pass" in putnam
    assert '[[ -f "${candidate_marker}" ]]' in putnam


def test_runtime_dependencies_and_pipeline_configs_are_effective():
    environment = (ROOT / "technical/environment.yml").read_text(encoding="utf-8")
    assert "pexpect==4.9.0" in environment

    standard, proofnet, putnam = (path.read_text(encoding="utf-8") for path in PIPELINES)
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
    for pipeline in PIPELINES:
        assert f"pipelines/{pipeline.name}" in text
    for supplied_input in ("--model-path", "--input", "--workspace"):
        assert supplied_input in text
    for private_history in ("cleanup history", "recovery script", "server experiment"):
        assert private_history not in text.lower()
