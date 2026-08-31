"""Stable hash bindings for reusable evaluation artifacts."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from technical.src.config import load_config, validate_model_run_config
from technical.src.generation.adapters.deepseek import deepseek_prompt_contract


class ManifestConflictError(ValueError):
    """Raised when an artifact belongs to different public run semantics."""


def canonical_sha256(value: object) -> str:
    """Hash canonical UTF-8 JSON independent of mapping insertion order."""
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    """Fsync and atomically replace one text artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _with_digest(payload: dict, digest_field: str) -> dict:
    result = dict(payload)
    result[digest_field] = canonical_sha256(payload)
    return result


def build_run_manifest(
    *,
    model_config_path: Path,
    run_config_path: Path,
    input_path: Path,
    pipeline: str,
    chunk_count: int,
) -> dict:
    """Build a path-free manifest from canonical public run semantics."""
    if pipeline not in {"deepseek-standard", "deepseek-putnam"}:
        raise ValueError(f"unsupported pipeline: {pipeline}")
    if (
        not isinstance(chunk_count, int)
        or isinstance(chunk_count, bool)
        or chunk_count < 1
    ):
        raise ValueError("chunk_count must be a positive integer")
    model_config = load_config(Path(model_config_path))
    run_config = load_config(Path(run_config_path))
    validate_model_run_config(model_config, run_config)
    payload = {
        "schema_version": 1,
        "pipeline": pipeline,
        "model": model_config["model"],
        "sampling": {
            **model_config["generation"],
            **run_config["generation"],
        },
        "prompt": {
            "adapter": "deepseek",
            "contract_sha256": canonical_sha256(deepseek_prompt_contract()),
        },
        "assembly": run_config["assembly"],
        "lean": run_config["lean"],
        "verification": run_config["verification"],
        "input_sha256": file_sha256(input_path),
        "chunk_count": chunk_count,
    }
    return _with_digest(payload, "manifest_sha256")


def _validate_self_digest(value: object, digest_field: str, label: str) -> dict:
    if not isinstance(value, dict):
        raise ManifestConflictError(f"{label} must be a JSON object")
    digest = value.get(digest_field)
    payload = {key: item for key, item in value.items() if key != digest_field}
    if not isinstance(digest, str) or digest != canonical_sha256(payload):
        raise ManifestConflictError(f"{label} digest mismatch")
    return value


def load_run_manifest(path: Path) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ManifestConflictError("run manifest is missing or malformed") from error
    return _validate_self_digest(value, "manifest_sha256", "run manifest")


@contextmanager
def _manifest_lock(path: Path) -> Iterator[None]:
    lock_path = path.with_name(f".{path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def ensure_run_manifest(path: Path, expected: dict) -> dict:
    """Atomically create or exactly validate one run manifest."""
    expected = _validate_self_digest(
        expected, "manifest_sha256", "expected run manifest"
    )
    path = Path(path)
    with _manifest_lock(path):
        if path.exists():
            actual = load_run_manifest(path)
            if actual != expected:
                raise ManifestConflictError("run manifest conflict")
            return actual
        lock_name = f".{path.name}.lock"
        existing_artifacts = [
            child for child in path.parent.iterdir() if child.name != lock_name
        ]
        if existing_artifacts:
            raise ManifestConflictError(
                "existing artifacts require a run manifest"
            )
        atomic_write_text(
            path,
            json.dumps(expected, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        )
    return expected


def _completion_record(manifest: dict, binding: dict) -> dict:
    manifest = _validate_self_digest(
        manifest, "manifest_sha256", "run manifest"
    )
    payload = {
        "schema_version": 1,
        "run_manifest_sha256": manifest["manifest_sha256"],
        "artifact": binding,
    }
    return _with_digest(payload, "marker_sha256")


def write_completion_marker(path: Path, manifest: dict, binding: dict) -> None:
    record = _completion_record(manifest, binding)
    atomic_write_text(
        path,
        json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    )


def load_completion_marker(path: Path) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ManifestConflictError("completion marker is missing or malformed") from error
    return _validate_self_digest(value, "marker_sha256", "completion marker")


def validate_completion_marker(path: Path, manifest: dict, binding: dict) -> dict:
    actual = load_completion_marker(path)
    if actual != _completion_record(manifest, binding):
        raise ManifestConflictError("completion marker conflict")
    return actual


def _parse_bindings(values: list[str]) -> dict:
    binding = {}
    for value in values:
        key, separator, raw = value.partition("=")
        if not separator or not key:
            raise ValueError("bindings must use KEY=VALUE")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = raw
        binding[key] = parsed
    return binding


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    ensure = subparsers.add_parser("ensure")
    ensure.add_argument("--manifest", type=Path, required=True)
    ensure.add_argument("--model-config", type=Path, required=True)
    ensure.add_argument("--run-config", type=Path, required=True)
    ensure.add_argument("--input", type=Path, required=True)
    ensure.add_argument(
        "--pipeline",
        choices=("deepseek-standard", "deepseek-putnam"),
        required=True,
    )
    ensure.add_argument("--chunks", type=int, required=True)
    for name in ("write-marker", "check-marker"):
        marker = subparsers.add_parser(name)
        marker.add_argument("--manifest", type=Path, required=True)
        marker.add_argument("--marker", type=Path, required=True)
        marker.add_argument("--binding", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "ensure":
        expected = build_run_manifest(
            model_config_path=args.model_config,
            run_config_path=args.run_config,
            input_path=args.input,
            pipeline=args.pipeline,
            chunk_count=args.chunks,
        )
        manifest = ensure_run_manifest(args.manifest, expected)
        print(manifest["manifest_sha256"])
        return 0
    manifest = load_run_manifest(args.manifest)
    binding = _parse_bindings(args.binding)
    if args.command == "write-marker":
        write_completion_marker(args.marker, manifest, binding)
    else:
        validate_completion_marker(args.marker, manifest, binding)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
