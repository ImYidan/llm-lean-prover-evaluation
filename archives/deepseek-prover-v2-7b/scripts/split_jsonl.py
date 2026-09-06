#!/usr/bin/env python

import argparse
import math
from pathlib import Path


def chunk_bounds(total: int, chunks: int, index: int) -> tuple[int, int]:
    base = total // chunks
    extra = total % chunks
    start = index * base + min(index, extra)
    end = start + base + (1 if index < extra else 0)
    return start, end


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a JSONL file into deterministic contiguous chunks."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunks", type=int, required=True)
    parser.add_argument("--prefix", default="chunk")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.chunks <= 0:
        raise ValueError("--chunks must be positive")
    if not args.input.is_file():
        raise FileNotFoundError(args.input)

    lines = args.input.read_text(encoding="utf-8").splitlines()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    width = max(2, len(str(args.chunks - 1)))
    manifest = ["chunk\tstart\tend\tcount\tpath\n"]

    for index in range(args.chunks):
        start, end = chunk_bounds(len(lines), args.chunks, index)
        name = f"{args.prefix}_{index:0{width}d}"
        output_path = args.output_dir / f"{name}.jsonl"
        chunk_lines = lines[start:end]
        output_path.write_text(
            "".join(f"{line}\n" for line in chunk_lines),
            encoding="utf-8",
        )
        manifest.append(
            f"{name}\t{start}\t{end}\t{len(chunk_lines)}\t{output_path}\n"
        )

    (args.output_dir / "manifest.tsv").write_text("".join(manifest), encoding="utf-8")
    ceil_size = math.ceil(len(lines) / args.chunks) if args.chunks else 0
    print(
        f"Split {len(lines)} rows into {args.chunks} chunks "
        f"(ceil size {ceil_size}) at {args.output_dir}"
    )


if __name__ == "__main__":
    main()
