#!/usr/bin/env python

import argparse
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any



PROMPT_TEMPLATE = """
Complete the following Lean 4 code:

```lean4
{}
```

Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.
The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof.
""".strip()

FORBIDDEN_PROOF_TERMS = ("apply?", "exact?", "sorry", "admit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--generation-offset", type=int, default=0)
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path, split: str, limit: int | None) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if split != "none" and str(item.get("split")) != split:
                continue
            records.append(item)
            if limit is not None and len(records) >= limit:
                break
    return records


def select_shard(
    items: list[dict[str, Any]], num_shards: int, shard_index: int
) -> list[tuple[int, dict[str, Any]]]:
    return [
        (source_index, item)
        for source_index, item in enumerate(items)
        if source_index % num_shards == shard_index
    ]


def formal_statement(lean4_code: str) -> str:
    if ":= by" not in lean4_code:
        raise ValueError("Lean statement does not contain ':= by'")
    return lean4_code.split(":= by", 1)[0] + ":= by sorry"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_prompt(lean4_code: str, tokenizer: Any) -> tuple[str, list[dict[str, str]]]:
    messages = [{"role": "user", "content": PROMPT_TEMPLATE.format(formal_statement(lean4_code))}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt, messages


def extract_lean_block(text: str) -> str | None:
    matches = list(
        re.finditer(r"```(?:lean4|lean)\s*\n(.*?)```", text, flags=re.DOTALL)
    )
    return matches[-1].group(1).strip() if matches else None


def combine_with_original_statement(lean4_code: str, model_output: str) -> tuple[str, str | None]:
    extracted = extract_lean_block(model_output)
    if extracted is None:
        return "", "No fenced Lean code block found"

    lowered = extracted.lower()
    forbidden = [term for term in FORBIDDEN_PROOF_TERMS if term in lowered]
    if forbidden:
        return "", f"Forbidden proof term(s): {', '.join(forbidden)}"

    theorem_match = re.search(r"\btheorem\b.*?:=\s*by", extracted, flags=re.DOTALL)
    if theorem_match is None:
        return "", "No complete theorem ending in ':= by' found"

    target_prefix = lean4_code.split(":= by", 1)[0] + ":= by"
    proof_body = extracted[theorem_match.end() :]
    return target_prefix + proof_body, None


def existing_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    ids = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                ids.add(json.loads(line)["problem_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return ids


def batched(items: list[Any], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def main() -> None:
    args = parse_args()
    if args.max_new_tokens >= args.max_model_len:
        raise ValueError("--max-new-tokens must be smaller than --max-model-len")
    if args.samples < 1:
        raise ValueError("--samples must be positive")
    if args.num_shards < 1:
        raise ValueError("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")
    if args.generation_offset < 0:
        raise ValueError("--generation-offset must be non-negative")
    if args.batch_size < 1 or args.max_num_seqs < 1:
        raise ValueError("--batch-size and --max-num-seqs must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "inference.jsonl"
    manifest_path = output_dir / "inference_manifest.json"
    input_path = Path(args.input_path)
    model_path = Path(args.model_path)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    all_source_items = load_jsonl(input_path, args.split, args.limit)
    prompts = []
    seen_origin_ids = set()
    for source_index, item in select_shard(
        all_source_items, args.num_shards, args.shard_index
    ):
        origin_id = item.get("problem_id", item.get("name"))
        if not origin_id:
            raise ValueError(f"Missing problem_id/name at source index {source_index}")
        if origin_id in seen_origin_ids:
            raise ValueError(f"Duplicate problem id in shard: {origin_id}")
        seen_origin_ids.add(origin_id)
        prompt, messages = build_prompt(item["lean4_code"], tokenizer)
        prompts.append(
            {
                "item": item,
                "origin_id": origin_id,
                "source_index": source_index,
                "prompt": prompt,
                "messages": messages,
            }
        )

    completed = existing_ids(records_path)
    pending = [
        entry
        for entry in prompts
        if not all(
            f"{entry['origin_id']}_g{i}" in completed
            for i in range(
                args.generation_offset, args.generation_offset + args.samples
            )
        )
    ]

    prompt_hash = hashlib.sha256(PROMPT_TEMPLATE.encode("utf-8")).hexdigest()
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "arguments": vars(args),
        "input_path": str(input_path.resolve()),
        "input_sha256": sha256_file(input_path),
        "model_path": str(model_path.resolve()),
        "total_problem_count": len(all_source_items),
        "shard_problem_count": len(prompts),
        "pending_problem_count": len(pending),
        "prompt_sha256": prompt_hash,
        "output": str(records_path.resolve()),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if not pending:
        print("All requested generations are already present.")
        manifest["status"] = "complete"
        manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=str(model_path),
        tokenizer=str(model_path),
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        seed=args.seed,
        enable_prefix_caching=True,
        enforce_eager=args.enforce_eager,
        download_dir=str(Path(args.output_dir).parent / "vllm_download"),
    )
    sampling = SamplingParams(
        n=args.samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    started = time.time()
    with records_path.open("a", encoding="utf-8", buffering=1) as writer:
        for batch_index, batch in enumerate(batched(pending, args.batch_size), start=1):
            batch_prompts = [entry["prompt"] for entry in batch]
            outputs = llm.generate(batch_prompts, sampling)
            if len(outputs) != len(batch):
                raise RuntimeError(
                    f"vLLM returned {len(outputs)} outputs for a batch of {len(batch)}"
                )
            for entry, request_output in zip(batch, outputs):
                if len(request_output.outputs) != args.samples:
                    raise RuntimeError(
                        f"{entry['origin_id']}: expected {args.samples} generations, "
                        f"received {len(request_output.outputs)}"
                    )
                for sample_index, candidate in enumerate(request_output.outputs):
                    generation_id = args.generation_offset + sample_index
                    problem_id = f"{entry['origin_id']}_g{generation_id}"
                    if problem_id in completed:
                        continue
                    full_code, extraction_error = combine_with_original_statement(
                        entry["item"]["lean4_code"],
                        candidate.text,
                    )
                    record = {
                        "problem_id": problem_id,
                        "origin_problem_id": entry["origin_id"],
                        "generation_id": generation_id,
                        "source_index": entry["source_index"],
                        "lean4_code": entry["item"]["lean4_code"],
                        "messages": entry["messages"],
                        "model_input": entry["prompt"],
                        "model_output": candidate.text,
                        "full_code": full_code,
                        "extraction_error": extraction_error,
                        "finish_reason": candidate.finish_reason,
                        "stop_reason": candidate.stop_reason,
                    }
                    writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                    completed.add(problem_id)
            print(
                f"batch={batch_index} completed_generations={len(completed)} "
                f"elapsed_seconds={time.time() - started:.1f}",
                flush=True,
            )

    expected_ids = {
        f"{entry['origin_id']}_g{sample_index}"
        for entry in prompts
        for sample_index in range(
            args.generation_offset, args.generation_offset + args.samples
        )
    }
    final_ids = existing_ids(records_path)
    missing_ids = sorted(expected_ids - final_ids)
    if missing_ids:
        raise RuntimeError(
            f"Inference output is incomplete; missing {len(missing_ids)} generations. "
            f"First missing id: {missing_ids[0]}"
        )

    manifest["status"] = "complete"
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["completed_generation_count"] = len(expected_ids)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
