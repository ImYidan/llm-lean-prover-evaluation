# DeepSeek-Prover-V2-7B CoT Integration Design

## Goal

Add DeepSeek-Prover-V2-7B CoT to the existing public evaluation repository.
The integration covers miniF2F, ProofNet, PutnamBench, FATE-M, and FATE-H.
It preserves the DeepSeek sampling and checkpoint semantics while reusing the
repository's proof assembly, Lean verification, chunking, and Pass@k modules.

## Release boundary

The repository publishes code and pinned configuration only. Users supply
benchmark JSONL files, model weights, and writable output paths. The model
configuration records:

```text
model_id: deepseek-ai/DeepSeek-Prover-V2-7B
revision: a8d9e14432b2e8dd9df2a4d4e70f1ba9bc8d9b7b
```

The release excludes weights, tokenizer snapshots, datasets, result arrays,
logs, caches, local Lean builds, machine paths, upload utilities, and unrelated
model scripts. In particular, no `*.safetensors`, model download directory, or
Hugging Face cache belongs in Git.

## Architecture

The generation seam has two adapters and shared record machinery:

```text
benchmark JSONL
    -> benchmark adapter
    -> model adapter and sampling policy
    -> raw checkpoint
    -> normalized public records
    -> shared Lean verifier
    -> shared Pass@k summary
```

Goedel keeps its current one-candidate request behavior and JSON-array outputs.
DeepSeek keeps grouped `n` sampling per prompt and a resumable JSONL checkpoint.
Both adapters use the same generation IDs, candidate representation, proof
assembly rules, normalized verification records, and evaluation code.

### Shared candidate

A candidate contains:

```text
text: str
finish_reason: str | None
stop_reason: str | int | None
```

Goedel leaves the optional reasons empty. DeepSeek fills them from vLLM. This
keeps backend metadata out of benchmark and verifier modules.

### DeepSeek adapter

The DeepSeek adapter owns only model-specific behavior:

- render the detailed-proof-plan chat prompt;
- require a local model path and reviewed remote-code implementation;
- request multiple candidates for each prompt;
- preserve candidate termination metadata;
- append resumable `inference.jsonl` records;
- normalize those records for the common verifier.

The adapter uses the shared standard or ProofNet assembler selected by the run
profile. It does not implement Lean verification or Pass@k.

## Outputs and recovery

DeepSeek writes `inference.jsonl` as the raw durable checkpoint. Each line has:

- origin, generation, and source IDs;
- locked benchmark source;
- rendered prompt and message history;
- raw model output;
- assembled source or extraction error;
- `finish_reason` and `stop_reason`.

The runner also produces the repository's normalized `full_records.json` and
`to_inference_codes.json`. Existing verification and summary modules consume
only the normalized files.

On restart, the checkpoint loader validates every complete JSONL line. It may
discard one incomplete final line caused by an interrupted append. It rejects
malformed interior lines, duplicate generation IDs, conflicting records,
negative offsets, gaps, and counts that exceed the requested sampling range.
Normalized JSON and manifests use temporary files plus atomic replacement.
`COMPLETE` is written only after every required artifact passes validation.

## Run profiles

Model and benchmark configuration remain separate. A run profile binds them to
the historical sampling, verification, timeout, and Lean settings.

| Profile | Samples | Model length | New tokens | Verifier | Lean profile | Timeout |
|---|---:|---:|---:|---|---|---:|
| miniF2F | 32 | 32768 | 8192 | full-header REPL | v4.9 | 300 s |
| ProofNet | 32 | 40960 | 32768 | full-header REPL | v4.9 | 300 s |
| PutnamBench | 1+7+8+16 | 40960 | 32768 | full-header REPL | v4.9 | 300 s |
| FATE-M | 32 | 32768 | 8192 | full-header REPL | v4.28 | 4000 s |
| FATE-H | 32 | 32768 | 8192 | full-header REPL | v4.28 | 4000 s |

All profiles use temperature 1.0, top-p 0.95, seed 30, bfloat16, tensor
parallel size 1, and GPU memory utilization 0.90. The loader rejects
`max_new_tokens >= max_model_len`.

ProofNet uses target-aware assembly and post-sanitization unique IDs.
PutnamBench accepts only cumulative generation prefixes `g0`, `g0..g7`,
`g0..g15`, and `g0..g31`.

## Lean profiles

miniF2F, ProofNet, and PutnamBench use the existing v4.9 Mathlib gitlink at:

```text
2f65ba7f1a9144b20c8e7358513548e317d26de1
```

FATE-M and FATE-H use a small Lake project under `technical/lean/fate-v428/`:

```text
Lean:    v4.28.0
Mathlib: 8f9d9cff6bd728b17a24e163c9402775d9e6a365
REPL:    527590ce2b9f3b5c4a9a1031e5b8fcfb909b9a4a
```

The project contains `lean-toolchain`, `lakefile.toml`, and the resolved
`lake-manifest.json`. It downloads dependencies during setup; it does not add a
second vendored source tree.

## Files

The public change is limited to:

```text
technical/configs/models/deepseek-prover-v2-7b.yaml
technical/configs/runs/deepseek/*.yaml
technical/lean/fate-v428/*
technical/src/generation/candidates.py
technical/src/generation/checkpoints.py
technical/src/generation/adapters/deepseek.py
technical/src/generation/generate.py
technical/pipelines/run_deepseek.sbatch
technical/pipelines/run_deepseek_putnam_chunked.sbatch
technical/tests/*
technical/docs/deepseek.md
README.md
NOTICE
```

Exact implementation filenames may be combined when a smaller interface keeps
the same behavior. No old DeepSeek verifier, summarizer, splitter, merger,
installer, downloader, uploader, submit wrapper, smoke wrapper, or recovery
script is copied.

## Failure semantics

- Missing fenced Lean code and blocked proof terms become explicit extraction
  failures and do not start Lean.
- `finish_reason=length` remains visible and is counted separately from Lean
  rejection.
- Lean errors, remaining sorries, timeouts, process failures, and protocol
  errors retain distinct fields.
- `apply?`, `exact?`, `sorry`, and `admit` never count as a successful proof.
- A backend sample-count mismatch or incomplete generation prefix stops the
  stage without a completion marker.

Generated Lean executes with the verifier process's permissions. Documentation
requires a credential-free, network-restricted runtime with resource limits.
Remote model code is allowed only for the reviewed local model snapshot pinned
by the public revision metadata.

## Tests

CPU tests use fake tokenizers and generation backends. They cover:

- grouped sampling and offsets;
- preservation of finish and stop reasons;
- JSONL resume, duplicate rejection, and final-line recovery;
- normalized record compatibility;
- DeepSeek prompt and extraction policy;
- ProofNet ID and target handling;
- exact Putnam cumulative prefixes;
- all five run profiles and both Lean profiles;
- unchanged Goedel behavior;
- rejection of private paths, credentials, weights, caches, and large files.

Optional integration tests compile small proofs with both configured Lean
profiles. Release verification runs the full test suite, Python 3.10 compile,
all CLI help paths, Slurm syntax checks, staged-diff checks, and public-tree
security scans.

## Release process

Development occurs on `deepseek-7b-cot`; the experiment directory remains
read-only. An independent review checks the final diff against this design.
No push occurs until tests and release scans pass and the repository owner
authorizes integration. Commit author and committer use only the repository
owner's configured identity, with no additional attribution trailers.

## Acceptance criteria

The integration is complete when a caller can select any of the five DeepSeek
run profiles, resume interrupted grouped generation without duplicate IDs,
retain termination metadata, verify through the correct Lean profile, and
produce the same normalized Pass@k outputs as the existing repository without
installing model weights or benchmark data from Git.
