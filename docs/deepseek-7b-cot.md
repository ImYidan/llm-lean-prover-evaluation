# DeepSeek-Prover-V2-7B CoT Operation

## Scope

This pipeline evaluates `deepseek-ai/DeepSeek-Prover-V2-7B` on miniF2F,
ProofNet, PutnamBench, FATE-M, and FATE-H. The public model configuration pins
the reviewed revision:

```text
a8d9e14432b2e8dd9df2a4d4e70f1ba9bc8d9b7b
```

Users supply the model through `--model-path`. Use a local model directory for
offline runs, or use a resolvable model ID with the pinned `revision` value
from `technical/configs/models/deepseek-prover-v2-7b.yaml`. The repository
does not ship weights, tokenizer files, benchmark data, checkpoints, or
results.

## Setup

Create the Python environment:

```bash
conda env create -f technical/environment.yml
conda activate goedel-prover-v2
```

Build a separate Lean v4.9.0-rc2 workspace for miniF2F, ProofNet, and
PutnamBench. The source revision is shared with the repository's Mathlib
submodule, but the toolchain is not: the submodule retains Goedel's rc1 pin.

```bash
git clone https://github.com/xinhjBrant/mathlib4.git <deepseek-workspace>
git -C <deepseek-workspace> checkout 2f65ba7f1a9144b20c8e7358513548e317d26de1
cp technical/lean/deepseek-v49-rc2/lean-toolchain <deepseek-workspace>/lean-toolchain
cd <deepseek-workspace>
lake build repl
```

The `deepseek-v49-rc2` profile pins Lean `v4.9.0-rc2`, Mathlib
`2f65ba7f1a9144b20c8e7358513548e317d26de1`, and the source manifest's REPL
revision `3334a97b268ecc67beb36a75787f7e831208a724`. Keep this workspace separate
from `technical/vendor/mathlib4`; no second Mathlib source tree is committed.

Build the FATE workspace from the pinned Lake profile:

```bash
cd technical/lean/fate-v428
lake update
lake exe cache get
lake build repl
```

The FATE profile pins Lean v4.28.0, Mathlib
`8f9d9cff6bd728b17a24e163c9402775d9e6a365`, and REPL
`527590ce2b9f3b5c4a9a1031e5b8fcfb909b9a4a`. Users build those dependencies
during setup. The source tree keeps `lean-toolchain`, `lakefile.toml`, and
`lake-manifest.json`; `.gitignore` excludes fetched dependency trees and build
products.

## Runtime Boundary

Run generated Lean verification in a credential-free account, container, or
batch job. Remove API tokens from the environment, disable network access, and
mount benchmark data and model weights read-only. Give the output directory a
disposable writable mount. Apply CPU, memory, process, and wall-clock limits at
the job or container layer. The Python runners enforce timeouts. They do not
sandbox generated Lean code.

Remote model code remains off unless `model.trust_remote_code: true` appears
in a reviewed model profile. Keep the immutable model revision in the profile
when loading by model ID.

## Profiles

| Run profile | Config | Samples | Model length | New tokens | Lean profile | Timeout | Assembly |
|---|---|---:|---:|---:|---|---:|---|
| miniF2F | `technical/configs/runs/deepseek/minif2f.yaml` | 32 | 32768 | 8192 | `deepseek-v49-rc2` | 300 s | standard |
| ProofNet | `technical/configs/runs/deepseek/proofnet.yaml` | 32 | 40960 | 32768 | `deepseek-v49-rc2` | 300 s | proofnet |
| PutnamBench | `technical/configs/runs/deepseek/putnam.yaml` | 1+7+8+16 | 40960 | 32768 | `deepseek-v49-rc2` | 300 s | standard |
| FATE-M | `technical/configs/runs/deepseek/fate-m.yaml` | 32 | 32768 | 8192 | `fate-v428` | 4000 s | standard |
| FATE-H | `technical/configs/runs/deepseek/fate-h.yaml` | 32 | 32768 | 8192 | `fate-v428` | 4000 s | standard |

All five profiles use temperature `1.0`, top-p `0.95`, seed `30`, bfloat16,
tensor parallel size `1`, and GPU memory utilization `0.90`.

## Standard Runs

Use `technical/pipelines/run_deepseek.sbatch` for miniF2F, ProofNet, FATE-M,
and FATE-H. The script reads sampling, verification, and assembly settings from
the model and run profiles.

Each run config names its benchmark explicitly. The wrapper validates that
benchmark's Lean profile, stage arguments, and Lean
workspace before it creates or reuses `run_manifest.json`. Its stable hashes
bind the model revision, effective
sampling, prompt contract, assembly mode, Lean and verification settings,
input bytes, and chunk count without recording local paths. A conflict, or an
existing artifact directory without a manifest, fails closed.

```bash
sbatch <site-options> technical/pipelines/run_deepseek.sbatch \
  technical/configs/models/deepseek-prover-v2-7b.yaml \
  technical/configs/runs/deepseek/minif2f.yaml \
  <benchmark.jsonl> \
  <output-dir> \
  <model-dir> \
  <lean-workspace> \
  <repl-executable>
```

Swap the run profile for ProofNet, FATE-M, or FATE-H. ProofNet selects
target-aware assembly through `assembly.mode: proofnet`; the script passes that
mode to generation. FATE runs use the `fate-v428` Lean profile.

## Putnam Runs

PutnamBench uses cumulative chunks. Submit stages in this order:

| TARGET_PASS | GENERATION_OFFSET | SAMPLES |
|---:|---:|---:|
| 1 | 0 | 1 |
| 8 | 1 | 7 |
| 16 | 8 | 8 |
| 32 | 16 | 16 |

```bash
sbatch <site-options> technical/pipelines/run_deepseek_putnam_chunked.sbatch \
  technical/configs/models/deepseek-prover-v2-7b.yaml \
  technical/configs/runs/deepseek/putnam.yaml \
  <putnam.jsonl> \
  <output-dir> \
  <model-dir> \
  <lean-workspace> \
  <repl-executable> \
  8 0 7 1 8
```

The final five arguments in the example request 8 chunks, chunk index 0,
7 samples, generation offset 1, and `TARGET_PASS=8`. That stage requires the
same chunk's completed Pass@1 stage.

The root `<output-dir>/COMPLETE` marker records the highest completed
`TARGET_PASS`: all manifest-bound chunks exist and the summarizer wrote that
target's shared summary. It does not imply Pass@32 unless its hash-bound JSON
payload records `TARGET_PASS=32`.

## Outputs

DeepSeek generation writes raw checkpoints before it rewrites normalized JSON:

- `inference.jsonl` stores one raw line per candidate, including prompt text,
  raw model output, assembled source status, `finish_reason`, and
  `stop_reason`.
- `full_records.json` stores normalized generation records.
- `to_inference_codes.json` stores the verification-facing normalized records.
- `proofnet_duplicate_problem_ids.json` appears for ProofNet runs when source
  IDs need stable disambiguation.
- `code_compilation_full_header.json` stores Lean verification results.
- `summary/origin_problem_id_summarize.csv`,
  `summary/generation_id_summarize.csv`, and `summary/meta_summarize.json`
  retain the historical Pass@k summary format.
- `summary/generation_outcomes.json` counts `finish_reason` and
  `extraction_status` separately from Lean results without changing the
  historical Goedel metadata schema.
- `run_manifest.json` binds reusable artifacts to stable public semantics.
- `COMPLETE` is an atomic, hash-bound JSON marker for a completed standard
  run.

Putnam chunked runs also write:

- `chunks/chunk_<n>/stage_offset_<offset>_add_<samples>/COMPLETE` after a
  stage passes generation and verification; the marker is manifest-bound.
- `chunks/chunk_<n>/cumulative_pass<TARGET_PASS>/COMPLETE` after the chunk
  cumulative files pass prefix validation; the marker also binds the chunk
  index and target.
- `progress_pass<TARGET_PASS>.json` at the root output directory.
- `cumulative_pass<TARGET_PASS>/summary/*` after every chunk for that
  `TARGET_PASS` has a completion marker.

Historical Goedel runs may contain `code_compilation_repl.json`; DeepSeek
wrappers write `code_compilation_full_header.json`.

## Resume Rules

`inference.jsonl` is append-only. On restart, the loader accepts complete
UTF-8 JSONL records and may discard one interrupted final line. If a crash
leaves a valid but incomplete terminal prompt group, the loader durably
truncates only that terminal group and regenerates it. It rejects
blank interior lines, malformed interior JSON, non-object records, duplicate
generation IDs, conflicting IDs, source mismatches, generation gaps, and a
candidate count outside the requested range.

Putnam stages resume only through manifest-bound marker directories. A later
stage validates prerequisite markers before reuse. The cumulative step
validates the exact generation prefix for each original problem. Finalization
uses a portable per-target file lock, rescans every chunk while holding it,
and publishes arrays, progress, summary, then the target marker. A stale
partial finalizer therefore cannot overwrite already completed global data.

## Verification Commands

Run the CPU-safe suite:

```bash
conda activate goedel-prover-v2
python -m pytest -q
python -m compileall -q technical/src technical/tests
bash -n technical/pipelines/*.sbatch
git diff --check
```

Run the v4.9 Lean smoke after building the caller-supplied workspace:

```bash
LEAN_TEST_WORKSPACE=<deepseek-v49-rc2-workspace> \
LEAN_TEST_REPL_COMMAND='lake env <repl-executable>' \
python -m pytest -q technical/tests/test_integration_lean.py
```

Run the optional FATE v4.28 smoke after the `fate-v428` workspace builds on
the current host:

```bash
LEAN_FATE_V428_REPL_SMOKE=1 \
python -m pytest -q technical/tests/test_integration_lean.py::test_fate_v428_repl_profile_accepts_trivial_full_header_proof
```

The public profile and manifest pins are valid without a checked-in build
cache. Do not treat a skipped optional smoke as a completed v4.28 runtime test.
