# Technical evaluation pipeline

The pipeline turns benchmark statements into checked Lean candidates through
four stages:

1. **Generation** builds the Goedel proof-planning prompt, samples the model,
   and saves the rendered prompt and raw model response.
2. **Extraction** selects the final fenced Lean block and combines its proof
   body with the benchmark statement. ProofNet uses a target-aware adapter for
   files that contain helper declarations.
3. **Verification** sends the assembled source to a persistent Lean REPL or
   compiles a temporary `.lean` file, depending on the benchmark configuration.
4. **Summary** maps generation IDs back to problem IDs and reports whether any
   candidate solved each problem at the selected Pass@k checkpoint.

## Setup

Clone the Mathlib submodule and create the Conda environment:

```bash
git submodule update --init technical/vendor/mathlib4
conda env create -f technical/environment.yml
conda activate goedel-prover-v2
cd technical/vendor/mathlib4
lake exe cache get
lake build repl
lake env repl
```

The submodule records the Lean/Mathlib source revision used by this pipeline.
The final command starts the REPL; press Ctrl-C after confirming it launches,
then return to the repository root. A site-provided Lean cache may replace
`lake exe cache get`, but the workspace and REPL must use the pinned submodule.

Run commands from the repository root so Python can import `technical.src`.
Every stage requires caller-supplied paths. In particular:

- `--model-path` points to a local model directory or a resolvable model ID;
- `--input` points to benchmark JSONL or a generated JSON array;
- `--workspace` points to the prepared Lean/Mathlib workspace;
- `--repl-command` points to the REPL executable used in that workspace;
- output options point to a result directory. Kimina, Pythagoras, and DeepSeek
  grouped generation append raw JSONL checkpoints and validate the completed
  prefix before resuming.

Kimina and Pythagoras use `technical/environment-autoregressive.yml` and the
shared `technical/pipelines/run_autoregressive.sbatch` entry point. Their exact
model and five benchmark profiles are under `technical/configs/models` and
`technical/configs/runs/{kimina,pythagoras}`. See the model-specific documents
in `docs/` for commands and version pins.

## Security boundary

Model and benchmark inputs are executable trust boundaries. Remote model code
is disabled by default. Set `model.trust_remote_code: true` only for a reviewed
model repository, and set `model.revision` to an immutable commit when loading
by model ID. A local model directory does not need a remote revision.

Lean checks execute generated source with the permissions of the verification
process. Run verification in a credential-free container or batch account with
network access disabled, a read-only benchmark/model mount, a disposable output
directory, and site-enforced CPU, memory, process, and wall-clock limits. These
Python runners apply timeouts but do not provide an operating-system sandbox.

## Common records

The first release retains three historical JSON-array filenames so readers can
match the public code to existing evaluations.

### `full_records.json`

The generation runner writes one object per attempt. Each object contains:

- `origin_problem_id` and the generation-specific `problem_id`;
- `id_maps`, including the origin and generation IDs;
- the locked `lean4_code` statement;
- the rendered `model_input` and message history;
- the raw `model_output`;
- the assembled `full_code`, or the string `"None"` when extraction or assembly
  does not produce a submission.

### `to_inference_codes.json`

This file contains the verification-facing subset of each generation record.
It keeps the same IDs, statement, prompt, messages, raw output, and assembled
source.

### `code_compilation_repl.json`

Each verification object contains `name`, `code`, `verify_time`, and
`compilation_result`. The result separates `errors`, `warnings`, `infos`, and
`sorries`, records transport failures in `system_errors`, and exposes `pass`
and `complete`. A candidate is complete only when Lean reports no errors and no
remaining sorry evidence.

The summary stage also rejects a source containing `apply?` or `exact?`, even if
the selected verifier field is true.

## Standard miniF2F and FATE pipeline

miniF2F and FATE share
[`pipelines/run_standard.sbatch`](pipelines/run_standard.sbatch). Their YAML
files choose the same standard REPL verifier; FATE retains `open` and
`set_option` lines from each source.

```bash
sbatch [site-options] technical/pipelines/run_standard.sbatch \
  technical/configs/models/goedel-prover-v2-32b.yaml \
  technical/configs/benchmarks/minif2f.yaml \
  /path/to/benchmark.jsonl \
  /path/to/output \
  /path/to/model \
  /path/to/mathlib-workspace \
  /path/to/repl
```

Use `technical/configs/benchmarks/fate.yaml` for FATE. The script reads model
sampling values from the model YAML, runs generation, verifies the assembled
sources, writes the historical JSON arrays, and produces summary tables.

The equivalent modules are:

```bash
python -m technical.src.generation.generate --help
python -m technical.src.verification.verify_standard_repl --help
python -m technical.src.evaluation.summarize_passk --help
```

## ProofNet incremental pipeline

ProofNet can contain repeated source IDs and helper declarations before the
target. The adapter creates stable unique IDs, identifies the declaration whose
body is the benchmark placeholder, and replaces that declaration by name in the
model output. It supports `theorem`, `lemma`, `def`, and `noncomputable def`.

[`pipelines/run_proofnet_incremental.sbatch`](pipelines/run_proofnet_incremental.sbatch)
runs four cumulative checkpoints. The added samples and offsets are `(1, 0)`,
`(7, 1)`, `(8, 8)`, and `(16, 16)`, yielding Pass@1, Pass@8, Pass@16, and
Pass@32.

```bash
sbatch [site-options] technical/pipelines/run_proofnet_incremental.sbatch \
  /path/to/proofnet.jsonl \
  /path/to/output \
  /path/to/model \
  /path/to/mathlib-workspace \
  technical/configs/models/goedel-prover-v2-32b.yaml \
  technical/configs/benchmarks/proofnet.yaml
```

ProofNet compiles complete temporary Lean files through `lake env lean`. The
verifier preserves the full source bytes and records stdout, stderr, exit code,
diagnostics, and timeout state.

## Putnam chunked pipeline

Putnam uses full-header REPL verification and balanced JSONL chunks.
[`pipelines/run_putnam_chunked.sbatch`](pipelines/run_putnam_chunked.sbatch)
runs one chunk and one sampling stage. Submit the checkpoints in order using
the exact `(target, offset, added samples)` triples `(1, 0, 1)`, `(8, 1, 7)`,
`(16, 8, 8)`, and `(32, 16, 16)`.

```bash
sbatch [site-options] technical/pipelines/run_putnam_chunked.sbatch \
  technical/configs/models/goedel-prover-v2-32b.yaml \
  technical/configs/benchmarks/putnam.yaml \
  /path/to/putnam.jsonl \
  /path/to/output \
  /path/to/model \
  /path/to/mathlib-workspace \
  /path/to/repl \
  8 0 7 1 8
```

The final five arguments mean 8 chunks, chunk index 0, 7 new samples,
generation offset 1, and target Pass@8. This requires the same chunk's completed
Pass@1 stage. The script verifies the new stage, rejects gaps or duplicate
generation IDs, atomically materializes cumulative Pass@8 arrays, merges all
completed chunks, and updates a machine-readable progress file. The chunk
utility also exposes direct commands:

```bash
python -m technical.src.benchmarks.putnam_chunking split --help
python -m technical.src.benchmarks.putnam_chunking merge --help
python -m technical.src.benchmarks.putnam_chunking cumulative --help
python -m technical.src.benchmarks.putnam_chunking progress --help
```

`merge` preserves input order. Callers must choose whether duplicate keys are
an error or whether the first or last record wins.

## Local checks

The unit suite uses synthetic statements and fake generation backends, so it
does not require a GPU or model weights:

```bash
python -m pytest technical/tests -v
python -m compileall -q technical/src technical/tests
bash -n technical/pipelines/*.sbatch
```

The integration smoke requires a compatible workspace and built REPL. Pytest
skips it unless you set `LEAN_TEST_WORKSPACE` and `LEAN_TEST_REPL_COMMAND`.
