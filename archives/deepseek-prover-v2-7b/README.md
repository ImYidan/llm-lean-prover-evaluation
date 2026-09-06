# Historical DeepSeek-Prover-V2-7B evaluation archive

This directory preserves the historical evaluation implementation separately
from the shared pipeline under `technical/`. Use it to recover the original
sampling, independent-REPL verification, shard handling, and Pass@k metrics.
The snapshot contains source code and path-free provenance. The input inventory
contains hashes and schemas, not benchmark problem text. Model weights,
tokenizers, generated responses, verification results, and build caches are not
included in this snapshot.

## Contents

| Path | Purpose |
|---|---|
| `src/inference.py` | Original prompt normalization, grouped sampling, raw JSONL checkpoint, extraction and assembly |
| `src/verify.py` | One `lake env REPL` subprocess per proof, timeout/process-group cleanup and diagnostics |
| `src/summarize.py` | Original empirical prefixes and unbiased Pass@k estimator, with sample-count checks |
| `scripts/*generate.slurm`, `scripts/*finalize.slurm` | Historical four-shard miniF2F/ProofNet orchestration |
| `scripts/fate_full.slurm` | FATE-M/H generation, independent verification and summary |
| `scripts/putnambench_chunked_incremental.slurm` | Original 24-chunk Putnam schedule, stages and cumulative merges |
| `scripts/merge_inference.py` | Historical duplicate-origin disambiguation using source indices |
| `scripts/merge_records.py`, `scripts/split_jsonl.py`, `scripts/putnam_progress_summary.py` | Validated record merges, contiguous chunks and historical progress calculation |
| `scripts/common.sh` | Portable caller-supplied paths and output caches |
| `provenance/source-files.json` | Original and archived file hashes; exact list of portability adaptations |
| `provenance/input-inventory.json` | Prepared input fingerprints, sizes, row counts, splits and schemas |
| `provenance/dependencies.json`, `provenance/run-profiles.json` | Historical dependency pins and effective experiment settings |
| `requirements.txt`, `NOTICE` | Minimal runtime dependencies and attribution |

The Python inference and summary algorithms are preserved. Private default
paths in verification were replaced by environment variables. Shell entries
retain their experiment parameters while removing host-specific activation,
preflight calls and log destinations. See the per-file adaptation list.

## Model and runtime

Use Python 3.10; the historical version was 3.10.16. Install dependencies into
an isolated environment:

```bash
pip install -r archives/deepseek-prover-v2-7b/requirements.txt
```

The recorded runtime uses PyTorch 2.6.0, Transformers 4.51.3 and vLLM
0.8.5.post1. A later vLLM 0.11.0 smoke environment is not the selected full-run
environment. Match the CUDA driver and GPU architecture to compatible wheels;
the archive is not a platform-specific environment export.

Obtain `deepseek-ai/DeepSeek-Prover-V2-7B` at model **and tokenizer** revision
`a8d9e14432b2e8dd9df2a4d4e70f1ba9bc8d9b7b`. This archive intentionally retains
local-only loading and the historical `trust_remote_code=True` setting. The
inference script does not verify a local model directory's revision itself;
download that exact snapshot and supply its directory through `MODEL_DIR`.

Both the model engine and `SamplingParams` receive seed 30. The latter is
significant: setting only the engine seed is not the same sampling contract.
Other shared settings are temperature 1.0, top-p 0.95, BF16, tensor parallel 1,
GPU memory fraction 0.90, batch size 1, max-num-seqs 8 and prefix caching on.
The original code uses the tokenizer's EOS handling, with no custom stop list.

## Lean dependencies

Follow the pinned workspace build instructions in
[the DeepSeek operation guide](../../docs/deepseek-7b-cot.md#setup).
miniF2F/ProofNet/Putnam use Lean v4.9.0-rc2, Mathlib
`2f65ba7f1a9144b20c8e7358513548e317d26de1`, and REPL
`3334a97b268ecc67beb36a75787f7e831208a724`.
Change the **root Mathlib** toolchain from rc1 to rc2 before building its
dependencies; the pinned dependency's own toolchain file records rc1.

FATE-M/H use Lean v4.28.0, Mathlib
`8f9d9cff6bd728b17a24e163c9402775d9e6a365`, and REPL
`527590ce2b9f3b5c4a9a1031e5b8fcfb909b9a4a`. Exact upstream URLs are in
`provenance/dependencies.json` and the reusable Lake profile is at
`technical/lean/fate-v428/` in the repository root.

No local Mathlib/REPL source patches need to be recovered beyond the documented
root toolchain substitution: the 4,739-file Mathlib snapshot differs from its
pinned upstream only in `lean-toolchain`; the 25-file REPL snapshot matches its
pinned upstream. Build outputs are deliberately rebuilt on the destination.

## Inputs that must be retained separately

The prepared inputs have these sizes. These are dataset row counts, not scores.

| File | Rows | Use |
|---|---:|---|
| `minif2f.jsonl` | 244 | miniF2F test |
| `proofnet.jsonl` | 120 | Initial ProofNet portion |
| `proofnet_remaining66.jsonl` | 66 | ProofNet continuation |
| `proofnet_test.jsonl` | 186 | Combined ProofNet test input |
| `putnambench_lean.jsonl` | 672 | Putnam, 24 contiguous chunks |
| `fate_m.jsonl` | 150 | FATE-M |
| `fate_h.jsonl` | 100 | FATE-H |

Retain these exact prepared files outside an expiring server, or preserve a
fully specified conversion recipe and upstream data revisions. The original
conversion recipes were not available in this source snapshot, so the code and
fingerprints alone do not reconstruct these input bytes. Check each file's
SHA-256 against `provenance/input-inventory.json`. Keep row order and all
`lean4_code` headers; a benchmark name alone does not identify the input.

ProofNet includes repeated source IDs. The historical 120-row and 66-row parts
each support the original four-shard generation. Passing the combined 186-row
file directly to the unchanged four-shard generator causes within-shard ID
collisions. Keep the two historical generation jobs separate; do not silently
deduplicate the benchmark down to fewer problems.

## Run environment

Activate the chosen Python environment and set absolute caller paths, then
change into this directory:

```bash
export DATA_ROOT=/path/to/prepared-inputs
export OUTPUT_ROOT=/path/to/writable-runs
export MODEL_DIR=/path/to/pinned-model-snapshot
export DEEPSEEK_LEAN_WORKSPACE=/path/to/pinned-lean-workspace
export DEEPSEEK_LAKE_PATH=/path/to/lake
export DEEPSEEK_REPL_PATH=/path/to/built-repl
cd archives/deepseek-prover-v2-7b
```

Submit from this archive directory, as shown below. Entries resolve their shared
environment from Slurm's submission directory, not its temporary script copy.
If submitting from elsewhere, export `ARCHIVE_ROOT` as this directory's absolute
path first. Replace `[site-options]` with your scheduler options or omit it.

For rc2, the REPL is normally under
`<workspace>/.lake/packages/REPL/.lake/build/bin/repl`. Supply the actual path
for FATE; its package directory may use a different letter case. The verifier
always launches `lake env <repl>` with the workspace as its working directory.

### miniF2F

Submit the four generation shards with a chosen `RUN_ID`, then submit
finalization only after all four complete successfully:

```bash
export RUN_ID=minif2f_run
sbatch [site-options] scripts/minif2f_generate.slurm
# After all four shards finish:
sbatch [site-options] scripts/minif2f_finalize.slurm
```

The generation script uses max-model-len 32768 and max-new-tokens 8192;
finalization checks 244 problems times 32 candidates before verifying and
summarizing Pass@1/8/16/32. Scheduler resources and log paths are caller policy.

### ProofNet

Run the initial portion and continuation as separate four-shard jobs, each
followed by its own finalization. Use fresh run IDs/output roots for retries
that change input or sampling settings:

```bash
export RUN_ID=proofnet_initial
export INPUT_PATH="${DATA_ROOT}/proofnet.jsonl"
export EXPECTED_PROBLEMS=120
sbatch [site-options] scripts/proofnet_generate.slurm
# After all initial shards finish:
sbatch [site-options] scripts/proofnet_finalize.slurm

export RUN_ID=proofnet_continuation
export INPUT_PATH="${DATA_ROOT}/proofnet_remaining66.jsonl"
export EXPECTED_PROBLEMS=66
sbatch [site-options] scripts/proofnet_generate.slurm
# After all continuation shards finish:
sbatch [site-options] scripts/proofnet_finalize.slurm

python scripts/merge_records.py \
  --input-format json --output-format json \
  --expected-problems 186 --expected-samples 32 \
  --output-path "${OUTPUT_ROOT}/proofnet_combined/verification.json" \
  "${OUTPUT_ROOT}/proofnet_proofnet_initial/verification.json" \
  "${OUTPUT_ROOT}/proofnet_proofnet_continuation/verification.json"
python src/summarize.py \
  --verification-path "${OUTPUT_ROOT}/proofnet_combined/verification.json" \
  --output-dir "${OUTPUT_ROOT}/proofnet_combined/summary" --ks 1,8,16,32
```

Each part uses max-model-len 40960 and max-new-tokens 32768. Its finalizer
normalizes repeated origin IDs using the original source index before merge.
The combined merge must contain exactly 186 distinct origins and 32 samples
each; a mismatch is an error, not permission to drop rows.

### Putnam

```bash
export RUN_ID=putnam_run
export CHUNKS=24
sbatch [site-options] scripts/putnambench_chunked_incremental.slurm
```

The retained array spans indices 0 through 23. Each chunk samples additions
1/7/8/16 at offsets 0/1/8/16, verifies them, checks cumulative IDs and writes
Pass@1/8/16/32 summaries. `putnam_progress_summary.py` describes partial runs
too: a progress file is not proof that all 24 chunks completed. To produce a
global metric, pass all 24 completed `cumulative_pass@32/verification.json`
files to `merge_records.py` with `--expected-problems 672 --expected-samples 32`,
then pass the merged file to `src/summarize.py`. The full list is mandatory.

### FATE-M and FATE-H

Set the workspace and REPL variables to the v4.28.0 profile first:

```bash
DATA_PATH="${DATA_ROOT}/fate_m.jsonl" DATASET_NAME=fate_m \
  sbatch [site-options] scripts/fate_full.slurm
DATA_PATH="${DATA_ROOT}/fate_h.jsonl" DATASET_NAME=fate_h \
  sbatch [site-options] scripts/fate_full.slurm
```

Both runs use max-model-len 32768, max-new-tokens 8192, 32 candidates,
16 verification workers and a 4000-second proof timeout.

## Historical behavior and limitations

`inference.jsonl` preserves raw responses, rendered prompts, source indices,
generation indices, finish/stop reasons and extraction failures. The historical
prompt truncates the input after its first `:= by` and appends `sorry`, even
when the full input includes trailing newlines. Extraction selects the final
Lean fence and grafts its proof body onto the benchmark prefix.

The archived verifier creates a separate process per candidate, shuffles with
seed 30, kills the subprocess group on timeout, and records process/parse errors
separately. The summarizer checks complete generation prefixes and reports both
empirical and unbiased Pass@k. Its exclusion policy includes `apply?`, `exact?`,
`sorry` and `admit`.

The archived resume implementation skips existing IDs and has weaker corruption
and configuration-drift checks than the shared pipeline. Reuse only a matching,
intact run directory; retain raw results before repairing interrupted outputs.
Do not combine this archive's historical records with the shared pipeline's
normalized JSON schemas without an explicit conversion.

Exact generated text is not guaranteed across different GPU types, CUDA stacks
or scheduling changes. CPU checks cannot validate GPU generation or a clean
Lean rebuild. Execute a small generation/verification run on the destination
machine before launching the complete benchmark.

Generated Lean is executable input. Verify in an isolated, credential-free
environment with network access disabled and process/memory limits. Runtime
manifests and diagnostics can contain the caller's paths; review them before
publishing any result artifact.
