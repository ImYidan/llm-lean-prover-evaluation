# Kimina-Prover-Distill-8B

This release reproduces the autoregressive evaluation path for
`AI-MO/Kimina-Prover-Distill-8B`. It deliberately contains no model weights,
tokenizer snapshot, benchmark data, generated responses, logs, or scores.

## Fixed inputs

- model and tokenizer revision: `74d328a7b1f001ab4871812582fc66d9bf70c68b`
- runtime: Python 3.12, vLLM 0.23.0, Transformers 5.12.1, PyTorch 2.11.0
- sampling: seed 0, temperature 0.6, top-p 0.95, top-k -1, 32 candidates
- miniF2F/ProofNet/Putnam: Lean `v4.9.0-rc1`, Mathlib
  `2f65ba7f1a9144b20c8e7358513548e317d26de1`, REPL
  `6592fd3bec6b3b7f8b9d8432e3f4be08451673b9`
- FATE-M/FATE-H: Lean 4.28.0, Mathlib
  `8f9d9cff6bd728b17a24e163c9402775d9e6a365`, REPL
  `527590ce2b9f3b5c4a9a1031e5b8fcfb909b9a4a`

Create the environment with
`conda env create -f technical/environment-autoregressive.yml`. Obtain the
model and benchmark JSONL separately. Each input row needs a stable
`problem_id` (or `name`) and `lean4_code`; `informal_prefix` and
`formal_statement` preserve the original Kimina prompt when present.

## Run

For `BENCHMARK` equal to `minif2f`, `proofnet`, `fate-m`, or `fate-h`:

```bash
sbatch [site-options] technical/pipelines/run_autoregressive.sbatch \
  technical/configs/models/kimina-prover-distill-8b.yaml \
  technical/configs/runs/kimina/BENCHMARK.yaml \
  /path/to/BENCHMARK.jsonl /path/to/output /path/to/model \
  /path/to/pinned/mathlib /absolute/path/to/pinned/repl
```

Putnam uses the recorded 16 balanced chunks; submit indices 0 through 15 by
adding `16 INDEX` to the command. Merge the three JSON arrays only after every
chunk has `COMPLETE`, using `python -m technical.src.benchmarks.putnam_chunking
merge --input <all chunk files> --output <merged file>` for each array, then run
`technical.src.evaluation.summarize_passk` on the merged verification and full
record arrays.

The generator stores the rendered prompt, exact raw response, finish metadata,
extraction status, and assembled source in an append-only `inference.jsonl`.
Restarting validates and resumes its complete prefix. Strict extraction uses
the last closed Lean fence; verification also tries the recorded unclosed-final-
fence recovery. Pass@k uses Lean's `complete` result and additionally rejects
`apply?` and `exact?`.

Generated Lean is untrusted executable input. Run verification without secrets
or network access and enforce site-level CPU, process, and memory limits.
